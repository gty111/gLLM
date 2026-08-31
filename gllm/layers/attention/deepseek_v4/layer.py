"""DeepSeek-V4 sparse attention: the packed, paged serving path.

One padded batch per phase -- decode rows then prefill rows -- over the
shared KV page table and the request-owned compressor-state arena. The
token-at-a-time numerical oracles this is verified against live in
:mod:`gllm.layers.attention.deepseek_v4.reference`.
"""

from __future__ import annotations

import torch

from gllm.distributed.parallel_state import (
    get_tp_rank,
    get_tp_size,
    tensor_model_parallel_all_reduce,
)
from gllm.layers.attention.deepseek_v4.compressor import DeepseekV4Compressor
from gllm.layers.attention.deepseek_v4.indexer import (
    DeepseekV4Indexer,
    indexer_scores,
)
from gllm.layers.ops.deepseek_v4 import scatter_rows_where
from gllm.layers.attention.deepseek_v4.ops import (
    precompute_rope_frequencies,
    serving_max_length,
    sparse_attention_fused,
)
from gllm.layers.attention.deepseek_v4.projection import (
    DeepseekV4AttentionProjections,
)
from gllm.layers.attention.deepseek_v4.reference import (
    DeepseekV4AttentionReference,
)

try:
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
except ImportError:  # pragma: no cover - optional backend dependency
    trtllm_batch_decode_sparse_mla_dsv4 = None


# Both fused kernels this layer dispatches to -- SGLang's FlashMLA sparse
# prefill and TRT-LLM-GEN's DSV4 sparse decode -- are specialized for the V4
# latent width.  A config that disagrees cannot be served, only referenced.
_SPARSE_MLA_HEAD_DIM = 512

# The TRT-LLM-GEN decode kernel reads a fixed-width compressed pool per
# request.  Its width is the indexer's ``index_topk``: the indexer selects at
# most that many compressed rows, and a non-indexed (C128) layer may therefore
# never have more candidate rows than the pool can hold.
_DEFAULT_COMPRESSED_POOL = 512


class DeepseekV4Attention(DeepseekV4AttentionReference, torch.nn.Module):
    """V4 sparse attention over the paged KV / compressor-state arenas.

    The token-at-a-time oracles the packed paths are verified against are
    inherited from :class:`DeepseekV4AttentionReference`; nothing in this class
    calls them.
    """

    def __init__(self, layer_id: int, config) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.window_size = getattr(
            config, "window_size", getattr(config, "sliding_window", 128)
        )
        compress_ratios = getattr(config, "compress_ratios", None)
        if compress_ratios is not None:
            self.compress_ratio = compress_ratios[layer_id]
        else:
            layer_type = config.layer_types[layer_id]
            self.compress_ratio = getattr(config, "compress_rates", {}).get(
                layer_type, 0
            )
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.max_sequence_length = serving_max_length(config)
        self.projections = DeepseekV4AttentionProjections(config)

        tp_size = get_tp_size()
        tp_rank = get_tp_rank()
        local_heads = config.num_attention_heads // tp_size
        self.attn_sink = torch.nn.Parameter(
            torch.empty(local_heads, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
        self._sink_slice = slice(tp_rank * local_heads, (tp_rank + 1) * local_heads)

        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(
                config.hidden_size,
                self.head_dim,
                self.rope_dim,
                self.compress_ratio,
                norm_eps=config.rms_norm_eps,
                rotate=False,
            )
        else:
            self.compressor = None
        self.indexer = DeepseekV4Indexer(config) if self.compress_ratio == 4 else None
        self.compressed_pool = int(
            getattr(config, "index_topk", _DEFAULT_COMPRESSED_POOL)
        )
        if self.compressor is not None and self.indexer is None:
            # A C128 layer has no indexer to rank candidates, so it feeds the
            # kernel its compressed rows directly.  Silently keeping only the
            # first ``compressed_pool`` of them would drop the most recent
            # history -- exactly the rows that matter -- so refuse instead.
            max_rows = self.max_sequence_length // self.compress_ratio
            if max_rows > self.compressed_pool:
                raise ValueError(
                    f"DeepSeek-V4 layer {layer_id} (compress_ratio="
                    f"{self.compress_ratio}) would produce {max_rows} "
                    f"compressed rows at the configured serving length "
                    f"{self.max_sequence_length}, which exceeds the "
                    f"{self.compressed_pool}-row decode pool. Lower "
                    f"--model-max-length to "
                    f"{self.compressed_pool * self.compress_ratio} or less."
                )

        all_rope_scaling = getattr(config, "rope_scaling", None) or {}
        if "main" in all_rope_scaling or "compress" in all_rope_scaling:
            rope_scaling = all_rope_scaling[
                "compress" if self.compress_ratio else "main"
            ]
        else:
            rope_scaling = all_rope_scaling
        rope_base = rope_scaling.get(
            "rope_theta",
            config.compress_rope_theta if self.compress_ratio else config.rope_theta,
        )
        original_length = (
            getattr(
                config,
                "original_seq_len",
                rope_scaling.get("original_max_position_embeddings", 0),
            )
            if self.compress_ratio
            else 0
        )
        frequencies = precompute_rope_frequencies(
            self.rope_dim,
            self.max_sequence_length,
            original_sequence_length=original_length,
            base=rope_base,
            factor=rope_scaling.get("factor", 1.0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
            device="cuda",
        )
        self.register_buffer("frequencies", frequencies, persistent=False)


    def forward_paged(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
    ) -> torch.Tensor:
        """Serve one packed batch: fused decode rows first, then prefill rows.

        There is deliberately no fallback here.  This used to drop to
        :meth:`forward_paged_reference` -- a Python token-at-a-time loop --
        whenever a precondition was unmet, which turned a configuration mistake
        into a silent ~100x slowdown rather than an error.
        """
        segment = input_data.memory_manager.segment
        if segment is None:
            # Startup activation profiling runs before the cache arena is
            # sized, so there are no pages to read yet.
            return self._forward_profile(input_data, hidden_states)
        if self.head_dim != _SPARSE_MLA_HEAD_DIM:
            raise ValueError(
                "DeepSeek-V4 serving requires the fused sparse-MLA kernels, "
                f"which are specialized for head_dim={_SPARSE_MLA_HEAD_DIM}; "
                f"this config has head_dim={self.head_dim}."
            )
        meta = getattr(input_data, "metadata", None)
        if meta is None:
            raise RuntimeError(
                "DeepSeek-V4 paged attention requires forward metadata"
            )

        num_decodes = meta.num_decode_tokens
        outputs = []
        if num_decodes:
            outputs.append(
                self._forward_paged_decode_batch(
                    input_data,
                    hidden_states[:num_decodes],
                    local_layer_id=local_layer_id,
                )
            )
        if meta.num_prefills:
            outputs.append(
                self._forward_paged_prefill_batch(
                    input_data,
                    hidden_states[num_decodes:],
                    local_layer_id=local_layer_id,
                )
            )
        if not outputs:
            return hidden_states.new_empty((0, self.projections.hidden_size))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def _forward_profile(
        self, input_data, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Learned-projection footprint for the startup memory profile run."""
        positions = input_data.get_position().long()
        frequencies = self.frequencies.index_select(0, positions)
        _, query, _ = self.projections.prepare_q_kv(
            hidden_states.unsqueeze(0), frequencies
        )
        return self.projections.project_output(
            torch.zeros_like(query), frequencies
        ).squeeze(0)

    def _forward_paged_prefill_batch(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
    ) -> torch.Tensor:
        """Run every prefill row -- fresh or continuation -- as one batch.

        A start-at-zero prefill is just the ``context_lens == 0`` case: the
        request-owned compressor state is zero/-inf filled when its arena slot
        is allocated, and the raw-KV pool degenerates to the suffix that was
        written a few lines above.  Keeping one path means a first chunk and a
        continuation chunk can never drift apart numerically, which is the one
        bug class chunked prefill is prone to.

        Existing raw/compressed KV and recurrent compressor states are gathered
        from the request-owned arenas.  Every new suffix is projected and
        compressed in bulk, so prefill never falls back to the token-wise
        numerical oracle.
        """
        meta = input_data.metadata
        prefill = meta.prefill
        if prefill is None:
            raise ValueError("DeepSeek-V4 prefill metadata is missing")
        segment = input_data.memory_manager.segment
        state_segment = input_data.memory_manager.dsv4_state_segment
        if segment is None or state_segment is None:
            raise RuntimeError("V4 batch prefill requires KV and state arenas")

        batch = meta.num_prefills
        max_length = input_data.prefill_max_query_len
        query_start = prefill.query_start_loc
        lengths = (query_start[1:] - query_start[:-1]).long()
        # A first-chunk prefill carries no ``context_lens`` at all -- the
        # metadata builder only populates it once some row has context.
        context_lens = getattr(prefill, "context_lens", None)
        context_lens = (
            context_lens.long()
            if context_lens is not None
            else lengths.new_zeros(batch)
        )
        columns = torch.arange(
            max_length, device=hidden_states.device, dtype=torch.long
        ).unsqueeze(0)
        packed_indices = query_start[:-1].long().unsqueeze(1) + columns
        valid_tokens = columns < lengths.unsqueeze(1)
        safe_indices = torch.where(
            valid_tokens, packed_indices, packed_indices.new_zeros(())
        )
        padded_hidden = hidden_states.index_select(
            0, safe_indices.flatten()
        ).view(batch, max_length, -1)
        padded_hidden.masked_fill_(~valid_tokens.unsqueeze(-1), 0)

        positions = context_lens.unsqueeze(1) + columns
        safe_positions = torch.where(valid_tokens, positions, positions.new_zeros(()))
        frequencies = self.frequencies.index_select(
            0, safe_positions.flatten()
        ).view(batch, max_length, -1)
        q_lora, query, raw_kv = self.projections.prepare_q_kv(
            padded_hidden, frequencies
        )
        raw_kv.masked_fill_(~valid_tokens.unsqueeze(-1), 0)

        block_table = prefill.block_table
        page_size = segment.page_size
        final_lens = context_lens + lengths
        state_slots = input_data.get_recurrent_state_slot_per_seq()[
            meta.num_decodes : meta.num_decodes + batch
        ]

        window_size = self.window_size

        # Assemble the smallest KV pool that contains every query's causal
        # sliding window: the cached prefix, then this chunk.
        #
        # The ring only holds the last W positions, so the chunk's own rows
        # cannot be read back out of it -- they are spliced in from ``raw_kv``.
        # Slot ``p`` still means absolute position ``pool_start + p``, so the
        # index tables below are unchanged.
        pool_start = (context_lens - window_size + 1).clamp_min(0)
        max_raw_pool = window_size - 1 + max_length
        prefix_width = window_size - 1
        raw_pool = raw_kv.new_zeros(batch, max_raw_pool, raw_kv.shape[-1])

        if prefix_width and input_data.max_context_len:
            # The prefix spans ``min(context_len, W-1)`` rows and always lands
            # in pool slots [0, W-1).
            prefix_columns = torch.arange(
                prefix_width, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)
            prefix_positions = pool_start.unsqueeze(1) + prefix_columns
            prefix_valid = prefix_positions < context_lens.unsqueeze(1)
            prefix = state_segment.gather_window(
                local_layer_id,
                state_slots.unsqueeze(1).expand_as(prefix_positions),
                torch.where(
                    prefix_valid, prefix_positions, prefix_positions.new_zeros(())
                ),
            )
            raw_pool[:, :prefix_width] = torch.where(
                prefix_valid.unsqueeze(-1), prefix, prefix.new_zeros(())
            )

        # Store *after* the prefix read. The pool spans ``W-1+L`` positions
        # while the ring holds only ``W``, so writing this chunk first would
        # overwrite prefix rows it still needs: at W=128 a chunk starting at
        # position 130 lands on ring rows 2.. while the prefix still needs
        # rows 3..127. Only the last W rows of the chunk are worth keeping --
        # earlier ones could not be read back by any later step.
        keep = columns >= (lengths - window_size).clamp_min(0).unsqueeze(1)
        keep &= valid_tokens
        if keep.any():
            state_segment.store_window(
                local_layer_id,
                state_slots.unsqueeze(1).expand_as(keep)[keep],
                positions[keep],
                raw_kv[keep],
            )

        suffix_columns = (context_lens - pool_start).unsqueeze(1) + columns
        pool_rows = (
            torch.arange(batch, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(1)
            * max_raw_pool
            + suffix_columns
        )
        raw_pool.view(batch * max_raw_pool, -1)[
            pool_rows[valid_tokens]
        ] = raw_kv[valid_tokens]

        window_offsets = torch.arange(
            self.window_size, device=hidden_states.device, dtype=torch.long
        ).view(1, 1, -1)
        window_positions = (
            positions.unsqueeze(-1) - self.window_size + 1 + window_offsets
        )
        window_valid = (
            valid_tokens.unsqueeze(-1)
            & window_positions.ge(0)
            & window_positions.ge(pool_start.view(batch, 1, 1))
        )
        window_local = window_positions - pool_start.view(batch, 1, 1)
        window_indices_local = torch.where(
            window_valid,
            window_local,
            window_local.new_full((), -1),
        ).to(torch.int32)

        attention_kv = raw_pool
        indices = window_indices_local
        if self.compressor is not None:
            ratio = self.compress_ratio
            main_state = state_segment.gather_states(state_slots, local_layer_id)
            # The continuation primitive uses a static worst-case number of
            # groups; build the matching RoPE rows without a per-layer .item().
            max_new_groups = (max_length + 2 * ratio - 2) // ratio
            group_ids = torch.arange(
                max_new_groups, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0)
            first_group = context_lens.div(ratio, rounding_mode="floor")
            new_logical = first_group.unsqueeze(1) + group_ids
            compressed_positions = (new_logical * ratio).clamp_max(
                self.max_sequence_length - 1
            )
            compressed_frequencies = self.frequencies.index_select(
                0, compressed_positions.flatten()
            ).view(batch, max_new_groups, -1)
            new_main, main_state, new_counts = self.compressor.prefill_continue_batch(
                padded_hidden,
                compressed_frequencies,
                starts=context_lens,
                lengths=lengths,
                state=main_state,
            )
            state_segment.store_states(state_slots, local_layer_id, main_state)

            new_valid = group_ids < new_counts.unsqueeze(1)
            safe_new_logical = torch.where(
                new_valid, new_logical, new_logical.new_zeros(())
            )
            new_pages, new_rows = self._compressed_slots(
                block_table,
                safe_new_logical,
                page_size=page_size,
                ratio=ratio,
            )
            main_cache = segment.dsv4_compressed_cache[local_layer_id]
            main_cache[new_pages[new_valid], new_rows[new_valid]] = new_main[new_valid]

            max_compressed = max(1, input_data.max_seq_len // ratio)
            logical = torch.arange(
                max_compressed, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(batch, -1)
            total_counts = final_lens.div(ratio, rounding_mode="floor")
            compressed_valid = logical < total_counts.unsqueeze(1)
            safe_logical = torch.where(
                compressed_valid, logical, logical.new_zeros(())
            )
            compressed_pages, compressed_rows = self._compressed_slots(
                block_table,
                safe_logical,
                page_size=page_size,
                ratio=ratio,
            )
            main_pool = main_cache[compressed_pages, compressed_rows]
            main_pool.masked_fill_(~compressed_valid.unsqueeze(-1), 0)
            attention_kv = torch.cat([raw_pool, main_pool], dim=1)

            causal_counts = (positions + 1).div(ratio, rounding_mode="floor")
            candidate_valid = (
                logical.unsqueeze(1) < causal_counts.unsqueeze(-1)
            ) & valid_tokens.unsqueeze(-1)
            if self.indexer is not None:
                index_state = state_segment.gather_states(
                    state_slots, local_layer_id, indexer=True
                )
                new_index, index_state, index_counts = (
                    self.indexer.compressor.prefill_continue_batch(
                        padded_hidden,
                        compressed_frequencies,
                        starts=context_lens,
                        lengths=lengths,
                        state=index_state,
                    )
                )
                state_segment.store_states(
                    state_slots, local_layer_id, index_state, indexer=True
                )
                index_cache = segment.dsv4_index_cache[local_layer_id]
                index_cache[new_pages[new_valid], new_rows[new_valid]] = new_index[
                    new_valid
                ]
                index_pool = index_cache[compressed_pages, compressed_rows]
                index_pool.masked_fill_(~compressed_valid.unsqueeze(-1), 0)
                index_query, head_weights = self.indexer.prepare_query(
                    padded_hidden, q_lora, frequencies
                )
                scores = indexer_scores(index_query, index_pool, head_weights)
                if get_tp_size() > 1:
                    scores = tensor_model_parallel_all_reduce(scores)
                scores.masked_fill_(~candidate_valid, -torch.inf)
                k = min(self.indexer.topk, max_compressed)
                selected = scores.topk(k, dim=-1).indices
                selected_valid = torch.gather(candidate_valid, 2, selected)
                compressed_indices_local = torch.where(
                    selected_valid,
                    selected + max_raw_pool,
                    selected.new_full((), -1),
                ).to(torch.int32)
            else:
                compressed_indices_local = torch.where(
                    candidate_valid,
                    logical.unsqueeze(1) + max_raw_pool,
                    logical.new_full((), -1),
                ).to(torch.int32)
            indices = torch.cat([window_indices_local, compressed_indices_local], dim=-1)

        output = sparse_attention_fused(
            query,
            attention_kv,
            indices,
            self.attn_sink,
            self.softmax_scale,
        )
        projected = self.projections.project_output(output, frequencies)
        return projected[valid_tokens]

    @staticmethod
    def _compressed_slots(
        block_table: torch.Tensor,
        compressed_positions: torch.Tensor,
        *,
        page_size: int,
        ratio: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a batched logical compressed position table to arena rows."""
        safe = compressed_positions.clamp_min(0)
        end_positions = (safe + 1) * ratio - 1
        page_columns = end_positions // page_size
        first_in_page = (page_columns * page_size) // ratio
        bank_rows = safe - first_in_page
        physical_pages = block_table.gather(1, page_columns.long()).long()
        return physical_pages, bank_rows.long()

    @staticmethod
    def _decode_cache(input_data) -> dict:
        """Per-forward memo for index tensors every layer would rebuild.

        The decode index math -- sliding-window offsets, the sparse-index
        table, the compressed candidate grid and its page/row mapping --
        depends only on the batch, the window size and the compression ratio,
        never on which layer is running. Recomputing it in all 43 layers was
        the largest single source of kernels in a decode step, and because
        decode is captured as one CUDA graph the redundancy is baked in and
        replayed every step. Building each tensor once per forward shrinks the
        captured graph itself.

        The values are identical either way, so this changes no numerics.
        """
        # Keyed on the metadata *object*, which ``copy_to_input_buffer``
        # rebuilds every forward. Holding the object rather than its ``id()``
        # matters: a freed object's address can be reused, and an id match
        # against a dead object would silently serve a stale batch's indices.
        current = input_data.metadata
        cache = getattr(input_data, "_dsv4_decode_cache", None)
        if cache is None or getattr(input_data, "_dsv4_decode_token", None) is not current:
            cache = {}
            input_data._dsv4_decode_cache = cache
            input_data._dsv4_decode_token = current
        return cache

    def _forward_paged_decode_batch(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
    ) -> torch.Tensor:
        """Run the complete heterogeneous one-token decode batch once.

        The scheduler places decode rows first.  Every input below is therefore
        a slice of the canonical ``InputData`` tensors; no request objects,
        position grouping, or per-row GPU launches participate in this path.
        """
        meta = input_data.metadata
        decode = meta.decode
        if decode is None or hidden_states.shape[0] != meta.num_decode_tokens:
            raise ValueError("invalid DeepSeek-V4 decode batch metadata")
        batch = hidden_states.shape[0]
        if batch == 0:
            return hidden_states.new_empty((0, self.projections.hidden_size))

        manager = input_data.memory_manager
        segment = manager.segment
        state_segment = manager.dsv4_state_segment
        if segment is None or state_segment is None:
            raise RuntimeError("V4 batch decode requires KV and state arenas")

        cache = self._decode_cache(input_data)
        if "positions" not in cache:
            cache["positions"] = input_data.get_position()[:batch].long()
        positions = cache["positions"]
        # The RoPE table depends only on ``compress_ratio``, so all layers
        # sharing one gather the same rows -- as the anchor memo below already
        # assumes.
        freq_key = ("freq", self.compress_ratio)
        if freq_key not in cache:
            cache[freq_key] = self.frequencies.index_select(
                0, positions
            ).unsqueeze(1)
        frequencies = cache[freq_key]
        states_3d = hidden_states.unsqueeze(1)
        q_lora, query, raw_kv = self.projections.prepare_q_kv(
            states_3d, frequencies
        )
        state_slots = input_data.get_recurrent_state_slot_per_seq()[:batch]
        state_segment.store_window(
            local_layer_id, state_slots, positions, raw_kv[:, 0]
        )

        block_table = decode.block_table
        page_size = segment.page_size
        window_key = ("window", self.window_size)
        if window_key not in cache:
            window_columns = torch.arange(
                self.window_size,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)
            window_lengths = (positions + 1).clamp_max(self.window_size)
            # FlashInfer's DSV4 ABI follows ``window_indices``: short histories
            # occupy the prefix [0, seq_len), full windows are chronological.
            window_positions = (
                positions[:, None] + 1 - window_lengths[:, None] + window_columns
            )
            window_valid = window_columns < window_lengths[:, None]
            cache[window_key] = (
                window_valid,
                torch.where(
                    window_valid, window_positions, window_positions.new_zeros(())
                ),
            )
        window_valid, safe_window = cache[window_key]
        window = state_segment.gather_window(
            local_layer_id,
            state_slots.unsqueeze(1).expand_as(safe_window),
            safe_window,
        )
        window = window.masked_fill(~window_valid.unsqueeze(-1), 0)

        selected_count = positions.new_zeros(batch)
        selected_main = hidden_states.new_zeros(
            batch, self.compressed_pool, self.head_dim, dtype=torch.bfloat16
        )

        if self.compressor is not None:
            ratio = self.compress_ratio
            count_key = ("count", ratio)
            if count_key not in cache:
                cache[count_key] = (positions + 1) // ratio
            compressed_count = cache[count_key]
            # Eager decode scores only the compressed rows that can exist in
            # this batch. A full CUDA graph, however, is captured with length-2
            # dummy sequences and replayed for arbitrary positions. Capturing
            # the eager bound would permanently bake one candidate into the
            # graph and silently drop long-context rows on replay. Use the
            # model-wide static bound only while capturing; replay then reads
            # current positions/block tables from the canonical InputData
            # device buffers and masks rows beyond ``compressed_count``.
            max_compressed = (
                max(1, self.max_sequence_length // ratio)
                if torch.cuda.is_current_stream_capturing()
                else max(1, (input_data.max_seq_len + ratio - 1) // ratio)
            )
            # Every layer sharing this ratio builds the same candidate grid and
            # resolves it to the same pages; 21 C4 layers and 20 C128 layers
            # each did that independently.
            grid_key = ("grid", ratio, max_compressed)
            if grid_key not in cache:
                logical = torch.arange(
                    max_compressed,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch, -1)
                # A full decode graph keeps ``max_compressed`` static so one
                # graph replays at every context length. Page-table columns
                # past the current request length are unspecified; do not
                # dereference them before the score mask is applied. Map every
                # inactive candidate to logical row 0, which is backed by a
                # real page (or the reserved dummy page for padded rows).
                valid = logical < compressed_count[:, None]
                cache[grid_key] = (
                    logical,
                    valid,
                    *self._compressed_slots(
                        block_table,
                        torch.where(valid, logical, logical.new_zeros(())),
                        page_size=page_size,
                        ratio=ratio,
                    ),
                )
            logical, valid, physical_pages, bank_rows = cache[grid_key]
            main_cache = segment.dsv4_compressed_cache[local_layer_id]

            main_state = state_segment.gather_states(
                state_slots, local_layer_id
            )
            anchor_key = ("anchor", ratio)
            if anchor_key not in cache:
                cache[anchor_key] = self.frequencies.index_select(
                    0, (positions + 1 - ratio).clamp_min(0)
                ).unsqueeze(1)
            compressed_frequencies = cache[anchor_key]
            new_main, boundary = self.compressor.decode_batch(
                states_3d,
                compressed_frequencies,
                positions=positions,
                state=main_state,
            )
            state_segment.store_states(state_slots, local_layer_id, main_state)
            dst_key = ("dst", ratio)
            if dst_key not in cache:
                cache[dst_key] = self._compressed_slots(
                    block_table,
                    (compressed_count - 1).clamp_min(0).unsqueeze(1),
                    page_size=page_size,
                    ratio=ratio,
                )
            dst_page, dst_row = cache[dst_key]
            scatter_rows_where(
                main_cache, dst_page[:, 0], dst_row[:, 0], new_main[:, 0], boundary
            )

            if self.indexer is not None:
                index_cache = segment.dsv4_index_cache[local_layer_id]
                index_state = state_segment.gather_states(
                    state_slots, local_layer_id, indexer=True
                )
                index_query, head_weights = self.indexer.prepare_query(
                    states_3d, q_lora, frequencies
                )
                new_index, index_boundary = self.indexer.compressor.decode_batch(
                    states_3d,
                    compressed_frequencies,
                    positions=positions,
                    state=index_state,
                )
                state_segment.store_states(
                    state_slots, local_layer_id, index_state, indexer=True
                )
                scatter_rows_where(
                    index_cache,
                    dst_page[:, 0],
                    dst_row[:, 0],
                    new_index[:, 0],
                    index_boundary,
                )

                index_kv = index_cache[physical_pages, bank_rows]
                scores = indexer_scores(
                    index_query, index_kv, head_weights
                )[:, 0]
                if get_tp_size() > 1:
                    scores = tensor_model_parallel_all_reduce(scores)
                scores.masked_fill_(~valid, -torch.inf)
                k = min(self.indexer.topk, max_compressed)
                selected = scores.topk(k, dim=-1).indices
                selected = torch.where(
                    selected < compressed_count[:, None],
                    selected,
                    selected.new_full((), -1),
                )
            else:
                # No indexer: the candidate prefix is the same for every layer
                # with this ratio, and so is the page mapping it resolves to.
                k = min(self.compressed_pool, max_compressed)
                dense_key = ("dense", ratio, k)
                if dense_key not in cache:
                    prefix = logical[:, :k]
                    dense_selected = torch.where(
                        prefix < compressed_count[:, None],
                        prefix,
                        prefix.new_full((), -1),
                    )
                    cache[dense_key] = (
                        dense_selected,
                        *self._compressed_slots(
                            block_table,
                            dense_selected.clamp_min(0),
                            page_size=page_size,
                            ratio=ratio,
                        ),
                    )
                selected, dense_pages, dense_rows = cache[dense_key]

            selected_count = compressed_count.clamp_max(k)
            if self.indexer is not None:
                selected_pages, selected_rows = self._compressed_slots(
                    block_table,
                    selected.clamp_min(0),
                    page_size=page_size,
                    ratio=ratio,
                )
            else:
                selected_pages, selected_rows = dense_pages, dense_rows
            gathered = main_cache[selected_pages, selected_rows]
            gathered.masked_fill_(selected.lt(0).unsqueeze(-1), 0)
            selected_main[:, :k] = gathered

        # The TRT-LLM-GEN DSV4 kernel requires contiguous pools.  Gather only
        # the sliding window and at most ``compressed_pool`` selected
        # compressed rows, then issue one kernel for the whole batch.
        sparse_key = ("sparse", self.window_size, self.compressed_pool)
        if sparse_key not in cache:
            rows = torch.arange(
                batch, device=hidden_states.device, dtype=torch.int32
            ).unsqueeze(1)
            cache[sparse_key] = torch.cat(
                [
                    rows * self.window_size
                    + torch.arange(
                        self.window_size,
                        device=hidden_states.device,
                        dtype=torch.int32,
                    ).unsqueeze(0),
                    rows * self.compressed_pool
                    + torch.arange(
                        self.compressed_pool,
                        device=hidden_states.device,
                        dtype=torch.int32,
                    ),
                ],
                dim=1,
            )
        sparse_indices = cache[sparse_key]
        sparse_lens = (selected_count + self.window_size).to(torch.int32)

        if (
            self.window_size == 128
            and trtllm_batch_decode_sparse_mla_dsv4 is not None
        ):
            workspace = input_data.workspace.view(torch.uint8)
            output = trtllm_batch_decode_sparse_mla_dsv4(
                query=query,
                swa_kv_cache=window.reshape(
                    batch * self.window_size, 1, 1, self.head_dim
                ).contiguous(),
                workspace_buffer=workspace,
                sparse_indices=sparse_indices.contiguous(),
                compressed_kv_cache=selected_main.reshape(
                    batch * self.compressed_pool, 1, 1, self.head_dim
                ).contiguous(),
                sparse_topk_lens=sparse_lens,
                seq_lens=decode.seq_lens,
                bmm1_scale=self.softmax_scale,
                sinks=self.attn_sink,
            )
        else:
            kv = torch.cat([window, selected_main], dim=1)
            valid_window = torch.arange(
                self.window_size, device=hidden_states.device
            ).unsqueeze(0) < window_lengths[:, None]
            valid_compressed = torch.arange(
                self.compressed_pool, device=hidden_states.device
            ).unsqueeze(0) < selected_count[:, None]
            local_indices = torch.arange(
                self.window_size + self.compressed_pool,
                device=hidden_states.device,
                dtype=torch.int32,
            ).view(1, 1, -1).expand(batch, 1, -1)
            valid_indices = torch.cat(
                [valid_window, valid_compressed], dim=1
            ).unsqueeze(1)
            local_indices = torch.where(
                valid_indices, local_indices, local_indices.new_full((), -1)
            )
            output = sparse_attention_fused(
                query,
                kv,
                local_indices,
                self.attn_sink,
                self.softmax_scale,
            )
        return self.projections.project_output(output, frequencies)[:, 0]


__all__ = ["DeepseekV4Attention"]
