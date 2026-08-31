"""Token-at-a-time DeepSeek-V4 attention oracles.

Every method here advances exactly one position through the checkpoint's
official update order. They exist to pin the numerics down, not to serve
traffic: the packed prefill/decode paths in
:mod:`gllm.layers.attention.deepseek_v4.layer` are validated against them.

They are deliberately unreachable from ``forward_paged``. An earlier revision
fell back to :meth:`forward_paged_reference` whenever the fused kernel's
preconditions were unmet, which turned a configuration mistake into a silent
~100x slowdown instead of an error.
"""

from __future__ import annotations

import torch

from gllm.layers.attention.deepseek_v4.cache import (
    DeepseekV4AttentionCache,
    _PreparedDecode,
    _PreparedPrefill,
)
from gllm.layers.attention.deepseek_v4.compressor import make_compressor_state
from gllm.layers.attention.deepseek_v4.ops import (
    compressed_indices,
    sparse_attention_reference,
    window_indices,
)


class DeepseekV4AttentionReference:
    """Oracle half of :class:`DeepseekV4Attention`, kept out of the hot path."""

    def make_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> DeepseekV4AttentionCache:
        compressed_length = (
            self.max_sequence_length // self.compress_ratio
            if self.compress_ratio
            else 0
        )
        compressed = (
            torch.zeros(
                batch_size,
                compressed_length,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            if self.compress_ratio
            else None
        )
        index_compressed = (
            torch.zeros(
                batch_size,
                compressed_length,
                self.indexer.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            if self.indexer is not None
            else None
        )
        return DeepseekV4AttentionCache(
            window=torch.zeros(
                batch_size,
                self.window_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            compressed=compressed,
            index_compressed=index_compressed,
            compressor_state=(
                make_compressor_state(
                    batch_size,
                    self.compress_ratio,
                    self.head_dim,
                    device=device,
                )
                if self.compressor is not None
                else None
            ),
            indexer_state=(
                make_compressor_state(
                    batch_size,
                    4,
                    self.indexer.head_dim,
                    device=device,
                )
                if self.indexer is not None
                else None
            ),
        )
    def forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run a start-position-zero, non-chunked prefill reference."""
        output, _ = self.forward_prefill_with_cache(hidden_states)
        return output
    def forward_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        cache: DeepseekV4AttentionCache | None = None,
    ) -> tuple[torch.Tensor, DeepseekV4AttentionCache]:
        """Run prefill and leave an official-layout cache ready for decode."""
        prepared = self._prepare_prefill_with_cache(hidden_states, cache)
        output = sparse_attention_reference(
            prepared.query,
            prepared.attention_kv,
            prepared.indices,
            self.attn_sink,
            self.softmax_scale,
        )
        return (
            self.projections.project_output(output, prepared.frequencies),
            prepared.cache,
        )
    def _prepare_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        cache: DeepseekV4AttentionCache | None = None,
    ) -> _PreparedPrefill:
        """Prepare a complete start-at-zero prefill without running attention."""
        if hidden_states.ndim != 3:
            raise ValueError("V4 reference prefill expects [B,S,H]")
        batch, sequence_length, _ = hidden_states.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError("V4 prefill exceeds configured maximum sequence length")
        if cache is None:
            cache = self.make_cache(batch, device=hidden_states.device)
        if cache.window.shape[0] != batch:
            raise ValueError("V4 cache batch size does not match prefill")
        frequencies = self.frequencies[:sequence_length]
        q_lora, query, kv = self.projections.prepare_q_kv(hidden_states, frequencies)
        window_count = min(sequence_length, self.window_size)
        window_positions = torch.arange(
            sequence_length - window_count,
            sequence_length,
            device=hidden_states.device,
        )
        cache.window[:, window_positions % self.window_size] = kv[:, -window_count:]
        indices = window_indices(
            self.window_size,
            batch,
            sequence_length,
            0,
            device=hidden_states.device,
        )
        attention_kv = kv
        compressed_kv = None
        index_kv = None

        if self.compressor is not None:
            compressed_cutoff = (
                sequence_length // self.compress_ratio * self.compress_ratio
            )
            compressed_frequencies = frequencies[
                0:compressed_cutoff:self.compress_ratio
            ]
            compressed_kv, compressor_state = self.compressor.prefill(
                hidden_states, compressed_frequencies
            )
            cache.compressor_state = compressor_state
            if self.indexer is not None:
                compressed_topk, index_kv, indexer_state = self.indexer.prefill(
                    hidden_states,
                    q_lora,
                    frequencies,
                    frequencies[0:compressed_cutoff:4],
                    offset=sequence_length,
                )
                cache.indexer_state = indexer_state
                if index_kv is not None:
                    cache.index_compressed[:, : index_kv.shape[1]].copy_(index_kv)
            if compressed_kv is not None:
                cache.compressed[:, : compressed_kv.shape[1]].copy_(compressed_kv)
                offset = sequence_length
                if self.indexer is not None:
                    if index_kv.shape[1] != compressed_kv.shape[1]:
                        raise RuntimeError("V4 C4 attention/index caches diverged")
                else:
                    compressed_topk = compressed_indices(
                        self.compress_ratio,
                        batch,
                        sequence_length,
                        0,
                        offset,
                        device=hidden_states.device,
                    )
                attention_kv = torch.cat([kv, compressed_kv], dim=1)
                indices = torch.cat([indices, compressed_topk], dim=-1)

        return _PreparedPrefill(
            query=query,
            attention_kv=attention_kv,
            indices=indices,
            frequencies=frequencies,
            raw_kv=kv,
            compressed_kv=compressed_kv,
            index_kv=index_kv,
            cache=cache,
        )
    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        *,
        position: int,
        cache: DeepseekV4AttentionCache,
    ) -> torch.Tensor:
        """Run one official-order online decode update and attention step."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("V4 reference decode expects [B,1,H]")
        if not 0 <= position < self.max_sequence_length:
            raise ValueError(f"V4 decode position is out of range: {position}")
        prepared = self._prepare_decode(
            hidden_states, position=position, cache=cache
        )
        output = sparse_attention_reference(
            prepared.query,
            prepared.attention_kv,
            prepared.indices,
            self.attn_sink,
            self.softmax_scale,
        )
        return self.projections.project_output(output, prepared.frequency)
    def _prepare_decode(
        self,
        hidden_states: torch.Tensor,
        *,
        position: int,
        cache: DeepseekV4AttentionCache,
    ) -> _PreparedDecode:
        """Advance one token's learned cache/state without running attention."""
        frequency = self.frequencies[position : position + 1]
        q_lora, query, kv = self.projections.prepare_q_kv(
            hidden_states, frequency
        )
        cache.window[:, position % self.window_size].copy_(kv[:, 0])
        indices = window_indices(
            self.window_size,
            hidden_states.shape[0],
            1,
            position,
            device=hidden_states.device,
        )

        if self.compressor is not None:
            offset = self.window_size
            compressed_anchor = position + 1 - self.compress_ratio
            compressed_frequency = self.frequencies[
                max(compressed_anchor, 0) : max(compressed_anchor, 0) + 1
            ]
            if self.indexer is not None:
                compressed_topk, _ = self.indexer.decode(
                    hidden_states,
                    q_lora,
                    frequency,
                    compressed_frequency,
                    cache.index_compressed,
                    position=position,
                    offset=offset,
                    state=cache.indexer_state,
                )
            else:
                compressed_topk = compressed_indices(
                    self.compress_ratio,
                    hidden_states.shape[0],
                    1,
                    position,
                    offset,
                    device=hidden_states.device,
                )
            indices = torch.cat([indices, compressed_topk], dim=-1)
            compressed = self.compressor.decode(
                hidden_states,
                compressed_frequency,
                position=position,
                state=cache.compressor_state,
            )
            compressed_count = (position + 1) // self.compress_ratio
            if compressed is not None:
                cache.compressed[
                    :, compressed_count - 1 : compressed_count
                ].copy_(compressed)
            attention_kv = torch.cat(
                [cache.window, cache.compressed[:, :compressed_count]], dim=1
            )
        else:
            attention_kv = cache.window
        return _PreparedDecode(
            query=query,
            attention_kv=attention_kv,
            indices=indices,
            frequency=frequency,
            raw_kv=kv,
        )
    def forward_paged_reference(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
    ) -> torch.Tensor:
        """Correctness-first packed prefill/decode over V4 paged cache banks.

        Tokens are intentionally advanced one at a time so chunked prefill,
        mixed request lengths and decode all share exactly the verified online
        update order. Optimized backends may replace this loop while comparing
        their layer output against it.
        """
        if hidden_states.ndim != 2:
            raise ValueError("V4 paged attention expects packed [N,H] states")
        manager = input_data.memory_manager
        segment = manager.segment
        state_segment = manager.dsv4_state_segment
        if segment is None or state_segment is None or (
            segment.dsv4_kv_cache_config is None
        ):
            raise RuntimeError("V4 paged attention requires KV and state arenas")

        starts = input_data.query_start_loc_cpu.tolist()
        slot_mapping = input_data.get_slot_mapping()
        outputs = [
            self._forward_paged_reference_row(
                input_data,
                hidden_states,
                local_layer_id=local_layer_id,
                row=row,
                starts=starts,
                slot_mapping=slot_mapping,
            )
            for row in range(len(input_data.seqs))
        ]
        if not outputs:
            return hidden_states.new_empty((0, self.projections.hidden_size))
        return torch.cat(outputs, dim=0)
    def _forward_paged_reference_row(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
        row: int,
        starts: list[int],
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one packed request with the token-wise numerical oracle."""
        segment = input_data.memory_manager.segment
        state_segment = input_data.memory_manager.dsv4_state_segment
        seq = input_data.seqs[row]
        token_start, token_end = starts[row], starts[row + 1]
        state_slot = seq.recurrent_state_slot
        if state_slot is None:
            raise RuntimeError(f"V4 sequence {seq.seq_id} has no state slot")
        main_state = (
            state_segment.state_view(state_slot, local_layer_id)
            if self.compressor is not None
            else None
        )
        index_state = (
            state_segment.state_view(state_slot, local_layer_id, indexer=True)
            if self.indexer is not None
            else None
        )
        outputs = []
        for packed_index in range(token_start, token_end):
            position = seq.computed_token_num + packed_index - token_start
            window_start = max(0, position - self.window_size + 1)
            previous_positions = list(range(window_start, position))
            window = torch.zeros(
                1,
                self.window_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            if previous_positions:
                previous_index = torch.as_tensor(
                    previous_positions,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                previous = state_segment.gather_window(
                    local_layer_id,
                    previous_index.new_full(previous_index.shape, state_slot),
                    previous_index,
                )
                window[0, previous_index % self.window_size] = previous

            compressed_count = (
                (position + 1) // self.compress_ratio if self.compress_ratio else 0
            )
            stored_compressed_count = (
                position // self.compress_ratio if self.compress_ratio else 0
            )
            previous_compressed = list(range(stored_compressed_count))
            compressed = (
                torch.zeros(
                    1,
                    compressed_count,
                    self.head_dim,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                if self.compressor is not None
                else None
            )
            index_compressed = (
                torch.zeros(
                    1,
                    compressed_count,
                    self.indexer.head_dim,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                if self.indexer is not None
                else None
            )
            if previous_compressed:
                compressed[:, :stored_compressed_count].copy_(
                    segment.gather_dsv4_compressed(
                        local_layer_id, seq.page_table, previous_compressed
                    ).unsqueeze(0)
                )
                if self.indexer is not None:
                    index_compressed[:, :stored_compressed_count].copy_(
                        segment.gather_dsv4_compressed(
                            local_layer_id,
                            seq.page_table,
                            previous_compressed,
                            indexer=True,
                        ).unsqueeze(0)
                    )
            cache = DeepseekV4AttentionCache(
                window=window,
                compressed=compressed,
                index_compressed=index_compressed,
                compressor_state=main_state,
                indexer_state=index_state,
            )
            step_output = self.forward_decode(
                hidden_states[packed_index : packed_index + 1].unsqueeze(0),
                position=position,
                cache=cache,
            )
            outputs.append(step_output.squeeze(0))

            state_segment.store_window(
                local_layer_id,
                torch.as_tensor(
                    [state_slot], dtype=torch.long, device=hidden_states.device
                ),
                torch.as_tensor(
                    [position], dtype=torch.long, device=hidden_states.device
                ),
                cache.window[:, position % self.window_size],
            )
            if self.compress_ratio and (position + 1) % self.compress_ratio == 0:
                compressed_position = compressed_count - 1
                segment.store_dsv4_compressed(
                    local_layer_id,
                    seq.page_table,
                    [compressed_position],
                    cache.compressed[:, compressed_position],
                )
                if self.indexer is not None:
                    segment.store_dsv4_compressed(
                        local_layer_id,
                        seq.page_table,
                        [compressed_position],
                        cache.index_compressed[:, compressed_position],
                        indexer=True,
                    )
        if not outputs:
            return hidden_states.new_empty((0, self.projections.hidden_size))
        return torch.cat(outputs, dim=0)

__all__ = ["DeepseekV4AttentionReference"]
