from types import SimpleNamespace

import pytest
import torch

from gllm.layers.attention.deepseek_v4.layer import DeepseekV4Attention
from gllm.runtime.cache_arena import CacheArena
from gllm.runtime.memory_manager import (
    DeepseekV4KVCacheConfig,
    DeepseekV4StateCacheConfig,
    DeepseekV4StateSegment,
    MemoryManager,
    Segment,
)
from gllm.runtime.sequence import GenerationSequence


FP8_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
}


def _config(ratio):
    return SimpleNamespace(
        hidden_size=512,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        rms_norm_eps=1e-6,
        quantization_config=FP8_CONFIG,
        window_size=8,
        compress_ratios=(ratio,),
        compress_rope_theta=40000.0,
        rope_theta=10000.0,
        original_seq_len=16,
        max_position_embeddings=32,
        rope_scaling={"factor": 4.0, "beta_fast": 32, "beta_slow": 1},
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
    )


def _load_fp8(linear):
    from deep_gemm.utils import per_block_cast_to_fp8

    weight = torch.randn_like(linear.weight, dtype=torch.bfloat16) * 0.02
    q, scale = per_block_cast_to_fp8(weight, use_ue8m0=True, gran_k=128)
    linear.weight.data.copy_(q)
    linear.weight_scale_inv.data.copy_(scale)


def _initialize_module(module):
    for linear in (
        module.projections.wq_a,
        module.projections.wq_b,
        module.projections.wkv,
        module.projections.wo_a,
        module.projections.wo_b,
    ):
        _load_fp8(linear)
    module.projections.q_norm.weight.data.fill_(1)
    module.projections.kv_norm.weight.data.fill_(1)
    module.attn_sink.data.normal_(std=0.1)
    if module.compressor is not None:
        module.compressor.wkv.weight.data.normal_(std=0.02)
        module.compressor.wgate.weight.data.normal_(std=0.02)
        module.compressor.ape.data.normal_(std=0.02)
    if module.indexer is not None:
        _load_fp8(module.indexer.wq_b)
        module.indexer.weights_proj.weight.data.normal_(std=0.02)
        module.indexer.compressor.wkv.weight.data.normal_(std=0.02)
        module.indexer.compressor.wgate.weight.data.normal_(std=0.02)
        module.indexer.compressor.ape.data.normal_(std=0.02)


@pytest.mark.parametrize("ratio", [0, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_full_prefill_attention_is_finite(ratio):
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(67 + ratio)
    module = DeepseekV4Attention(0, _config(ratio))
    _initialize_module(module)

    hidden = torch.randn(1, 8, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    output = module.forward_prefill(hidden)
    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("ratio", [0, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_online_decode_matches_full_prefill_last_token(ratio):
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(91 + ratio)
    module = DeepseekV4Attention(0, _config(ratio))
    _initialize_module(module)

    hidden = torch.randn(1, 10, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    _, cache = module.forward_prefill_with_cache(hidden[:, :3])
    for position in range(3, hidden.shape[1]):
        decoded = module.forward_decode(
            hidden[:, position : position + 1],
            position=position,
            cache=cache,
        )
        reference = module.forward_prefill(hidden[:, : position + 1])[:, -1:]
        torch.testing.assert_close(decoded, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_c128_decode_boundary_matches_prefill():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(219)
    config = _config(128)
    config.max_position_embeddings = 160
    module = DeepseekV4Attention(0, config)
    _initialize_module(module)

    hidden = torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    _, cache = module.forward_prefill_with_cache(hidden[:, :127])
    decoded = module.forward_decode(hidden[:, 127:], position=127, cache=cache)
    reference = module.forward_prefill(hidden)[:, -1:]
    torch.testing.assert_close(decoded, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_paged_chunked_prefill_matches_contiguous_reference():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    torch.manual_seed(401)
    config = _config(4)
    module = DeepseekV4Attention(0, config)
    _initialize_module(module)
    hidden = torch.randn(1, 10, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    expected = module.forward_prefill(hidden).squeeze(0)

    kv_cfg = DeepseekV4KVCacheConfig([4], head_dim=128, index_head_dim=128)
    state_cfg = DeepseekV4StateCacheConfig(
        [4], head_dim=128, index_head_dim=128
    )
    manager = MemoryManager(
        0.9,
        num_layers=1,
        dtype=torch.bfloat16,
        page_size=64,
        kv_head_num=1,
        kv_head_dim=128,
        vocab_size=128,
        use_mla=True,
        dsv4_kv_cache_config=kv_cfg,
        dsv4_state_cache_config=state_cfg,
    )
    kv_layout = manager._kv_cache_layout()
    kv_arena = CacheArena(
        torch.empty(2 * kv_layout.entry_bytes, dtype=torch.uint8, device="cuda"),
        physical_page_bytes=kv_layout.entry_bytes,
    )
    manager.segment = Segment(
        1,
        64,
        1,
        128,
        True,
        kv_arena.register_cache(kv_layout),
        dsv4_kv_cache_config=kv_cfg,
    )
    state_layout = manager._dsv4_state_cache_layout()
    state_arena = CacheArena(
        torch.empty(
            3 * state_layout.entry_bytes, dtype=torch.uint8, device="cuda"
        ),
        physical_page_bytes=state_layout.entry_bytes,
    )
    manager.dsv4_state_segment = DeepseekV4StateSegment(
        state_cfg,
        state_cache=state_arena.register_cache(state_layout),
    )

    seq = GenerationSequence(1, list(range(10)), [], output_len=1)
    seq.page_table = [manager.segment.allocate()]
    seq.recurrent_state_slot = manager.dsv4_state_segment.allocate_block()

    def run_chunk(start, end):
        seq.computed_token_num = start
        seq.to_compute_token_num = end - start
        slots = torch.arange(start, end, dtype=torch.long, device="cuda")
        fake_input = SimpleNamespace(
            memory_manager=manager,
            seqs=[seq],
            query_start_loc_cpu=torch.tensor([0, end - start]),
            get_slot_mapping=lambda: slots,
        )
        return module.forward_paged_reference(
            fake_input, hidden[0, start:end], local_layer_id=0
        )

    actual = torch.cat([run_chunk(0, 3), run_chunk(3, 10)], dim=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_batched_continuation_prefill_matches_full_reference(ratio):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SGLang FlashMLA sparse prefill test targets SM100")
    pytest.importorskip("sgl_kernel")
    pytest.importorskip("deep_gemm")

    torch.manual_seed(457 + ratio)
    config = _config(ratio)
    config.head_dim = 512
    config.num_attention_heads = 8
    config.index_n_heads = 8
    config.window_size = 128
    config.max_position_embeddings = 192
    module = DeepseekV4Attention(0, config)
    _initialize_module(module)

    context_lengths = [5, 8] if ratio != 128 else [127, 130]
    suffix_lengths = [3, 2] if ratio != 128 else [2, 3]
    full_hidden = [
        torch.randn(
            context + suffix,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
        for context, suffix in zip(
            context_lengths, suffix_lengths, strict=True
        )
    ]
    expected = torch.cat(
        [
            module.forward_prefill(row.unsqueeze(0))[0, context:]
            for row, context in zip(
                full_hidden, context_lengths, strict=True
            )
        ],
        dim=0,
    )

    kv_cfg = DeepseekV4KVCacheConfig(
        [ratio], head_dim=512, index_head_dim=128
    )
    state_cfg = DeepseekV4StateCacheConfig(
        [ratio if ratio else 4], head_dim=512, index_head_dim=128
    )
    manager = MemoryManager(
        0.9,
        num_layers=1,
        dtype=torch.bfloat16,
        page_size=64,
        kv_head_num=1,
        kv_head_dim=512,
        vocab_size=128,
        use_mla=True,
        dsv4_kv_cache_config=kv_cfg,
        dsv4_state_cache_config=state_cfg,
    )
    kv_layout = manager._kv_cache_layout()
    kv_arena = CacheArena(
        torch.empty(16 * kv_layout.entry_bytes, dtype=torch.uint8, device="cuda"),
        physical_page_bytes=kv_layout.entry_bytes,
    )
    manager.segment = Segment(
        1,
        64,
        1,
        512,
        True,
        kv_arena.register_cache(kv_layout),
        dsv4_kv_cache_config=kv_cfg,
    )
    state_layout = manager._dsv4_state_cache_layout()
    state_arena = CacheArena(
        torch.empty(4 * state_layout.entry_bytes, dtype=torch.uint8, device="cuda"),
        physical_page_bytes=state_layout.entry_bytes,
    )
    manager.dsv4_state_segment = DeepseekV4StateSegment(
        state_cfg,
        state_cache=state_arena.register_cache(state_layout),
    )

    seqs = []
    page_tables = []
    state_slots = []
    for seq_id, total in enumerate(
        [
            context + suffix
            for context, suffix in zip(
                context_lengths, suffix_lengths, strict=True
            )
        ]
    ):
        seq = GenerationSequence(seq_id, list(range(total)), [], output_len=1)
        pages = [
            manager.segment.allocate() for _ in range((total + 63) // 64)
        ]
        seq.page_table = pages
        seq.recurrent_state_slot = manager.dsv4_state_segment.allocate_block()
        seqs.append(seq)
        page_tables.append(pages)
        state_slots.append(seq.recurrent_state_slot)
    max_pages = max(map(len, page_tables))
    block_table = torch.tensor(
        [pages + [pages[0]] * (max_pages - len(pages)) for pages in page_tables],
        dtype=torch.int32,
        device="cuda",
    )
    state_slots_tensor = torch.tensor(
        state_slots, dtype=torch.int32, device="cuda"
    )

    def slots_for(starts, lengths):
        slots = []
        for pages, start, length in zip(
            page_tables, starts, lengths, strict=True
        ):
            for position in range(start, start + length):
                slots.append(
                    pages[position // 64] * 64 + position % 64
                )
        return torch.tensor(slots, dtype=torch.long, device="cuda")

    prefix_hidden = torch.cat(
        [row[:context] for row, context in zip(full_hidden, context_lengths, strict=True)]
    )
    prefix_starts = torch.tensor(
        [0, context_lengths[0], sum(context_lengths)],
        dtype=torch.int32,
        device="cuda",
    )
    prefix_slots = slots_for([0, 0], context_lengths)
    prefix_input = SimpleNamespace(
        memory_manager=manager,
        seqs=seqs,
        metadata=SimpleNamespace(
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=2,
            slot_mapping=prefix_slots,
            prefill=SimpleNamespace(
                query_start_loc=prefix_starts,
                block_table=block_table,
                context_lens=torch.zeros(2, dtype=torch.int32, device="cuda"),
            ),
        ),
        max_context_len=0,
        max_seq_len=max(context_lengths),
        prefill_max_query_len=max(context_lengths),
        get_recurrent_state_slot_per_seq=lambda: state_slots_tensor,
    )
    module.forward_paged(prefix_input, prefix_hidden, local_layer_id=0)

    for seq, context, suffix in zip(
        seqs, context_lengths, suffix_lengths, strict=True
    ):
        seq.computed_token_num = context
        seq.to_compute_token_num = suffix
    suffix_hidden = torch.cat(
        [row[context:] for row, context in zip(full_hidden, context_lengths, strict=True)]
    )
    suffix_starts = torch.tensor(
        [0, suffix_lengths[0], sum(suffix_lengths)],
        dtype=torch.int32,
        device="cuda",
    )
    suffix_slots = slots_for(context_lengths, suffix_lengths)
    continuation_input = SimpleNamespace(
        memory_manager=manager,
        seqs=seqs,
        metadata=SimpleNamespace(
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=2,
            slot_mapping=suffix_slots,
            prefill=SimpleNamespace(
                query_start_loc=suffix_starts,
                block_table=block_table,
                context_lens=torch.tensor(
                    context_lengths, dtype=torch.int32, device="cuda"
                ),
            ),
        ),
        max_context_len=max(context_lengths),
        max_seq_len=max(
            context + suffix
            for context, suffix in zip(
                context_lengths, suffix_lengths, strict=True
            )
        ),
        prefill_max_query_len=max(suffix_lengths),
        get_recurrent_state_slot_per_seq=lambda: state_slots_tensor,
    )
    actual = module.forward_paged(
        continuation_input, suffix_hidden, local_layer_id=0
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.012)

    # A scheduler tick can combine one-token decode with another request's
    # continuation prefill. Both halves must retain their batched kernels; the
    # presence of cached prefill context must not drag decode into the oracle.
    decode_position = context_lengths[0] + suffix_lengths[0]
    continuation_start = context_lengths[1] + suffix_lengths[1]
    decode_hidden = torch.randn(
        1, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    continued_hidden = torch.randn(
        2, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    expected_mixed = torch.cat(
        [
            module.forward_prefill(
                torch.cat([full_hidden[0], decode_hidden], dim=0).unsqueeze(0)
            )[:, -1],
            module.forward_prefill(
                torch.cat([full_hidden[1], continued_hidden], dim=0).unsqueeze(0)
            )[:, -2:].squeeze(0),
        ],
        dim=0,
    )
    seqs[0].computed_token_num = decode_position
    seqs[0].to_compute_token_num = 1
    seqs[1].computed_token_num = continuation_start
    seqs[1].to_compute_token_num = 2
    decode_slot = torch.tensor(
        [
            page_tables[0][decode_position // 64] * 64
            + decode_position % 64
        ],
        dtype=torch.long,
        device="cuda",
    )
    continued_slots = torch.tensor(
        [
            page_tables[1][position // 64] * 64 + position % 64
            for position in range(continuation_start, continuation_start + 2)
        ],
        dtype=torch.long,
        device="cuda",
    )
    mixed_slots = torch.cat([decode_slot, continued_slots])
    mixed_input = SimpleNamespace(
        memory_manager=manager,
        seqs=seqs,
        metadata=SimpleNamespace(
            num_decode_tokens=1,
            num_decodes=1,
            num_prefills=1,
            slot_mapping=mixed_slots,
            decode=SimpleNamespace(
                block_table=block_table[:1],
                seq_lens=torch.tensor(
                    [decode_position + 1], dtype=torch.int32, device="cuda"
                ),
            ),
            prefill=SimpleNamespace(
                query_start_loc=torch.tensor(
                    [0, 2], dtype=torch.int32, device="cuda"
                ),
                block_table=block_table[1:2],
                context_lens=torch.tensor(
                    [continuation_start], dtype=torch.int32, device="cuda"
                ),
            ),
        ),
        max_context_len=continuation_start,
        max_seq_len=max(decode_position + 1, continuation_start + 2),
        prefill_max_query_len=2,
        workspace=torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        get_position=lambda: torch.tensor(
            [decode_position], dtype=torch.long, device="cuda"
        ),
        get_recurrent_state_slot_per_seq=lambda: state_slots_tensor,
    )
    mixed = module.forward_paged(
        mixed_input,
        torch.cat([decode_hidden, continued_hidden], dim=0),
        local_layer_id=0,
    )
    torch.testing.assert_close(mixed, expected_mixed, rtol=0.02, atol=0.012)


@pytest.mark.parametrize("ratio", [0, 4])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_fused_paged_prefill_matches_per_request_reference(ratio):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SGLang FlashMLA sparse prefill test targets SM100")
    pytest.importorskip("sgl_kernel")
    pytest.importorskip("deep_gemm")

    torch.manual_seed(503 + ratio)
    config = _config(ratio)
    config.head_dim = 512
    config.num_attention_heads = 8
    config.index_n_heads = 8
    config.window_size = 128
    module = DeepseekV4Attention(0, config)
    _initialize_module(module)

    lengths = [6, 8]
    hidden = torch.randn(
        sum(lengths), 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    expected = torch.cat(
        [
            module.forward_prefill(hidden[: lengths[0]].unsqueeze(0)).squeeze(0),
            module.forward_prefill(hidden[lengths[0] :].unsqueeze(0)).squeeze(0),
        ],
        dim=0,
    )

    kv_cfg = DeepseekV4KVCacheConfig(
        [ratio], head_dim=512, index_head_dim=128
    )
    # A state arena is part of the online V4 contract even for an SWA-only
    # layer.  Use one C4 layout row when this test's layer itself has no state.
    state_cfg = DeepseekV4StateCacheConfig(
        [ratio if ratio else 4], head_dim=512, index_head_dim=128
    )
    manager = MemoryManager(
        0.9,
        num_layers=1,
        dtype=torch.bfloat16,
        page_size=64,
        kv_head_num=1,
        kv_head_dim=512,
        vocab_size=128,
        use_mla=True,
        dsv4_kv_cache_config=kv_cfg,
        dsv4_state_cache_config=state_cfg,
    )
    kv_layout = manager._kv_cache_layout()
    kv_arena = CacheArena(
        torch.empty(4 * kv_layout.entry_bytes, dtype=torch.uint8, device="cuda"),
        physical_page_bytes=kv_layout.entry_bytes,
    )
    manager.segment = Segment(
        1,
        64,
        1,
        512,
        True,
        kv_arena.register_cache(kv_layout),
        dsv4_kv_cache_config=kv_cfg,
    )
    state_layout = manager._dsv4_state_cache_layout()
    state_arena = CacheArena(
        torch.empty(
            4 * state_layout.entry_bytes, dtype=torch.uint8, device="cuda"
        ),
        physical_page_bytes=state_layout.entry_bytes,
    )
    manager.dsv4_state_segment = DeepseekV4StateSegment(
        state_cfg,
        state_cache=state_arena.register_cache(state_layout),
    )

    seqs = []
    slots = []
    pages = []
    for seq_id, length in enumerate(lengths):
        seq = GenerationSequence(seq_id, list(range(length)), [], output_len=1)
        page = manager.segment.allocate()
        pages.append(page)
        seq.page_table = [page]
        seq.recurrent_state_slot = manager.dsv4_state_segment.allocate_block()
        seq.to_compute_token_num = length
        seqs.append(seq)
        slots.append(
            page * 64
            + torch.arange(length, dtype=torch.long, device="cuda")
        )

    prefill_slots = torch.cat(slots)
    prefill_query_start = torch.tensor(
        [0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda"
    )
    prefill_block_table = torch.tensor(
        [[page] for page in pages], dtype=torch.int32, device="cuda"
    )
    fake_input = SimpleNamespace(
        memory_manager=manager,
        seqs=seqs,
        query_start_loc_cpu=torch.tensor([0, lengths[0], sum(lengths)]),
        metadata=SimpleNamespace(
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=2,
            slot_mapping=prefill_slots,
            prefill=SimpleNamespace(
                query_start_loc=prefill_query_start,
                block_table=prefill_block_table,
            ),
        ),
        max_context_len=0,
        max_seq_len=max(lengths),
        prefill_max_query_len=max(lengths),
        get_slot_mapping=lambda: prefill_slots,
        get_recurrent_state_slot_per_seq=lambda: torch.tensor(
            [seq.recurrent_state_slot for seq in seqs],
            dtype=torch.int32,
            device="cuda",
        ),
    )
    actual = module.forward_paged(fake_input, hidden, local_layer_id=0)
    # FlashMLA accumulates in a different tiled order from the explicit FP32
    # oracle; the observed worst case after the native FP8 output projection is
    # about 1e-2, while all cache/state tensors remain bit-identical.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.012)

    # The fused prefill must leave the exact paged KV and recurrent compressor
    # state expected by the existing one-token online update.
    batch_next = torch.randn(
        2, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    for seq, length in zip(seqs, lengths, strict=True):
        seq.computed_token_num = length
        seq.to_compute_token_num = 1
    decode_positions = torch.tensor(lengths, dtype=torch.long, device="cuda")
    decode_seq_lens = (decode_positions + 1).to(torch.int32)
    decode_block_table = torch.tensor(
        [[page] for page in pages], dtype=torch.int32, device="cuda"
    )
    decode_slots = torch.tensor(
        [page * 64 + length for page, length in zip(pages, lengths, strict=True)],
        dtype=torch.long,
        device="cuda",
    )
    decode_state_slots = torch.tensor(
        [seq.recurrent_state_slot for seq in seqs],
        dtype=torch.int32,
        device="cuda",
    )
    batch_decode_input = SimpleNamespace(
        memory_manager=manager,
        seqs=seqs,
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
            metadata=SimpleNamespace(
                num_decode_tokens=2,
                num_decodes=2,
                num_prefills=0,
                slot_mapping=decode_slots,
            decode=SimpleNamespace(
                block_table=decode_block_table,
                seq_lens=decode_seq_lens,
            ),
        ),
        max_seq_len=max(lengths) + 1,
        workspace=torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        ),
        get_position=lambda: decode_positions,
        get_slot_mapping=lambda: decode_slots,
        get_recurrent_state_slot_per_seq=lambda: decode_state_slots,
    )

    # Full decode graphs are captured with short dummy sequences and replayed
    # at real positions. Snapshot the request-owned cache/state so eager and
    # capture/replay consume exactly the same prefix. This specifically guards
    # the C4 candidate-width invariant: capture must not bake the dummy
    # sequence's one-row compressed horizon into every later replay.
    cache_tensors = []
    if manager.segment.dsv4_compressed_cache[0] is not None:
        cache_tensors.append(manager.segment.dsv4_compressed_cache[0])
    if manager.segment.dsv4_index_cache[0] is not None:
        cache_tensors.append(manager.segment.dsv4_index_cache[0])
    cache_tensors.extend(
        [
            manager.dsv4_state_segment.kv,
            manager.dsv4_state_segment.score,
            # The sliding window is request-owned too, so it has to be part of
            # the snapshot for eager and replay to see the same prefix.
            manager.dsv4_state_segment.window,
        ]
    )
    prefix_snapshot = [tensor.clone() for tensor in cache_tensors]

    batch_decoded = module.forward_paged(
        batch_decode_input, batch_next, local_layer_id=0
    )
    for tensor, snapshot in zip(cache_tensors, prefix_snapshot, strict=True):
        tensor.copy_(snapshot)

    graph_output = torch.empty_like(batch_decoded)
    real_positions = decode_positions.clone()
    real_seq_lens = decode_seq_lens.clone()
    real_max_seq_len = batch_decode_input.max_seq_len
    decode_positions.zero_()
    decode_seq_lens.fill_(1)
    batch_decode_input.max_seq_len = 2
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph):
            graph_output.copy_(
                module.forward_paged(
                    batch_decode_input,
                    batch_next,
                    local_layer_id=0,
                )
            )
    torch.cuda.current_stream().wait_stream(capture_stream)
    decode_positions.copy_(real_positions)
    decode_seq_lens.copy_(real_seq_lens)
    batch_decode_input.max_seq_len = real_max_seq_len
    for tensor, snapshot in zip(cache_tensors, prefix_snapshot, strict=True):
        tensor.copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, batch_decoded, rtol=0, atol=0)

    expected_batch = torch.cat(
        [
            module.forward_prefill(torch.cat([prefix, batch_next[row : row + 1]], dim=0).unsqueeze(0))[:, -1]
            for row, prefix in enumerate(
                (hidden[: lengths[0]], hidden[lengths[0] :])
            )
        ],
        dim=0,
    )
    torch.testing.assert_close(
        batch_decoded, expected_batch, rtol=0.02, atol=0.012
    )

    next_hidden = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    decode_seq = seqs[1]
    decode_seq.computed_token_num = lengths[1] + 1
    decode_seq.to_compute_token_num = 1

    # Exercise the token-wise oracle on the scheduler's mixed batch shape.
    # ``forward_paged`` no longer falls back here -- it requires forward
    # metadata and raises without it -- so call the oracle directly, which is
    # what this section was always actually testing.
    new_length = 5
    new_hidden = torch.randn(
        new_length, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    new_seq = GenerationSequence(2, list(range(new_length)), [], output_len=1)
    new_page = manager.segment.allocate()
    new_seq.page_table = [new_page]
    new_seq.recurrent_state_slot = manager.dsv4_state_segment.allocate_block()
    new_seq.to_compute_token_num = new_length
    mixed_input = SimpleNamespace(
        memory_manager=manager,
        seqs=[decode_seq, new_seq],
        query_start_loc_cpu=torch.tensor([0, 1, 1 + new_length]),
        get_slot_mapping=lambda: torch.tensor(
            [pages[1] * 64 + lengths[1] + 1]
            + [new_page * 64 + offset for offset in range(new_length)],
            dtype=torch.long,
            device="cuda",
        ),
    )
    mixed = module.forward_paged_reference(
        mixed_input,
        torch.cat([next_hidden, new_hidden], dim=0),
        local_layer_id=0,
    )
    expected_mixed = torch.cat(
        [
            module.forward_prefill(
                torch.cat(
                    [
                        hidden[lengths[0] :],
                        batch_next[1:2],
                        next_hidden,
                    ],
                    dim=0,
                ).unsqueeze(0)
            )[:, -1],
            module.forward_prefill(new_hidden.unsqueeze(0)).squeeze(0),
        ],
        dim=0,
    )
    torch.testing.assert_close(
        mixed[:1], expected_mixed[:1], rtol=0.02, atol=0.012
    )
    torch.testing.assert_close(
        mixed[1:], expected_mixed[1:], rtol=0.02, atol=0.012
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_index_cache_is_invalidated_per_forward():
    """The per-forward memo must never serve one batch's indices to another.

    The decode path caches index tensors that depend on the batch's positions.
    They are keyed on the metadata object, which is rebuilt every forward; a
    stale hit would be a silent wrong answer rather than a crash.
    """
    from gllm.layers.attention.deepseek_v4.layer import DeepseekV4Attention

    first = SimpleNamespace(a=1)
    holder = SimpleNamespace(metadata=first)

    cache = DeepseekV4Attention._decode_cache(holder)
    cache["probe"] = "first-batch"
    assert DeepseekV4Attention._decode_cache(holder) is cache, "same forward reuses"

    # A new forward installs a new metadata object; the memo must reset.
    holder.metadata = SimpleNamespace(a=2)
    fresh = DeepseekV4Attention._decode_cache(holder)
    assert fresh is not cache
    assert "probe" not in fresh

    # Identity, not equality: a distinct object that compares equal is still a
    # new forward.
    holder.metadata = SimpleNamespace(a=2)
    assert DeepseekV4Attention._decode_cache(holder) is not fresh


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "pages,rows,width,batch", [(8, 4, 576, 1), (16, 8, 576, 5), (4, 2, 64, 3)]
)
def test_scatter_rows_where_matches_read_modify_write(pages, rows, width, batch):
    """The masked commit must be identical to the gather/where/scatter it replaces."""
    from gllm.layers.ops.deepseek_v4.scatter import scatter_rows_where

    torch.manual_seed(pages + batch)
    fused = torch.randn(pages, rows, width, device="cuda", dtype=torch.bfloat16)
    reference = fused.clone()
    page_ids = torch.randint(0, pages, (batch,), device="cuda", dtype=torch.int64)
    row_ids = torch.randint(0, rows, (batch,), device="cuda", dtype=torch.int64)
    src = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(batch, device="cuda") > 0.5

    scatter_rows_where(fused, page_ids, row_ids, src, mask)
    old = reference[page_ids, row_ids]
    reference[page_ids, row_ids] = torch.where(mask.unsqueeze(1), src, old)

    assert torch.equal(fused, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value", [False, True])
def test_scatter_rows_where_uniform_mask(value):
    """An all-false mask must leave the cache untouched; all-true writes every row."""
    from gllm.layers.ops.deepseek_v4.scatter import scatter_rows_where

    cache = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()
    page_ids = torch.arange(4, device="cuda", dtype=torch.int64)
    row_ids = torch.zeros(4, device="cuda", dtype=torch.int64)
    src = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((4,), value, device="cuda", dtype=torch.bool)

    scatter_rows_where(cache, page_ids, row_ids, src, mask)
    if value:
        assert torch.equal(cache[page_ids, row_ids], src)
    else:
        assert torch.equal(cache, before)


def _rope_reference(x, frequencies, *, inverse=False):
    """Plain-PyTorch RoPE oracle, kept here so it cannot drift with the kernel."""
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if complex_x.ndim == 3:
        if frequencies.ndim == 2:
            frequencies = frequencies.view(1, complex_x.size(1), complex_x.size(-1))
    else:
        if frequencies.ndim == 2:
            frequencies = frequencies.view(
                1, complex_x.size(1), 1, complex_x.size(-1)
            )
        elif frequencies.ndim == 3:
            frequencies = frequencies.unsqueeze(-2)
    if inverse:
        frequencies = frequencies.conj()
    return torch.view_as_real(complex_x * frequencies).flatten(-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,freq_shape",
    [
        ((4, 1, 16, 128), (4, 1, 32)),   # decode q / indexer: per-row frequencies
        ((2, 7, 8, 128), (7, 32)),       # prefill: one row per position
        ((3, 1, 576), (3, 1, 32)),       # kv latent, 3D
        ((1, 1, 4, 128), (1, 1, 32)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_fused_rope_is_bit_exact(shape, freq_shape, inverse):
    """RoPE feeds attention scores directly; a drifted rotation is silent."""
    from gllm.layers.attention.deepseek_v4.ops import apply_rope_inplace

    torch.manual_seed(sum(shape))
    rope_dim = 64
    full = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    got, want = full.clone(), full.clone()
    frequencies = torch.polar(
        torch.ones(*freq_shape, device="cuda"),
        torch.randn(*freq_shape, device="cuda"),
    )

    # Every call site rotates a trailing slice of a wider tensor, so the rows
    # the kernel touches are strided.
    apply_rope_inplace(got[..., -rope_dim:], frequencies, inverse=inverse)
    want[..., -rope_dim:] = _rope_reference(
        want[..., -rope_dim:], frequencies, inverse=inverse
    )
    assert torch.equal(got, want)
