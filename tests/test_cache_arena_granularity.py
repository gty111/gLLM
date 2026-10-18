"""Physical arena grain selection must preserve logical cache geometry."""
from itertools import permutations

import pytest
import torch

from gllm.runtime.cache_arena import (
    CacheArena,
    CacheLayout,
    CacheTensorLayout,
    select_physical_page_bytes,
)
from gllm.runtime.memory_manager import MemoryManager, SSMCacheConfig


def byte_layout(name, size):
    return CacheLayout(name, (CacheTensorLayout("data", torch.uint8, (size,)),))


def qwen_manager(cls=MemoryManager):
    return cls(
        1.0, num_layers=16, dtype=torch.bfloat16, page_size=16,
        kv_head_num=4, kv_head_dim=256, vocab_size=32,
        ssm_cache_config=SSMCacheConfig(
            num_layers=48, conv_dim=10240, conv_kernel=4,
            num_v_heads=48, head_v_dim=128, head_k_dim=128,
            dtype=torch.float32, conv_state_dtype=torch.bfloat16,
        ),
    )


def test_qwen_layout_plan_includes_working_states_and_snapshots():
    manager = qwen_manager()
    layouts = manager._cache_layouts()
    assert tuple(layouts) == ("kv_cache", "ssm_state", "ssm_snapshot")
    assert layouts["kv_cache"].entry_bytes == 1048576
    assert layouts["ssm_state"].entry_bytes == 153944064
    assert select_physical_page_bytes(layouts.values()) == 65536


def test_qwen_target_draft_and_state_gcd_is_order_independent():
    layouts = list(qwen_manager()._cache_layouts().values())
    layouts.append(CacheLayout("draft", (
        CacheTensorLayout("key", torch.bfloat16, (5, 16, 8, 128)),
        CacheTensorLayout("value", torch.bfloat16, (5, 16, 8, 128)),
    )))
    for ordered in permutations(layouts):
        grain = select_physical_page_bytes(iter(ordered))
        assert grain == 65536
    assert [layout.entry_bytes // grain for layout in layouts] == [16, 2349, 2349, 5]


def test_single_group_and_non_power_of_two_gcd():
    assert select_physical_page_bytes([byte_layout("kv", 1048576)]) == 1048576
    layouts = [byte_layout("a", 3072), byte_layout("b", 4608)]
    assert select_physical_page_bytes(layouts) == 1536


@pytest.mark.parametrize("sizes", [(1536, 2560), (16, 48), (1024, 2048)])
def test_floor_padding_preserves_tensor_views_and_exact_slot_capacity(sizes):
    layouts = [byte_layout("a", sizes[0]), byte_layout("b", sizes[1])]
    grain = select_physical_page_bytes(layouts)
    assert grain == 1024
    arena = CacheArena(torch.zeros(12 * grain, dtype=torch.uint8), grain)
    for layout in layouts:
        cache = arena.register_cache(layout)
        pages = (layout.entry_bytes + grain - 1) // grain
        assert cache.cache_type.pages_per_slot == pages
        assert cache.slot_bytes == pages * grain
        assert cache.entry_bytes == layout.entry_bytes
        assert cache.num_slots == 12 // pages
        view = cache.tensor("data")
        assert view.shape == (12 // pages, layout.entry_bytes)
        assert view.stride(0) == pages * grain
        view[-1].fill_(37)
        assert torch.all(view[-1] == 37)
    a = arena.allocator.allocate("a")[0]
    arena.cache("a").tensor("data")[a].fill_(19)
    b = arena.allocator.allocate("b")[0]
    arena.cache("b").tensor("data")[b].fill_(23)
    assert torch.all(arena.cache("a").tensor("data")[a] == 19)
    arena.allocator.free("b", [b])
    arena.allocator.free("a", [a])
    assert arena.allocator.num_used_physical_pages == 0


def test_floor_keeps_mixed_scalar_banks_aligned():
    layout = CacheLayout("mixed", (
        CacheTensorLayout("tiny", torch.uint8, (1,)),
        CacheTensorLayout("wide", torch.float64, (3,)),
    ))
    grain = select_physical_page_bytes([layout], min_page_bytes=1025)
    assert grain == 1040
    arena = CacheArena(torch.zeros(4 * grain, dtype=torch.uint8), grain)
    cache = arena.register_cache(layout)
    view = cache.tensor("wide")
    assert view.stride(0) * view.element_size() == grain
    assert view.data_ptr() % 8 == 0
    view[3].fill_(42)
    assert torch.all(view[3] == 42)


@pytest.mark.parametrize("layouts,minimum", [([], 1024), ([byte_layout("a", 16)], 0)])
def test_invalid_grain_selection_is_rejected(layouts, minimum):
    with pytest.raises(ValueError):
        select_physical_page_bytes(layouts, min_page_bytes=minimum)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("distributed", [False, True])
def test_manager_init_uses_layout_grain_and_gathers_logical_kv_slots(monkeypatch, distributed):
    # Small real SSM layouts have a GCD below 1 KiB and require slot padding.
    manager = MemoryManager(
        1.0, 2, torch.bfloat16, 16, 1, 16, 32,
        ssm_cache_config=SSMCacheConfig(
            num_layers=1, conv_dim=16, conv_kernel=4,
            num_v_heads=1, head_v_dim=8, head_k_dim=8,
            dtype=torch.float32, conv_state_dtype=torch.bfloat16,
        ),
    )
    layouts = manager._cache_layouts()
    grain = select_physical_page_bytes(layouts.values())
    assert grain == 1024
    kv_stride = layouts["kv_cache"].entry_bytes
    assert kv_stride == 2048
    budget = 64 * kv_stride + grain - 1
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (budget, budget))
    import gllm.runtime.memory_manager as module
    monkeypatch.setattr(module.dist, "is_initialized", lambda: distributed)
    if distributed:
        monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)

        def gather(result, local):
            assert local == 64  # Not 128 physical pages.
            result[:] = [64, 48]

        monkeypatch.setattr(module.dist, "all_gather_object", gather)
    manager.init(reserve_dummy_page=True)
    expected = 48 if distributed else 64
    arena = manager.cache_arena
    assert arena.physical_page_bytes == grain
    assert arena.backing.numel() == expected * kv_stride
    assert manager.num_pages == expected
    assert manager.segment.k_cache[0].stride(0) * 2 == kv_stride
    assert manager.segment.k_cache[0].shape[1:] == (16, 1, 16)
    assert arena.allocator.cache_type_names == tuple(layouts)
    state = arena.allocator.cache_type("ssm_state")
    assert state.pages_per_slot == 1
    assert manager.dummy_page is not None
    assert manager.validate_model_max_length(16) >= 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen_init_registers_predeclared_draft_with_exact_strides(monkeypatch):
    # The common layout plan is also the extension point for draft cache groups.
    class WithDraftLayout(MemoryManager):
        def _cache_layouts(self):
            layouts = super()._cache_layouts()
            layouts["draft"] = CacheLayout("draft", (
                CacheTensorLayout("key", torch.bfloat16, (5, 16, 8, 128)),
                CacheTensorLayout("value", torch.bfloat16, (5, 16, 8, 128)),
            ))
            return layouts

    manager = qwen_manager(WithDraftLayout)
    budget = 4 * manager._ssm_cache_layout("ssm_state").entry_bytes
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (budget, budget))
    manager.init(reserve_dummy_page=True)
    arena = manager.cache_arena
    assert arena.physical_page_bytes == 65536
    target, draft = arena.cache("kv_cache"), arena.cache("draft")
    assert target.cache_type.pages_per_slot == 16
    assert draft.cache_type.pages_per_slot == 5
    assert target.tensor("key").stride(0) * 2 == 1048576
    assert draft.tensor("key").stride(0) * 2 == 327680
    assert target.num_slots == arena.backing.numel() // 1048576
    assert draft.num_slots == arena.backing.numel() // 327680
    t = arena.allocator.allocate("kv_cache")[0]
    d = arena.allocator.allocate("draft")[0]
    target.tensor("key")[t].fill_(11)
    target.tensor("value")[t].fill_(13)
    draft.tensor("key")[d].fill_(17)
    draft.tensor("value")[d].fill_(19)
    assert torch.all(target.tensor("key")[t] == 11)
    assert torch.all(target.tensor("value")[t] == 13)
    assert torch.all(draft.tensor("key")[d] == 17)
    assert torch.all(draft.tensor("value")[d] == 19)
    assert arena.allocator.num_used_physical_pages == 2349 + 16 + 16 + 5
    arena.allocator.free("draft", [d])
    arena.allocator.free("kv_cache", [t])


def test_manager_rejects_budget_below_one_padded_kv_slot(monkeypatch):
    manager = MemoryManager(1.0, 1, torch.bfloat16, 4, 1, 8, 32)
    assert manager._kv_cache_layout().entry_bytes == 128
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (512, 512))
    import gllm.runtime.memory_manager as module
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)

    def unexpected_context():
        pytest.fail("an undersized budget must fail before allocating GPU storage")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_context)
    with pytest.raises(RuntimeError, match="not enough GPU memory"):
        manager.init()
