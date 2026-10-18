"""Copy request-slot states to optional snapshot slots without host decisions."""

import math

import triton
import triton.language as tl


@triton.jit
def _copy_slots(
    State,
    Sources,
    Targets,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    target = tl.load(Targets + row).to(tl.int64)
    if target >= 0:
        source = tl.load(Sources + row).to(tl.int64)
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(State + source * STRIDE + columns, columns < WIDTH, 0)
        tl.store(State + target * STRIDE + columns, value, columns < WIDTH)


def copy_slot_snapshots(state, sources, targets):
    width = math.prod(state.shape[1:])
    _copy_slots[(sources.numel(), triton.cdiv(width, 256))](
        state, sources, targets, state.stride(0), width, 256
    )
