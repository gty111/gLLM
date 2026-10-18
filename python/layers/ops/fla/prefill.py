"""Indexed GDN prefill using a persistent recurrent state pool."""

import inspect
from functools import cache

import torch
import triton
import triton.language as tl

from gllm.layers.ops.fla.l2norm import l2norm_fwd


@triton.jit
def _prepare_prefill_kernel(
    Q,
    K,
    V,
    A,
    B,
    A_LOG,
    BIAS,
    Q_OUT,
    K_OUT,
    V_OUT,
    ALPHA,
    BETA,
    T,
    QS,
    KS,
    VS,
    AS,
    BS,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token = rows // HV
    head = rows % HV
    col = tl.arange(0, D)
    live = token < T
    q_live = live & (head < HK)
    q = tl.load(
        Q + token[:, None] * QS + head[:, None] * D + col[None, :],
        q_live[:, None],
        other=0,
    ).to(tl.float32)
    k = tl.load(
        K + token[:, None] * KS + head[:, None] * D + col[None, :],
        q_live[:, None],
        other=0,
    ).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q, axis=1) + 1e-6)[:, None]
    k = k / tl.sqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
    q_off = (token[:, None] * HK + head[:, None]) * D + col[None, :]
    tl.store(Q_OUT + q_off, q, q_live[:, None])
    tl.store(K_OUT + q_off, k, q_live[:, None])
    v = tl.load(
        V + token[:, None] * VS + head[:, None] * D + col[None, :],
        live[:, None],
        other=0,
    )
    tl.store(V_OUT + rows[:, None] * D + col[None, :], v, live[:, None])
    a = tl.load(A + token * AS + head, live, other=0).to(tl.float32)
    b = tl.load(B + token * BS + head, live, other=0).to(tl.float32)
    bias = tl.load(BIAS + head).to(tl.float32)
    a_log = tl.load(A_LOG + head).to(tl.float32)
    x = a + bias
    softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    log_alpha = -tl.exp(a_log) * softplus
    # PyTorch's exp uses expf. Preserve the separate FP32 log-alpha boundary.
    alpha = tl.extra.cuda.libdevice.exp(log_alpha)
    # The existing gate rounds sigmoid to the projection dtype before FP32 storage.
    beta = tl.sigmoid(b).to(B.dtype.element_ty).to(tl.float32)
    tl.store(ALPHA + rows, alpha, live)
    tl.store(BETA + rows, beta, live)


class PrefillInputWorkspace:
    """Retain scratch addresses for eager callbacks and captured graph replays."""

    def __init__(self):
        self._buffers = {}

    def prepare(self, q, k, v, a, b, a_log, bias):
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        tokens, hk, dim = q.shape
        hv = v.shape[1]
        assert dim == 128 and v.shape[2] == dim and hk <= hv
        assert all(x.stride(-1) == 1 for x in (q, k, v, a, b))
        assert q.stride(1) == k.stride(1) == v.stride(1) == dim
        capacity = triton.next_power_of_2(max(tokens, 1))
        key = (q.device, q.dtype, hk, hv, dim, capacity)
        if key not in self._buffers:
            # Keep every allocated capacity alive: an earlier graph may retain it.
            self._buffers[key] = (
                torch.empty((capacity, hk, dim), device=q.device, dtype=q.dtype),
                torch.empty((capacity, hk, dim), device=q.device, dtype=k.dtype),
                torch.empty((capacity, hv, dim), device=q.device, dtype=v.dtype),
                torch.empty((capacity, hv), device=q.device, dtype=torch.float32),
                torch.empty((capacity, hv), device=q.device, dtype=torch.float32),
            )
        outputs = tuple(x[:tokens] for x in self._buffers[key])
        _prepare_prefill_kernel[(triton.cdiv(tokens * hv, 16),)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            *outputs,
            tokens,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            a.stride(0),
            b.stride(0),
            hk,
            hv,
            dim,
            16,
            num_warps=8,
        )
        return outputs


def share_prefill_input_workspace(layers):
    """Sequential decoder layers consume each prepared input before reusing it."""
    workspace = PrefillInputWorkspace()
    for layer in layers:
        attention = getattr(layer, "linear_attn", None)
        if attention is not None:
            attention._prefill_input_workspace = workspace


@cache
def supports_flashinfer_prefill(key_dim, value_dim):
    if key_dim != 128 or value_dim != 128 or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        return False
    try:
        from flashinfer.gdn_prefill import (
            chunk_gated_delta_rule,
            chunk_gated_delta_rule_sm100,
        )
    except ImportError:
        return False
    return (
        chunk_gated_delta_rule_sm100 is not None
        and "state_indices" in inspect.signature(chunk_gated_delta_rule).parameters
    )


def flashinfer_prefill(
    q,
    k,
    v,
    g,
    beta,
    state,
    slots,
    boundaries,
    scale,
    *,
    out=None,
    gate_inputs=None,
    workspace=None,
):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    # FlashInfer's indexed kernel consumes normalized vectors and decay alpha.
    # The vendored FLA API consumes raw vectors and log(alpha).
    if gate_inputs is not None:
        a, b, a_log, bias = gate_inputs
        q, k, v, alpha, beta = workspace.prepare(q, k, v, a, b, a_log, bias)
    else:
        q = l2norm_fwd(q).squeeze(0).contiguous()
        k = l2norm_fwd(k).squeeze(0).contiguous()
        v = v.squeeze(0).contiguous()
        alpha = g.squeeze(0).exp()
        beta = beta.squeeze(0).contiguous()
    output, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=alpha,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_state=state,
        output_final_state=True,
        state_indices=slots,
        cu_seqlens=boundaries,
        output=out,
        use_cp=False,
    )
    return output
