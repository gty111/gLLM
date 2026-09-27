"""Regression probe for qlen>1 GDN gate projection views.

Qwen's fused projections produce strided Q/K/V and gate views.  The recurrent
kernel must consume those views directly without changing its result or state.
"""

import torch

from gllm.layers.ops.fla.fused_recurrent import fused_recurrent_gdn_spec


def main():
    torch.manual_seed(9)
    T, H, HV, K, V = 4, 2, 2, 16, 16
    mixed = torch.randn(T, H * K * 2 + HV * V, device="cuda", dtype=torch.bfloat16)
    q0, k0, v0 = mixed.split((H * K, H * K, HV * V), dim=-1)
    q = q0.reshape(1, T, H, K)
    k = k0.reshape(1, T, H, K)
    v = v0.reshape(1, T, HV, V)
    ba = torch.randn(T, 2 * HV, device="cuda", dtype=torch.bfloat16)
    b, a = ba.split(HV, dim=-1)
    assert not a.is_contiguous() and not b.is_contiguous()
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
    h0 = torch.randn(HV, V, K, device="cuda", dtype=torch.float32) * 0.05
    pool = torch.zeros(T + 1, HV, V, K, device="cuda", dtype=torch.float32)
    pool[1].copy_(h0)
    indices = torch.arange(1, T + 1, device="cuda", dtype=torch.int32)[None]
    accepted = torch.ones(1, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)

    out = fused_recurrent_gdn_spec(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        scale=K**-0.5,
        state_source=pool,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    state = h0.clone()
    ref_out, ref_states = [], []
    for t in range(T):
        qt, kt, vt = q[0, t].float(), k[0, t].float(), v[0, t].float()
        qt *= torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
        kt *= torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        decay = -torch.exp(A_log) * torch.nn.functional.softplus(a[t].float() + dt_bias)
        beta = torch.sigmoid(b[t].float())
        state *= torch.exp(decay)[:, None, None]
        delta = (vt - (state * kt[:, None, :]).sum(-1)) * beta[:, None]
        state += delta[:, :, None] * kt[:, None, :]
        ref_out.append((state * (qt * (K**-0.5))[:, None, :]).sum(-1))
        ref_states.append(state.clone())

    ref_out = torch.stack(ref_out).to(out.dtype)
    ref_states = torch.stack(ref_states)
    out_diff = (out[0] - ref_out).abs().float()
    state_diff = (pool[1:] - ref_states).abs()
    print(
        f"out_max={out_diff.max().item():.6g} "
        f"state_max={state_diff.max().item():.6g}"
    )
    assert torch.allclose(out[0], ref_out, atol=2e-3, rtol=2e-3)
    assert torch.allclose(pool[1:], ref_states, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    main()
