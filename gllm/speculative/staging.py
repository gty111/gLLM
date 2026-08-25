"""Persistent host/device staging used by Qwen3.5 MTP hot paths."""

from typing import Optional

import torch


def _pinned_i64(capacity: int) -> torch.Tensor:
    return torch.empty(capacity, dtype=torch.int64, device="cpu", pin_memory=True)


class MtpStagingBuffers:
    """Own the small persistent buffers shared by draft, verify, and KV sync.

    Keeping these tensors together makes their lifetime and capacity contract
    explicit. It also keeps allocation policy out of ``ModelRunner.__init__``;
    hot paths still reuse the exact same pinned/device storage as before.
    """

    def __init__(
        self,
        capacity: int,
        max_tokens: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.capacity = capacity

        self.mixed_last_rows = torch.empty(capacity, dtype=torch.int64, device=device)
        self.mixed_new_idx_host = _pinned_i64(capacity)
        self.mixed_new_ctx_host = _pinned_i64(capacity)
        self.mixed_new_idx_host_np = self.mixed_new_idx_host.numpy()
        self.mixed_new_ctx_host_np = self.mixed_new_ctx_host.numpy()
        self.mixed_new_idx = torch.empty(capacity, dtype=torch.int64, device=device)
        self.mixed_new_ctx = torch.empty_like(self.mixed_new_idx)

        self.seed_hidden = torch.empty(
            (capacity, hidden_size), dtype=hidden_dtype, device=device
        )
        self.row_idx = torch.arange(capacity, dtype=torch.int64, device=device)
        self.x1_host = _pinned_i64(capacity)
        self.x1_host_np = self.x1_host.numpy()
        self.x1_gpu = torch.empty(capacity, dtype=torch.int64, device=device)

        self.refresh_tokens = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.kv_idx_host = _pinned_i64(capacity)
        self.kv_val_host = _pinned_i64(capacity)
        self.kv_gidx_host = _pinned_i64(capacity)
        self.kv_gsrc_host = _pinned_i64(capacity)
        self.kv_idx_host_np = self.kv_idx_host.numpy()
        self.kv_val_host_np = self.kv_val_host.numpy()
        self.kv_gidx_host_np = self.kv_gidx_host.numpy()
        self.kv_gsrc_host_np = self.kv_gsrc_host.numpy()
        self.kv_idx_gpu = torch.empty(capacity, dtype=torch.int64, device=device)
        self.kv_val_gpu = torch.empty_like(self.kv_idx_gpu)
        self.kv_gidx_gpu = torch.empty_like(self.kv_idx_gpu)
        self.kv_gsrc_gpu = torch.empty_like(self.kv_idx_gpu)

        self.bootstrap_rows_host = _pinned_i64(capacity)
        self.bootstrap_rows_host_np = self.bootstrap_rows_host.numpy()
        self.bootstrap_rows_gpu = torch.empty(
            capacity, dtype=torch.int64, device=device
        )
        self.install_ctx_host = _pinned_i64(capacity)
        self.install_ctx_host_np = self.install_ctx_host.numpy()

    def patch_shifted_tokens(
        self,
        shifted: torch.Tensor,
        host_pairs,
        gpu_src: Optional[torch.Tensor],
        gpu_pairs,
    ) -> None:
        """Patch row-boundary tokens without a pageable host transfer."""
        if host_pairs:
            n = len(host_pairs)
            self.kv_idx_host_np[:n] = [pair[0] for pair in host_pairs]
            self.kv_val_host_np[:n] = [pair[1] for pair in host_pairs]
            idx = self.kv_idx_gpu[:n]
            val = self.kv_val_gpu[:n]
            idx.copy_(self.kv_idx_host[:n], non_blocking=True)
            val.copy_(self.kv_val_host[:n], non_blocking=True)
            shifted.index_copy_(0, idx, val)

        if gpu_pairs and gpu_src is not None:
            n = len(gpu_pairs)
            self.kv_gidx_host_np[:n] = [pair[0] for pair in gpu_pairs]
            self.kv_gsrc_host_np[:n] = [pair[1] for pair in gpu_pairs]
            gidx = self.kv_gidx_gpu[:n]
            gsrc = self.kv_gsrc_gpu[:n]
            gidx.copy_(self.kv_gidx_host[:n], non_blocking=True)
            gsrc.copy_(self.kv_gsrc_host[:n], non_blocking=True)
            shifted.index_copy_(
                0, gidx, gpu_src.to(torch.int64).index_select(0, gsrc)
            )
