"""Paged attention backends for ordinary MHA/GQA layers.

Each backend owns both its runtime metadata preparation and its kernel call.
This keeps backend-specific construction and graph-captured metadata updates
out of ``InputData`` and the scheduler; ``InputData`` only carries the opaque
metadata object produced for its current forward.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from logger import logger

if TYPE_CHECKING:
    from gllm.input_data import InputData


@dataclass
class PagedAttentionMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    batch_size: int


@dataclass
class FA3PagedAttentionMetadata(PagedAttentionMetadata):
    """Metadata consumed by sgl-kernel FA3."""


@dataclass
class FlashInferPagedAttentionMetadata(PagedAttentionMetadata):
    """Metadata consumed by FlashInfer's TRT-LLM generation kernel."""

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor
    fast_path_rows: int
    fast_path_tokens: int
    fast_q_len_per_req: int
    context_max_query_len: int


class AttentionBackend(ABC):
    """Interface shared by standard paged-attention implementations."""

    name: str

    def __init__(self, model_max_length: int, max_running_seqs: int):
        self.model_max_length = model_max_length
        self.max_running_seqs = max_running_seqs

    @abstractmethod
    def prepare_metadata(self, input_data: "InputData") -> PagedAttentionMetadata:
        """Prepare metadata immediately before a model forward.

        GPU operations issued here are intentionally captured as part of CUDA
        graphs, so graph replay refreshes metadata from the current static input
        buffers without any scheduler/backend coupling.
        """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: PagedAttentionMetadata,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run paged causal attention with backend-prepared ``metadata``."""


class FA3AttentionBackend(AttentionBackend):
    """sgl-kernel FlashAttention-3 backend for its SM8x/SM9x builds."""

    name = "fa3"

    def __init__(self, model_max_length: int, max_running_seqs: int):
        super().__init__(model_max_length, max_running_seqs)
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        self._flash_attn_with_kvcache = flash_attn_with_kvcache

    def prepare_metadata(self, input_data: "InputData") -> FA3PagedAttentionMetadata:
        seq_lens = input_data.get_seq_lens()
        return FA3PagedAttentionMetadata(
            block_table=input_data.get_block_table(),
            seq_lens=seq_lens,
            query_start_loc=input_data.get_query_start_loc(),
            max_query_len=input_data.max_query_len,
            batch_size=seq_lens.shape[0],
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: FA3PagedAttentionMetadata,
        softmax_scale: float,
    ) -> torch.Tensor:
        return self._flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=metadata.seq_lens,
            page_table=metadata.block_table,
            cu_seqlens_q=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            softmax_scale=softmax_scale,
            causal=True,
            return_softmax_lse=False,
        )


class FlashInferAttentionBackend(AttentionBackend):
    """FlashInfer TRT-LLM generation backend for Blackwell/SM100.

    The cumulative KV lengths and workspace have stable addresses. Ragged
    context graphs capture the cumulative-length refresh, while uniform decode
    and MTP-verify graphs use the decode API and avoid that preparation work.
    """

    name = "flashinfer"
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(self, model_max_length: int, max_running_seqs: int):
        super().__init__(model_max_length, max_running_seqs)
        try:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache
            from flashinfer.prefill import trtllm_batch_context_with_kv_cache
        except Exception as exc:
            raise RuntimeError(
                "FlashInfer attention backend requested, but flashinfer could "
                f"not be imported: {exc}"
            ) from exc

        self._decode_attention = trtllm_batch_decode_with_kv_cache
        self._context_attention = trtllm_batch_context_with_kv_cache
        device = torch.device("cuda", torch.cuda.current_device())
        # FlashInfer requires this workspace to be zero-initialized on first use.
        self.workspace = torch.zeros(
            self._WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        self.cum_seq_lens_kv = torch.zeros(
            max_running_seqs + 1, dtype=torch.int32, device=device
        )
        self.cum_seq_lens_q = torch.zeros(
            max_running_seqs + 1, dtype=torch.int32, device=device
        )

    def prepare_metadata(
        self, input_data: "InputData"
    ) -> FlashInferPagedAttentionMetadata:
        seq_lens = input_data.get_seq_lens()
        batch_size = seq_lens.shape[0]
        if batch_size > self.max_running_seqs:
            raise RuntimeError(
                f"attention batch size {batch_size} exceeds metadata capacity "
                f"{self.max_running_seqs}"
            )

        # The scheduler keeps ordinary decode rows at the front. Mixed MTP
        # target forwards use the same layout, with a uniform verify prefix.
        # TRT-LLM Gen exposes separate decode and context kernels. Run the
        # uniform prefix with the decode kernel and send only the ragged
        # prefill suffix to the context kernel. Sending a mixed
        # decode+prefill batch through context corrupts row metadata on SM100.
        num_mtp_rows = int(getattr(input_data, "num_mtp_verify_rows", 0))
        fast_path_rows = num_mtp_rows or int(
            getattr(input_data, "num_decodes", 0)
        )
        if not 0 <= fast_path_rows <= batch_size:
            raise RuntimeError(
                f"invalid FlashInfer fast-path prefix {fast_path_rows}/{batch_size}"
            )

        query_start_loc_cpu = input_data.query_start_loc_cpu
        fast_path_tokens = int(query_start_loc_cpu[fast_path_rows])
        if fast_path_rows:
            fast_q_len_per_req = fast_path_tokens // fast_path_rows
            prefix_lens = (
                query_start_loc_cpu[1 : fast_path_rows + 1]
                - query_start_loc_cpu[:fast_path_rows]
            )
            if fast_q_len_per_req <= 0 or not bool(
                torch.all(prefix_lens == fast_q_len_per_req)
            ):
                raise RuntimeError(
                    "FlashInfer decode/verify prefix must have a uniform query "
                    f"length, got {prefix_lens.tolist()}"
                )
        else:
            fast_q_len_per_req = 1

        context_rows = batch_size - fast_path_rows
        cumulative_q = self.cum_seq_lens_q[: context_rows + 1]
        cumulative_kv = self.cum_seq_lens_kv[: context_rows + 1]
        if context_rows:
            query_start_loc = input_data.get_query_start_loc()
            torch.sub(
                query_start_loc[fast_path_rows:],
                query_start_loc[fast_path_rows],
                out=cumulative_q,
            )
            context_query_lens = (
                query_start_loc_cpu[fast_path_rows + 1 :]
                - query_start_loc_cpu[fast_path_rows:-1]
            )
            context_max_query_len = int(context_query_lens.max().item())

            # TRT-LLM Gen's paged-context API names this argument
            # ``cum_seq_lens_kv``, but for a paged KV cache it is the indptr
            # into the page table: each delta is the number of cache pages
            # owned by that request, not its number of KV tokens.  Passing a
            # token-length prefix sum makes request i index pages belonging to
            # other rows once a ragged/mixed batch contains more than one
            # request. Therefore the required indptr is
            # ``cumsum(ceil(seq_lens / page_size))``.
            page_size = input_data.page_size
            context_seq_lens = seq_lens[fast_path_rows:]
            num_pages = torch.div(
                context_seq_lens + page_size - 1,
                page_size,
                rounding_mode="floor",
            )
            cumulative_kv[0].zero_()
            torch.cumsum(num_pages, dim=0, out=cumulative_kv[1:])
        else:
            cumulative_q.zero_()
            cumulative_kv.zero_()
            context_max_query_len = 0
        return FlashInferPagedAttentionMetadata(
            block_table=input_data.get_block_table(),
            seq_lens=seq_lens,
            query_start_loc=input_data.get_query_start_loc(),
            max_query_len=input_data.max_query_len,
            batch_size=batch_size,
            cum_seq_lens_q=cumulative_q,
            cum_seq_lens_kv=cumulative_kv,
            fast_path_rows=fast_path_rows,
            fast_path_tokens=fast_path_tokens,
            fast_q_len_per_req=fast_q_len_per_req,
            context_max_query_len=context_max_query_len,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: FlashInferPagedAttentionMetadata,
        softmax_scale: float,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        if metadata.fast_path_rows:
            self._decode_attention(
                query=q[: metadata.fast_path_tokens],
                kv_cache=(k_cache, v_cache),
                workspace_buffer=self.workspace,
                block_tables=metadata.block_table[: metadata.fast_path_rows],
                seq_lens=metadata.seq_lens[: metadata.fast_path_rows],
                max_seq_len=self.model_max_length,
                bmm1_scale=softmax_scale,
                bmm2_scale=1.0,
                out=out[: metadata.fast_path_tokens],
                kv_layout="NHD",
                enable_pdl=True,
                backend="auto",
                q_len_per_req=metadata.fast_q_len_per_req,
                uses_shared_paged_kv_idx=True,
            )
        context_rows = metadata.batch_size - metadata.fast_path_rows
        if context_rows:
            token_start = metadata.fast_path_tokens
            row_start = metadata.fast_path_rows
            self._context_attention(
                q[token_start:],
                (k_cache, v_cache),
                self.workspace,
                metadata.block_table[row_start:],
                metadata.seq_lens[row_start:],
                metadata.context_max_query_len,
                self.model_max_length,
                softmax_scale,
                1.0,
                context_rows,
                metadata.cum_seq_lens_q,
                metadata.cum_seq_lens_kv,
                out=out[token_start:],
                kv_layout="NHD",
                enable_pdl=True,
                uses_shared_paged_kv_idx=True,
                causal=True,
            )
        return out


def create_attention_backend(
    requested: str, model_max_length: int, max_running_seqs: int
) -> AttentionBackend:
    """Resolve and construct the standard attention backend on this worker."""
    requested = (requested or "auto").lower()
    if requested not in ("auto", "fa3", "flashinfer"):
        raise ValueError(
            "attention_backend must be 'auto', 'fa3', or 'flashinfer', "
            f"got {requested!r}."
        )

    capability = torch.cuda.get_device_capability()
    if requested == "auto":
        if capability[0] == 10:
            resolved = "flashinfer"
        elif capability[0] in (8, 9):
            # sgl-kernel's FA3 extension is built for SM80/86/89/90a. This is
            # broader than the upstream FA3 project's Hopper-focused support.
            resolved = "fa3"
        else:
            raise RuntimeError(
                "No automatic standard attention backend for compute "
                f"capability SM{capability[0]}{capability[1]}; choose a "
                "supported backend explicitly."
            )
    else:
        resolved = requested

    if resolved == "fa3" and capability[0] not in (8, 9):
        raise RuntimeError(
            "sgl-kernel FA3 requires an SM8x or SM9x GPU, but this worker is "
            f"SM{capability[0]}{capability[1]}."
        )

    backend_cls = (
        FlashInferAttentionBackend if resolved == "flashinfer" else FA3AttentionBackend
    )
    backend = backend_cls(model_max_length, max_running_seqs)
    logger.info(
        "Standard attention backend: %s (requested %s, compute capability SM%d%d)",
        resolved,
        requested,
        capability[0],
        capability[1],
    )
    return backend


def find_attention_layers(model):
    """Return the standard attention helpers contained in ``model``.

    ``FlashAttention`` is deliberately a lightweight helper rather than an
    ``nn.Module``, so it is stored as a plain attribute of registered model
    modules. Inspect those direct attributes instead of relying on a model-wide
    ``use_mla`` flag.
    """
    from gllm.layers.attention import FlashAttention

    layers = []
    seen = set()
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, FlashAttention) and id(value) not in seen:
                seen.add(id(value))
                layers.append(value)
    return layers


def bind_attention_backend(attention_layers, backend: AttentionBackend) -> int:
    """Inject one shared backend into discovered standard attention layers."""
    for layer in attention_layers:
        layer.set_backend(backend)
    return len(attention_layers)
