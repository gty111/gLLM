"""One forward's metadata contract.

Input metadata can be materialized by the ordinary CPU builders, written
directly by the MTP GPU prep, or built on the CPU and patched from
GPU-resident MTP state.  Those are different *materialization mechanisms*,
but they must describe the same batch geometry to attention and recurrent
layers.  :class:`ForwardMetadataPlan` is that shared description.

The plan deliberately contains only small host-side facts.  Device tensors
remain owned by :class:`gllm.runtime.input_data.InputData`, while backend-specific
attention metadata is prepared from the plan immediately before an eager
forward (or while a CUDA graph is captured).
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    from gllm.runtime.input_data import InputData
    from gllm.layers.attention.qkv_backends import QKVAttentionBackend


class MetadataMaterialization(str, Enum):
    """How the current forward's persistent device buffers were populated."""

    CPU = "cpu"
    GPU_UNIFORM = "gpu_uniform"
    CPU_WITH_GPU_PATCH = "cpu_with_gpu_patch"
    DEFERRED_MTP = "deferred_mtp"


class ForwardMetadataMaterializer(Protocol):
    """Writes one plan's physical tensors into an ``InputData`` instance.

    Materializers are intentionally stateless from the plan's point of view:
    they may own persistent staging buffers (as ``MtpGpuPrep`` does), but they
    do not install or mutate the plan.  This keeps the dependency direction
    ``plan -> materializer -> InputData buffers``.
    """

    materialization: MetadataMaterialization

    def materialize_buffers(
        self,
        input_data: "InputData",
        plan: "ForwardMetadataPlan",
    ) -> None: ...


@dataclass
class ForwardMetadataPlan:
    """Canonical batch geometry shared by every metadata producer.

    ``query_lens`` includes CUDA-graph padding rows when present.  Decode or
    MTP-verify rows must form one uniform leading prefix; FlashInfer can then
    send that prefix to its decode kernel and the ragged suffix to its context
    kernel without re-deriving the split from whichever staging tensors happen
    to exist for the current materialization path.

    ``attention_metadata`` is intentionally transient.  It is populated by
    :meth:`prepare_attention` for eager execution/capture and is not copied
    when a prebuilt CPU plan is installed on the runtime ``InputData``.
    """

    materialization: MetadataMaterialization
    query_lens: Tuple[int, ...]
    num_decodes: int
    num_mtp_verify_rows: int = 0
    gpu_patch_rows: int = 0
    device_buffers_ready: bool = True
    attention_metadata: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.query_lens = tuple(int(n) for n in self.query_lens)
        rows = len(self.query_lens)
        if rows <= 0:
            raise ValueError("forward metadata plan requires at least one row")
        if any(n <= 0 for n in self.query_lens):
            raise ValueError(f"query lengths must be positive: {self.query_lens}")
        for name, value in (
            ("num_decodes", self.num_decodes),
            ("num_mtp_verify_rows", self.num_mtp_verify_rows),
            ("gpu_patch_rows", self.gpu_patch_rows),
        ):
            if not 0 <= value <= rows:
                raise ValueError(f"{name}={value} is outside batch size {rows}")
        if self.num_decodes and self.num_mtp_verify_rows:
            raise ValueError(
                "ordinary decode and MTP verify rows cannot share the fast prefix"
            )
        fast_rows = self.fast_path_rows
        if fast_rows:
            prefix = self.query_lens[:fast_rows]
            if any(n != prefix[0] for n in prefix[1:]):
                raise ValueError(
                    "attention fast-path rows must have one uniform query length: "
                    f"{prefix}"
                )
        if self.gpu_patch_rows:
            if self.materialization is not MetadataMaterialization.CPU_WITH_GPU_PATCH:
                raise ValueError(
                    "gpu_patch_rows requires CPU_WITH_GPU_PATCH materialization"
                )
            if self.gpu_patch_rows != self.num_mtp_verify_rows:
                raise ValueError(
                    "GPU-patched rows must match the MTP verify rows"
                )

    @classmethod
    def from_sequences(
        cls,
        seqs: Sequence[Any],
        *,
        num_decodes: Optional[int] = None,
        num_mtp_verify_rows: Optional[int] = None,
    ) -> "ForwardMetadataPlan":
        flags = [bool(getattr(seq, "_mtp_verify", False)) for seq in seqs]
        inferred_mtp_rows = 0
        while inferred_mtp_rows < len(flags) and flags[inferred_mtp_rows]:
            inferred_mtp_rows += 1
        if any(flags[inferred_mtp_rows:]):
            raise ValueError("MTP verify rows must form a contiguous batch prefix")
        if num_mtp_verify_rows is None:
            num_mtp_verify_rows = inferred_mtp_rows
        elif int(num_mtp_verify_rows) != inferred_mtp_rows:
            raise ValueError(
                "MTP row classification disagrees with sequence flags: "
                f"{num_mtp_verify_rows} != {inferred_mtp_rows}"
            )

        if num_decodes is None:
            num_decodes = len(seqs)
            for index, seq in enumerate(seqs):
                if (not seq.computed_prompt) or bool(
                    getattr(seq, "_mtp_verify", False)
                ):
                    num_decodes = index
                    break
        return cls(
            materialization=MetadataMaterialization.CPU,
            query_lens=tuple(int(seq.to_compute_token_num) for seq in seqs),
            num_decodes=int(num_decodes),
            num_mtp_verify_rows=int(num_mtp_verify_rows),
        )

    @classmethod
    def uniform_gpu(
        cls,
        *,
        num_rows: int,
        qlen: int,
        is_mtp_verify: bool,
    ) -> "ForwardMetadataPlan":
        return cls(
            materialization=MetadataMaterialization.GPU_UNIFORM,
            query_lens=(int(qlen),) * int(num_rows),
            num_decodes=0 if is_mtp_verify else int(num_rows),
            num_mtp_verify_rows=int(num_rows) if is_mtp_verify else 0,
        )

    @classmethod
    def deferred_mtp(cls, num_rows: int) -> "ForwardMetadataPlan":
        return cls(
            materialization=MetadataMaterialization.DEFERRED_MTP,
            query_lens=(1,) * int(num_rows),
            num_decodes=int(num_rows),
            device_buffers_ready=False,
        )

    @property
    def batch_size(self) -> int:
        return len(self.query_lens)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens)

    @property
    def fast_path_rows(self) -> int:
        return self.num_mtp_verify_rows or self.num_decodes

    @property
    def fast_path_tokens(self) -> int:
        return sum(self.query_lens[: self.fast_path_rows])

    @property
    def fast_q_len_per_req(self) -> int:
        return self.query_lens[0] if self.fast_path_rows else 1

    @property
    def context_max_query_len(self) -> int:
        suffix = self.query_lens[self.fast_path_rows :]
        return max(suffix, default=0)

    def clone_for_runtime(self) -> "ForwardMetadataPlan":
        """Copy a prebuilt plan without carrying backend runtime objects."""
        return replace(self, attention_metadata=None)

    def materialize(
        self,
        input_data: "InputData",
        materializer: ForwardMetadataMaterializer,
    ) -> "ForwardMetadataPlan":
        """Write physical buffers, then atomically install this plan.

        Attention preparation remains a separate phase because overlap
        scheduling writes inputs on ``forward_stream`` and consumes them later on
        ``forward_stream``.  The same plan spans both phases; callers invoke
        :meth:`prepare_attention` only after the stream dependency is in place.
        """
        if materializer.materialization is not self.materialization:
            raise ValueError(
                "metadata materializer/plan mismatch: "
                f"{materializer.materialization.value} != "
                f"{self.materialization.value}"
            )
        materializer.materialize_buffers(input_data, self)
        input_data.install_forward_metadata_plan(self)
        return self

    def with_gpu_patch(self, *, num_rows: int, qlen: int) -> "ForwardMetadataPlan":
        """Record a GPU correction of the leading MTP rows.

        The correction may change token ids, positions and sequence lengths,
        but not the already-planned ragged geometry.  Treat a mismatch as a
        producer bug instead of letting the attention backend infer a different
        interpretation from stale CPU placeholders.
        """
        num_rows = int(num_rows)
        qlen = int(qlen)
        if num_rows != self.num_mtp_verify_rows:
            raise ValueError(
                f"GPU patch rows {num_rows} != MTP rows {self.num_mtp_verify_rows}"
            )
        if any(n != qlen for n in self.query_lens[:num_rows]):
            raise ValueError(
                f"GPU patch qlen {qlen} disagrees with plan {self.query_lens[:num_rows]}"
            )
        return replace(
            self,
            materialization=MetadataMaterialization.CPU_WITH_GPU_PATCH,
            gpu_patch_rows=num_rows,
            attention_metadata=None,
        )

    def prepare_attention(
        self,
        backend: Optional["QKVAttentionBackend"],
        input_data: "InputData",
    ) -> Any:
        """Build and retain backend metadata for the current forward."""
        if not self.device_buffers_ready:
            raise RuntimeError(
                "attention metadata requested before deferred MTP input was materialized"
            )
        if backend is None:
            self.attention_metadata = None
        else:
            self.attention_metadata = backend.prepare_metadata(input_data, self)
        return self.attention_metadata
