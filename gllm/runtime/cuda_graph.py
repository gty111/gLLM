"""CUDA graph capture and bucket-selection logic for the model runner.

Method bodies here were moved verbatim out of ``gllm.runtime.model_runner``;
:class:`CudaGraphMixin` is mixed into ``ModelRunner`` (and thereby
``OverlapModelRunner``) so every ``self`` reference and call site keeps its
original meaning.

Boundary with :mod:`gllm.runtime.piecewise_cuda_graph`: that module is the
piecewise compiler/runner proper (segment splitting, ``capture_bucket``,
``run``); this mixin owns the runner-side orchestration -- per-bucket decode
full-graph capture, piecewise bucket capture, bucket-size computation, and
bucket selection for DP decode replay. MTP draft/verify graph capture lives in
``gllm.speculative.mtp`` (:class:`MtpMixin`).
"""

import gc
from contextlib import nullcontext as _nullcontext
from typing import Optional

import torch
from logger import logger
from tqdm import tqdm

from gllm.distributed.parallel_state import (
    get_dp_size,
    get_local_rank,
    is_dp_attn,
    set_dp_forward_counts,
)
from gllm.runtime.sequence import GenerationSequence


class CudaGraphMixin:
    """CUDA graph capture/replay helpers mixed into ``ModelRunner``."""

    @staticmethod
    def _build_capture_sizes(max_bs: int):
        """Return power-of-two bucket sizes up to max_bs, in descending order.

        For example, max_bs=20 → [20, 16, 8, 4, 2, 1].
        We always include 1 as a floor bucket.
        """
        if max_bs <= 0:
            return []
        sizes = []
        s = 1
        while s <= max_bs:
            sizes.append(s)
            s *= 2
        # If max_bs is not itself a power of two, add it as the top bucket so
        # that batches of exactly max_bs can still use CUDA graph.
        if sizes[-1] != max_bs:
            sizes.append(max_bs)
        return list(reversed(sizes))

    @torch.inference_mode()
    def capture_graph(self, stream: Optional[torch.cuda.Stream] = None):
        """Capture per-bucket decode CUDA graphs.

        ``stream`` controls which CUDA stream the graph is captured on.
        ``torch.cuda.graph`` otherwise allocates a brand-new private stream
        each call, which is fine for kernels but interacts poorly with
        captured NCCL ops if replay later happens on a *different* stream
        (the symptom we hit in TP+overlap runs was gradual KV-cache drift
        between TP ranks surfacing as repetition loops). Subclasses that
        replay on a known stream (e.g. ``OverlapModelRunner.forward_stream``)
        should pass that same stream here so capture and replay agree.
        """
        # Raw ``CUDAGraph.capture_begin`` (used by the piecewise segment
        # runner) is illegal on CUDA's default stream. The overlap runner
        # supplies its persistent forward stream; the ordinary runner creates
        # one persistent capture stream and reuses it for every graph family.
        # Replaying the resulting graphs on the caller's current stream is
        # supported by PyTorch.
        if stream is None:
            stream = getattr(self, "_cuda_graph_capture_stream", None)
            if stream is None:
                stream = torch.cuda.Stream(device=torch.cuda.current_device())
                self._cuda_graph_capture_stream = stream

        iterator = self.capture_sizes if self._full_cuda_graph_on else []
        if get_local_rank() == 0 and self._full_cuda_graph_on:
            logger.info(
                f"Capturing decode full CUDA graphs for bucket sizes: {list(reversed(self.capture_sizes))}"
            )
            iterator = tqdm(
                self.capture_sizes, desc="Capturing Decode Full Graphs", ncols=100
            )
        memory_pool = torch.cuda.graph_pool_handle()

        # If the custom NVLink-P2P all-reduce is active, wrap the whole
        # capture in its ``capture()`` context so that, after all buckets
        # are captured, it broadcasts the per-rank IPC handles for the
        # buffers that ended up baked into the graphs. Without this,
        # graph replay on any rank-N>0 would try to dereference a local
        # pointer baked at capture time on rank 0 and crash. With NCCL
        # AR there's nothing to do (NCCL kernels handle their own IPC
        # internally), so a missing/disabled custom AR is a no-op.
        from gllm.distributed import get_custom_allreduce

        car = get_custom_allreduce()

        # Warm up lazy cuBLAS/Triton init on the capture stream. cuBLAS creates
        # its handle and per-stream workspace on first use; if that happens
        # mid-capture the implicit cudaMalloc is illegal and aborts the capture
        # (cudaErrorStreamCaptureInvalidated). The startup profile_run doesn't
        # survive the intervening memory_manager.init (~30 GB KV/SSM alloc), so
        # re-run it on the capture stream to force + sync that init first.
        self.profile_run(stream=stream)

        # Some FP8 backends (DeepGEMM, and FlashInfer's swapAB for M<32) JIT-
        # compile a distinct kernel per decode M-bucket on first use. That
        # compilation issues an implicit cudaMalloc, which is illegal mid-capture
        # and aborts the graph with cudaErrorStreamCaptureInvalidated. Run one
        # eager forward per bucket (outside the capture context) so every such
        # kernel is compiled *before* we capture it.
        try:
            from gllm.layers.quantization.fp8 import fp8_backend_requires_bucket_warmup

            warmup_per_bucket = fp8_backend_requires_bucket_warmup(
                self.model_loader.quantization_config
            )
        except Exception:  # noqa: BLE001
            warmup_per_bucket = False
        # In DP+EP every group captures each bucket with a uniform global batch
        # (``dp_size * size``): publish ``[size] * dp_size`` so the MoE layer's
        # gather/all-reduce is baked at the right static shape (SGLang MAX_LEN).
        dp_size = get_dp_size() if is_dp_attn() else 1

        def _set_dp_counts(size: int) -> None:
            if is_dp_attn():
                set_dp_forward_counts([size] * dp_size)

        try:
            if self._full_cuda_graph_on and warmup_per_bucket:
                for size in self.capture_sizes:
                    seqs = self.create_dummy_seqs(size)
                    self.input_data.cal_and_set_input(seqs=seqs)
                    if self.uses_mrope:
                        self.input_data.set_mrope_position(
                            torch.zeros((3, size), device="cpu")
                        )
                    _set_dp_counts(size)
                    self.forward()
                torch.cuda.synchronize()

            capture_ctx = car.capture() if car is not None else _nullcontext()
            with capture_ctx:
                if self._full_cuda_graph_on:
                    for size in iterator:
                        seqs = self.create_dummy_seqs(size)
                        self.input_data.cal_and_set_input(seqs=seqs)
                        if self.uses_mrope:
                            self.input_data.set_mrope_position(
                                torch.zeros((3, size), device="cpu")
                            )
                        _set_dp_counts(size)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            cuda_graph=g, pool=memory_pool, stream=stream
                        ):
                            self.forward()
                        self.size_to_graph[size] = g
                # MTP: capture the draft-step graph per bucket on the same
                # stream/pool (inside the custom-AR capture context) so replay in
                # ``_draft_chain_graph`` agrees on stream + IPC handles.
                if self._mtp_draft_graph:
                    self._capture_draft_graphs(memory_pool, stream)
                # MTP: capture the full verify forward per bucket (same stream/
                # pool/AR-context). This is the dominant MTP cost; replay in
                # ``_mtp_decode`` collapses the ~250ms eager forward to a graph.
                if self._mtp_verify_graph:
                    self._capture_verify_graphs(memory_pool, stream)
                if self._piecewise_runner is not None:
                    self._capture_piecewise_graphs(stream)
        finally:
            if is_dp_attn():
                set_dp_forward_counts(None)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    @torch.inference_mode()
    def _capture_piecewise_graphs(self, stream: Optional[torch.cuda.Stream]) -> None:
        """Capture every configured mixed-forward bucket during startup.

        Unlike decode, a piecewise graph's eager Attention/GDN breaks need real
        metadata, so each capture uses one temporary prefill request of exactly
        the bucket size. The graph-resident regions are independent of that
        request type; replay calls the eager breaks against the current mixed
        batch and its freshly prepared metadata.
        """
        runner = self._piecewise_runner
        if runner is None or not runner.capture_sizes:
            return

        sizes = list(reversed(runner.capture_sizes))
        iterator = sizes
        if get_local_rank() == 0:
            logger.info(
                "Capturing piecewise CUDA graphs for token bucket sizes: %s",
                list(reversed(sizes)),
            )
            iterator = tqdm(
                sizes,
                desc="Capturing Piecewise Graphs",
                ncols=100,
            )

        # Do allocator cleanup once. Per-bucket empty_cache/gc defeats graph
        # pool reuse and makes startup scale linearly with Python GC work.
        gc.collect()
        torch.cuda.empty_cache()
        stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with stream_ctx:
            for bucket in iterator:
                seq_id = -(1_000_000 + bucket)
                seq = GenerationSequence(seq_id, [1] * bucket, [], output_len=1)
                seq.prompt_len = bucket
                seq.computed_token_num = 0
                seq.to_compute_token_num = bucket
                self.memory_manager.pre_allocate_page([seq], cacheable=False)
                self.memory_manager.allocate_recurrent_slot(seq)
                try:
                    # Dynamic Attention/SSM boundaries are not executed while
                    # capturing, so graph-resident regions only need a correctly
                    # shaped activation. Skip token embedding, multimodal prep,
                    # and backend metadata construction for the synthetic row.
                    self.input_data.cal_and_set_input([seq])
                    capture_hidden = self.input_hidden_states[:bucket]
                    capture_hidden.zero_()
                    runner.capture_bucket(
                        self.input_data,
                        capture_hidden,
                    )
                finally:
                    self.memory_manager.free(seq)
                    self.embedding_cache.pop(seq_id, None)
                    self.disagg_embeds.pop(seq_id, None)

        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()

        # Bucket capture creates short-lived warmup activations and temporary
        # allocator blocks.  The graph-owned blocks remain pinned by the
        # shared graph pool, but returning unrelated cached blocks here avoids
        # charging one-time capture scratch to the steady-state server.  Do
        # this once after the whole family (never per bucket), so startup time
        # and graph-pool reuse are unaffected in the hot capture loop.
        gc.collect()
        torch.cuda.empty_cache()

    def _piecewise_input_embeddings(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Return explicit model inputs for a piecewise forward.

        Piecewise capture starts immediately after token embedding so every
        graph bucket has a single ``[bucket, hidden]`` input address. VL input
        preparation already materializes the exact embeddings in the shared
        runner buffer. Text-only models use their common ``embed_input_ids``
        API here; embedding remains eager because its row count is dynamic and
        it is a negligible fraction of prefill execution.
        """
        num_tokens = int(num_tokens)
        embedding_size = int(self.input_data.embedding_size)
        if embedding_size:
            if embedding_size != num_tokens:
                return None
            num_decode_tokens = sum(
                s.to_compute_token_num
                for s in self.input_data.seqs
                if s.computed_prompt
            )
            self._fixup_vl_decode_embeddings(num_decode_tokens)
            return self.input_hidden_states[:num_tokens]

        embed = getattr(self.model, "embed_input_ids", None)
        if embed is None:
            return None
        hidden_states = embed(self.input_data.tokens[:num_tokens])
        # Some VL wrappers return ``(text_embeddings, deepstack_embeddings)``.
        # The latter is published into a stable model-owned buffer by their
        # normal input-preparation path; only the primary embedding enters the
        # piecewise runner.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if hidden_states.shape[0] != num_tokens:
            return None
        return hidden_states

    def _run_generic_piecewise_forward(self, num_tokens: int) -> bool:
        """Run an ordinary prefill/mixed batch through piecewise graphs.

        Returns ``True`` only after producing ``output_hidden_states``. Every
        unsupported shape or request type is a non-fatal eager fallback.
        """
        runner = self._piecewise_runner
        if not self._piecewise_generic_on or runner is None or num_tokens <= 0:
            return False
        if not runner.can_run(num_tokens):
            return False
        # Visual embeddings and deepstack residuals have request-dependent
        # side buffers. Text-only traffic on a VL checkpoint is supported, but
        # actual media stays eager until those side buffers are bucketed too.
        if any(s.mm_contents is not None for s in self.input_data.seqs):
            return False

        self._prepare_attention_metadata(self.input_data)
        hidden_states = self._piecewise_input_embeddings(num_tokens)
        if hidden_states is None:
            return False
        with torch.profiler.record_function("gllm::generic_piecewise_forward"):
            output = runner.run(self.input_data, hidden_states)
        if output is None:
            return False
        self.output_hidden_states[:num_tokens].copy_(output)
        return True

    def dp_select_bucket(self, max_tokens: int) -> Optional[int]:
        """Pick the CUDA-graph bucket for a DP decode step, or ``None``.

        In DP+EP the graph bucket must be the *same* on every DP group (the
        global MoE batch is a static ``dp_size * bucket``), so the driver feeds
        the group-wide ``max_tokens`` here. Returns the smallest captured bucket
        ``>= max_tokens``, or ``None`` when graphs are disabled / the batch is
        larger than any captured bucket (caller then runs eager).
        """
        if not self._full_cuda_graph_on:
            return None
        padded_size = None
        for bucket in self.capture_sizes:
            if bucket >= max_tokens:
                padded_size = bucket
        if padded_size is not None and padded_size in self.size_to_graph:
            return padded_size
        return None

