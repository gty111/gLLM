"""MTP (Multi-Token Prediction) speculative decoding.

Method bodies here were moved verbatim out of ``gllm.runtime.model_runner``;
:class:`MtpMixin` is mixed into ``ModelRunner`` (and thereby
``OverlapModelRunner``) so every ``self`` reference and call site keeps its
original meaning.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
from attr import dataclass
from logger import logger
from tqdm import tqdm

from gllm.distributed.parallel_state import (
    get_ipc_tp_group,
    get_local_rank,
    get_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    is_dp_attn,
    is_first_pp_rank,
    is_last_pp_rank,
)
from gllm.runtime.forward_metadata import ForwardMetadataPlan
from gllm.runtime.input_data import InputData
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative.async_state import MtpAsyncBatchState, MtpAsyncCompletion
from gllm.speculative.gpu_prep import MTP_DRAFT_POS_OFFSET
from gllm.speculative.repetition_penalty import (
    SpeculativeRepetitionPenalty,
    apply_speculative_penalties,
)


@dataclass
class MtpQDist:
    """The MTP draft chain's proposal distribution ``q``, in one of two forms.

    * **dense** -- ``[nd, k, vocab]`` transformed probabilities (needed when a
      request leaves ``top_k`` unrestricted, so the support isn't bounded).
    * **sparse** -- ``vals``/``idx`` ``[nd, k, k_pad]``: q's top-k support and its
      token ids, plus ``drawn`` ``[nd, k]``, the probability of the token each
      step actually sampled. Anything the rejection accept needs is recoverable
      from this (q is exactly 0 outside the support), at 1/500th the bytes.
    """

    dense: Optional[torch.Tensor] = None
    vals: Optional[torch.Tensor] = None
    idx: Optional[torch.Tensor] = None
    drawn: Optional[torch.Tensor] = None


@dataclass
class MtpVerifyResult:
    """Outputs carried from target verification into draft acceptance."""

    logits: torch.Tensor
    hidden: torch.Tensor
    drafts: object
    prefill_tokens: list
    prefill_tokens_gpu: Optional[torch.Tensor] = None
    new_state_seq_ids: tuple = ()
    new_state_context_lens: Optional[torch.Tensor] = None
    new_state_tokens: Optional[torch.Tensor] = None
    new_state_hidden: Optional[torch.Tensor] = None


class MtpMixin:
    """MTP speculative-decoding methods mixed into ``ModelRunner``."""

    def mtp_speculate_batch(self, num_decodes: int) -> bool:
        """Should this step speculate, given the decode batch size?

        A pure function of ``num_decodes``, which every rank sees identically, so
        TP/PP ranks always agree without a collective (MTP's sync design requires
        TP-identical tokens). Callers that decline must also drop the fused
        relay -- see ``_mtp_drop_relay``.
        """
        if not self.mtp_enabled:
            return False
        # The decision is taken ONCE per iteration (``mtp_begin_iter``) and
        # cached: the four gate sites must agree within a step. If they could
        # disagree, a step whose prep was skipped as "fused MTP" could then take
        # the plain path -- running the plain forward on minimal input prep.
        cached = self._mtp_spec_decision
        if cached is not None and cached[0] == num_decodes:
            return cached[1]
        return self._mtp_decide(num_decodes)

    def _mtp_decide(self, num_decodes: int) -> bool:
        qlen = 1 + self._mtp_k
        fits_token_budget = (
            qlen > 1 and num_decodes * qlen <= self.max_num_batched_tokens
        )
        fits_perf_gate = self._mtp_max_batch <= 0 or num_decodes <= self._mtp_max_batch
        # DP-attention replicas must agree on the whole multi-forward MTP
        # cadence and publish different row counts for bootstrap, each draft
        # step, and verify. That protocol is not wired yet; entering MTP with
        # the ordinary decode counts can mismatch MoE collectives and deadlock.
        dec = not is_dp_attn() and fits_token_budget and fits_perf_gate
        self._mtp_spec_decision = (num_decodes, dec)
        return dec

    def mtp_begin_iter(self, num_decodes: Optional[int]) -> bool:
        """Open an iteration: decide once whether it speculates, and cache it.

        ``num_decodes=None`` marks a prefill / mixed / idle iteration, which never
        speculates. Called by both worker loops before any prep, so every later
        gate site reads the same decision.
        """
        if not num_decodes or not self.mtp_enabled:
            self._mtp_spec_decision = None
            return False
        return self._mtp_decide(num_decodes)

    def _mtp_drop_relay(self, seqs: Optional[List[GenerationSequence]] = None) -> None:
        """Invalidate fused relay state after a plain decode advance.

        A plain decode step advances every seq by one token without refreshing
        the relay, so the stashed ``(bonus_tok, bonus_hidden)`` no longer
        describes the seq's last position. Reusing it would seed the next draft
        from a stale hidden. ``seqs=None`` drops the whole batch (used when a
        pure-decode step declines speculation); mixed prefill+decode steps pass
        only the decode rows they actually advanced so unrelated relay entries
        remain reusable. Every dropped seq takes one padded-bootstrap step when
        it next enters MTP.
        """
        if not self._mtp_relay:
            return
        if seqs is None:
            self._mtp_relay = {}
            return
        for seq in seqs:
            self._mtp_relay.pop(seq.seq_id, None)

    def take_mtp_relay_token(self, seq_id: int) -> int:
        """Consume the GPU-relayed x1 when leaving the MTP pipeline.

        A completed mixed prefill normally hands x1 directly to its first MTP
        verify. If scheduling instead falls back to ordinary decode, x1 must
        become the request's one committed-but-uncached CPU token. The paired
        hidden state is no longer reusable after that transition, so removing
        the whole relay entry is intentional.
        """
        relay = self._mtp_relay.pop(seq_id, None)
        if relay is None:
            raise RuntimeError(f"missing materialized MTP relay for sequence {seq_id}")
        return int(relay[0])

    @staticmethod
    def mtp_sampling_compatible(seqs) -> bool:
        # Verify still does not collect generation logprobs. Repetition
        # penalties are applied causally in both draft and target sampling.
        return not any(
            getattr(s, "logprobs_enabled", False)
            for s in seqs
        )

    def mtp_prep_eligible(self, seqs: List[GenerationSequence]) -> bool:
        """True when :meth:`step_once` owns all input prep for an MTP step.

        A full-relay batch goes straight to draft/verify. A batch with fresh or
        relay-miss seqs first runs :meth:`_mtp_bootstrap_padded_verify`, which
        builds its own uniform ``1+k`` input. In both cases the ordinary
        scheduler-side one-token decode prep would be thrown away, so callers
        can install only the batch bookkeeping via
        :meth:`prepare_input_mtp`.
        """
        if not self.mtp_sampling_compatible(seqs):
            return False
        if not (self.mtp_enabled and not is_dp_attn() and is_last_pp_rank()):
            return False
        if not seqs or not seqs[-1].computed_prompt:
            return False  # prefill / mixed batch
        if not self.mtp_speculate_batch(len(seqs)):
            return False  # batch too large to profit from speculation
        return True

    def prepare_input_mtp(self, seqs: List[GenerationSequence]) -> None:
        """Minimal prep for a fused MTP step: batch bookkeeping only.

        Sets just what ``step_once``'s fused gate and ``_mtp_decode`` read
        (``seqs`` + the decode/prefill split). No per-token arrays, no H2D, no
        multimodal pass -- ``_mtp_decode``'s GPU-native prep fills the device
        buffers for the draft and verify forwards from scratch.
        """
        idata = self.input_data
        idata.seqs = seqs
        idata.embedding_size = 0
        idata.is_mtp_verify = False
        idata.num_mtp_verify_rows = 0
        idata.num_decodes = len(seqs)
        idata.num_decode_tokens = len(seqs)
        idata.num_prefills = 0
        idata.max_query_len = 1
        idata.install_forward_metadata_plan(ForwardMetadataPlan.deferred_mtp(len(seqs)))

    def mtp_async_can_chain(self, seqs: List[GenerationSequence]) -> bool:
        """Whether ``seqs`` can consume the predecessor wholly on the GPU.

        Chaining is a graph/GPU-prep optimization, never a correctness
        requirement.  A bucket miss drains and resumes from the CPU relay so
        eager fallbacks never see placeholder x1 token ids.
        """
        state = self._mtp_async_state
        n = len(seqs)
        return bool(
            n
            and state is not None
            and state.can_remap([s.seq_id for s in seqs])
            and self._mtp_gpu_prep_on
            and any(b >= n for b in self._draft_size_to_graph)
            and any(b >= n for b in self._verify_size_to_graph)
        )

    def mtp_async_remap(self, seqs: List[GenerationSequence]) -> bool:
        """Align persistent MTP state with the successor's scheduler rows."""
        state = self._mtp_async_state
        if state is None:
            raise RuntimeError("MTP async state is not installed")
        return state.remap([s.seq_id for s in seqs])

    def _record_mtp_metrics(self, nd: int, k: int, n_accepted: list) -> None:
        """Accumulate MTP acceptance stats and periodically log them (TP0 only).

        Metric definitions (so the numbers are comparable to common
        spec-decode reporting):

        * **draft acceptance rate** = accepted_draft_tokens / drafted_tokens.
          ``drafted_tokens = num_drafts * k`` (k proposals per draft/step).
        * **mean acceptance length** = 1 + accepted/num_drafts (the ``1`` is the
          always-committed target token x1, i.e. the "bonus"; so a value of
          ``k+1`` means every draft accepted).
        * **per-position acceptance rate**[i] = fraction of drafts whose accepted
          length reached position ``i`` (i in ``0..k-1``), i.e.
          ``count(n_accepted > i) / num_drafts`` -- the standard per-position
          histogram.

        ``num_drafts`` counts one draft *per sequence per step* (nd per call).
        Logging is time-based on a 1s window to match gLLM's scheduler status
        log period (``scheduler.py`` ``time.time() - log_time > 1``), and only on
        TP rank 0 to avoid N duplicate lines. Reset the window after each log.
        """
        import time as _time

        if not hasattr(self, "_mtp_m_drafts"):
            self._mtp_m_drafts = 0  # number of (seq,step) drafts
            self._mtp_m_accepted = 0  # total accepted draft tokens
            self._mtp_m_pos = [0] * k  # per-position accept counts
            self._mtp_m_t0 = _time.time()
        self._mtp_m_drafts += nd
        for na in n_accepted:
            self._mtp_m_accepted += na
            for i in range(na):
                self._mtp_m_pos[i] += 1

        if get_tp_rank() != 0:
            return
        now = _time.time()
        if now - self._mtp_m_t0 < 1.0:
            return
        drafts = self._mtp_m_drafts
        acc = self._mtp_m_accepted
        drafted_tokens = drafts * k
        rate = 100.0 * acc / drafted_tokens if drafted_tokens else 0.0
        mean_len = 1.0 + acc / drafts if drafts else 1.0
        per_pos = ", ".join(
            f"{(self._mtp_m_pos[i] / drafts if drafts else 0.0):.3f}" for i in range(k)
        )
        logger.info(
            "MTP metrics: Mean acceptance length: %.2f, Accepted: %d tokens, "
            "Drafted: %d tokens, Per-position acceptance rate: %s, "
            "Draft acceptance rate: %.1f%%",
            mean_len,
            acc,
            drafted_tokens,
            per_pos,
            rate,
        )
        # Sparse top-k window overflow (see ``_mtp_sparse_probs``). Read at most
        # once per log window so the hot path never syncs; a nonzero count means
        # probability ties spilled past ``top_k + _SPARSE_TIE_MARGIN`` and a few
        # tied tokens were dropped from p's support (tiny distribution skew, not a
        # correctness break) -- raise the margin if it ever shows up.
        if self._mtp_sampling_seen:
            n_of = int(self._mtp_tie_overflow)
            if n_of:
                logger.warning(
                    "MTP sparse top-k tie overflow on %d rows (margin=%d); "
                    "consider raising ModelRunner._SPARSE_TIE_MARGIN",
                    n_of,
                    self._SPARSE_TIE_MARGIN,
                )
                self._mtp_tie_overflow.zero_()
        self._mtp_sampling_seen = False
        # reset window
        self._mtp_m_drafts = 0
        self._mtp_m_accepted = 0
        self._mtp_m_pos = [0] * k
        self._mtp_m_t0 = now

    # ------------------------------------------------------------------
    # MTP rejection sampling (lossless speculative decoding under sampling)
    # ------------------------------------------------------------------
    def _mtp_rng_step(self, device):
        """Return a TP-synchronized ``torch.Generator`` seeded for this step.

        Every column driver runs the same deterministic schedule, so seeding
        from a per-runner step counter makes the seed identical on every TP
        rank -> identical draws -> TP-consistent committed tokens (the sampling
        analog of the greedy-argmax determinism the MTP sync path relies on).
        """
        if self._mtp_rng is None:
            self._mtp_rng = torch.Generator(device=device)
        # A fixed base keeps runs reproducible; the counter advances in lockstep
        # across ranks. 0x9E3779B9 (golden-ratio) spreads consecutive seeds.
        self._mtp_rng.manual_seed(
            0x9E3779B9 * (self._mtp_step + 1) & 0x7FFFFFFFFFFFFFFF
        )
        self._mtp_step += 1
        return self._mtp_rng

    def _mtp_bcast_tp(self, tok: torch.Tensor) -> torch.Tensor:
        """Broadcast an int64 token tensor from TP-rank-0 across the TP group.

        Rejection sampling draws random tokens, so unlike greedy argmax it is NOT
        deterministic across TP ranks (per-rank logits differ by fp all-reduce
        epsilon, and multinomial amplifies that into different token picks). To
        keep every column driver's committed tokens + draft-forward inputs
        identical (the invariant the MTP sync path relies on), TP-rank-0's draws
        win: sample only there, broadcast to peers. No-op for tp_size==1.

        Uses the SAME TP communicator (``get_tp_group``) as the model's
        all-reduces and ``run_batch_async``'s token broadcast, NOT the IPC group:
        NCCL's per-communicator FIFO ordering then implicitly serializes this
        broadcast against the surrounding forward all-reduces on every rank,
        which is exactly what prevents the cross-communicator ordering hazard
        that otherwise deadlocks (broadcast on one communicator racing the
        forward all-reduce on another when ranks reach them in different orders).
        """
        if get_tp_size() <= 1:
            return tok
        dist.broadcast(tok, src=get_rank() - get_tp_rank(), group=get_tp_group())
        return tok

    def _mtp_probs_from_logits(self, logits, seqs):
        """Apply the per-seq sampling transform (temp -> softmax -> top-k/top-p
        renorm) to ``logits`` [n, vocab], returning a proper prob distribution
        [n, vocab]. Mirrors ``Sampler.forward_gpu`` so the MTP draft dist ``q``
        and target dist ``p`` live on the SAME transformed space, which is what
        rejection sampling requires. ``seqs`` is aligned row-for-row with logits.
        """
        from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_prob

        dev = logits.device
        temps = torch.tensor(
            [s.temperature if s.temperature > 1e-5 else 1.0 for s in seqs],
            device=dev,
            dtype=torch.float32,
        ).unsqueeze(1)
        probs = torch.softmax(logits.float() / temps, dim=-1)
        top_ks = torch.tensor(
            [
                s.top_k if s.top_k != -1 else self.memory_manager.vocab_size
                for s in seqs
            ],
            device=dev,
            dtype=torch.int32,
        )
        top_ps = torch.tensor([s.top_p for s in seqs], device=dev, dtype=torch.float32)
        # Only renorm when some seq actually restricts (cheap guard).
        if int(top_ks.min().item()) < self.memory_manager.vocab_size:
            probs = top_k_renorm_prob(probs, top_ks)
        if float(top_ps.min().item()) < 1.0:
            probs = top_p_renorm_prob(probs, top_ps)
        return probs

    def _mtp_sample_params(self, seqs, dev):
        """Per-seq ``(temps[n,1], top_ks[n], top_ps[n])`` on the device.

        Staged through **persistent pinned** buffers with ``non_blocking`` copies.
        The obvious ``torch.tensor(list, device="cuda")`` form copies from
        *pageable* memory, which torch has to serialize with a
        ``cudaStreamSynchronize`` -- and since this runs right after the verify
        graph was enqueued, that sync blocks the host on the whole outstanding
        GPU queue. The torch profiler put 558 ms of a 2 s sampling window in
        exactly these three lines (3 syncs per MTP step, ~28% of the window),
        purely as a serialization bubble in the middle of the step.
        """
        V = self.memory_manager.vocab_size
        n = len(seqs)
        if self._sp_host_f is None:
            b = max(self.max_running_seqs, 1)
            self._sp_host_f = torch.empty(
                (2, b), dtype=torch.float32, device="cpu", pin_memory=True
            )
            self._sp_host_k = torch.empty(
                b, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self._sp_dev_f = torch.empty((2, b), dtype=torch.float32, device=dev)
            self._sp_dev_k = torch.empty(b, dtype=torch.int32, device=dev)
        hf, hk = self._sp_host_f.numpy(), self._sp_host_k.numpy()
        hf[0, :n] = [s.temperature if s.temperature > 1e-5 else 1.0 for s in seqs]
        hf[1, :n] = [s.top_p for s in seqs]
        hk[:n] = [s.top_k if s.top_k != -1 else V for s in seqs]
        self._sp_dev_f[:, :n].copy_(self._sp_host_f[:, :n], non_blocking=True)
        self._sp_dev_k[:n].copy_(self._sp_host_k[:n], non_blocking=True)
        return (
            self._sp_dev_f[0, :n].unsqueeze(1),
            self._sp_dev_k[:n],
            self._sp_dev_f[1, :n],
        )

    def _mtp_probs_static(self, logits, temps, top_ks, top_ps):
        """Graph-safe variant of ``_mtp_probs_from_logits``: all inputs are
        static GPU tensors (no python seq list, no ``.item()`` guards) so the
        whole thing is CUDA-graph capturable. Always applies top-k/top-p renorm
        kernels unconditionally (a seq that doesn't restrict passes top_k=vocab /
        top_p=1, which the kernels treat as no-ops). ``temps`` [n,1], ``top_ks``
        [n] int32, ``top_ps`` [n] float32.
        """
        from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_prob

        probs = torch.softmax(logits.float() / temps, dim=-1)
        probs = top_k_renorm_prob(probs, top_ks)
        probs = top_p_renorm_prob(probs, top_ps)
        return probs

    # Headroom over the largest per-request ``top_k`` when building the sparse
    # (top-k) distribution: the reference kernel is TIE-INCLUSIVE (it keeps every
    # token whose prob equals the k-th largest), and bf16 logits over a 248k
    # vocab tie often enough to matter -- measured support for ``top_k=20`` was
    # 20..24. ``torch.topk``'s cost is dominated by the vocab scan, so a fat
    # margin is nearly free (k=20 and k=64 both measured 0.16 ms at 64 rows).
    _SPARSE_TIE_MARGIN = 64
    # ``k_pad`` baked into the captured sparse sampled-draft graph. A batch whose
    # largest ``top_k`` needs a wider window falls back to the dense captured
    # graph (still correct, just slower), so this only has to cover the common
    # serving range (``top_k`` up to 64 with the tie margin on top).
    _sparse_kpad_capture = 128

    def _mtp_kpad(self, seqs) -> int:
        """Sparse top-k window width for this batch, computed on the HOST.

        The per-request ``top_k`` values are plain python ints, so taking the max
        here costs nothing -- doing it as ``int(top_ks.max().item())`` on the
        staged device tensor (as the first version did) inserted a
        ``cudaStreamSynchronize`` into every MTP step, which the torch profiler
        duly showed sitting in the critical path.
        """
        mx = 1
        for s in seqs:
            tk = s.top_k
            if tk is not None and tk > mx:
                mx = tk
        return min(mx + self._SPARSE_TIE_MARGIN, self.memory_manager.vocab_size)

    def _mtp_sparse_eligible(self, seqs) -> bool:
        """True when every seq's ``top_k`` fits the captured sparse window.

        The sparse path represents ``q``/``p`` by their top-k support, which is
        only exact when ``top_k`` is restricted -- an unrestricted request
        (``top_k == -1`` / vocab, i.e. top-p only) keeps the dense path.
        """
        cap = self._sparse_kpad_capture - self._SPARSE_TIE_MARGIN
        for s in seqs:
            tk = s.top_k
            if tk is None or tk <= 0 or tk > cap:
                return False
        return True

    def _mtp_sparse_probs(self, logits, temps, top_ks, top_ps, k_pad):
        """Top-k-sparse form of :meth:`_mtp_probs_static`.

        Returns ``(vals, idx)`` -- ``[n, k_pad]`` probabilities (descending, zero
        outside the kept support) and their token ids. Mathematically identical to
        the dense ``softmax -> top_k_renorm -> top_p_renorm`` chain (verified to
        1e-7 on non-tied logits), because
        ``softmax`` restricted to the kept set == the dense renormalization of
        that set, and ``keep`` is a prefix of the descending order.

        Dense costs one full-vocab softmax plus two renorm passes over
        ``[n, vocab]`` (1.5 ms at n=64, 3.1 ms at n=256 for this vocab); this is a
        single ``topk`` plus ``[n, k_pad]`` arithmetic (0.2 / 0.6 ms).

        Ties beyond ``k_pad`` would silently drop tokens the dense kernel keeps,
        so the (sync-free) overflow counter is accumulated on-device and reported
        by ``_record_mtp_metrics``.

        ``topk`` runs on the raw logits and only the selected ``[n, k_pad]`` slice
        is widened to fp32: ``topk`` costs the same for any ``k_pad`` in this range
        but scales with the bytes it scans, so selecting on bf16 halves it (0.31 ->
        0.16 ms at n=64, 1.01 -> 0.55 ms at n=256). Bit-identical to selecting on
        the widened logits -- the cast is lossless and order preserving.
        """
        vals, idx = torch.topk(logits, k_pad, dim=-1)  # descending
        vals = vals.float()
        # Tie-inclusive top-k: keep everything >= the top_k-th largest value.
        kth = vals.gather(1, (top_ks.long() - 1).clamp_(0, k_pad - 1).unsqueeze(1))
        keep = vals >= kth
        probs = torch.softmax(vals.masked_fill(~keep, float("-inf")) / temps, dim=-1)
        # top-p over the descending probs: keep the shortest prefix that reaches
        # ``top_p`` (exclusive cumsum < top_p), then renormalize.
        csum = probs.cumsum(dim=-1)
        probs = torch.where(
            (csum - probs) < top_ps.unsqueeze(1), probs, torch.zeros_like(probs)
        )
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Ties spilling past the window: ``keep`` reaching the last column means
        # more equal-valued tokens may exist beyond it.
        # Grammar masking can leave fewer than top_k finite logits. The -inf
        # padding then ties at the cutoff but carries no probability mass.
        self._mtp_tie_overflow += (keep[:, -1] & torch.isfinite(vals[:, -1])).sum()
        return probs, idx

    @staticmethod
    def _q_dense(dense):
        """Draft-distribution handle, dense form: ``dense`` is ``[nd, k, vocab]``."""
        return MtpQDist(dense=dense, vals=None, idx=None, drawn=None)

    @staticmethod
    def _q_sparse(vals, idx, drawn):
        """Draft-distribution handle, sparse form.

        ``vals``/``idx``: ``[nd, k, k_pad]`` top-k support of each step's ``q``;
        ``drawn``: ``[nd, k]`` probability of the token that step actually drew.
        """
        return MtpQDist(dense=None, vals=vals, idx=idx, drawn=drawn)

    def _gumbel_argmax(self, q):
        """Draw from ``q`` [n, vocab] via the Gumbel-max trick:
        ``argmax(q / Exp(1))``. Distributionally equal to ``torch.multinomial``
        but graph-safe -- the only randomness is ``exponential_`` (a capturable
        RNG kernel whose Philox offset advances correctly across graph replays),
        and ``multinomial``'s device-side distribution-validity assert (which a
        graph would capture + replay every step) is avoided. Uses the DEFAULT
        CUDA generator (the only one capturable inside ``torch.cuda.graph``).
        """
        noise = torch.empty_like(q, dtype=torch.float32).exponential_(1.0)
        noise.clamp_min_(torch.finfo(torch.float32).tiny)
        return (q.float() / noise).argmax(dim=-1).to(torch.int64)

    @torch.inference_mode()
    def _draft_chain_eager(
        self, decode_seqs, orig_tokens, x1, hidden, k, nd, gen=None, sparse=False
    ):
        """Eager k-step MTP draft chain (fallback / graph-disabled path).

        One per-step D2H (``.tolist()``) instead of a ``.item()`` per token --
        unavoidable in the eager path since the next step's positions/slots are
        rebuilt from python token_ids each step.

        Two modes, differing only in how each step's draft token is picked (the
        ``sample_step`` closure below; the seq-state advance, ``prepare_input``,
        ``mtp.forward`` and penalty application are shared):

        * **greedy** (``gen=None``): argmax. Returns ``drafts`` = per-seq
          ``[d1..dk]`` (CPU ints).
        * **sampled** (``gen`` given, rejection mode): draws each draft token
          from the per-seq transformed distribution ``q`` and records that ``q``
          so the accept step can compute ``min(1, p/q)`` and the residual
          ``(p-q)+``. Returns ``(drafts, q)`` where ``q`` is an
          :class:`MtpQDist` -- sparse (top-k support) when ``sparse``, dense
          ``[nd, k, vocab]`` otherwise.
        """
        mtp = self.model.mtp
        dev = hidden.device
        drafts_cols = [[] for _ in range(nd)]
        tok = torch.tensor(x1, device=dev, dtype=torch.int64)
        cur_hidden = hidden
        sampled = gen is not None
        q_steps, qv_steps, qi_steps, qd_steps = [], [], [], []
        penalties = getattr(self, "_mtp_penalties", None)
        penalty_history = penalties.history.clone() if penalties is not None else None

        if not sampled:
            def sample_step(logits):
                return logits.argmax(dim=-1).to(torch.int64)
        elif sparse:
            temps, top_ks, top_ps = self._mtp_sample_params(decode_seqs, dev)
            k_pad = self._mtp_kpad(decode_seqs)

            def sample_step(logits):
                qv, qi = self._mtp_sparse_probs(logits, temps, top_ks, top_ps, k_pad)
                col = torch.multinomial(qv, num_samples=1, generator=gen)
                tok = qi.gather(1, col).squeeze(1).to(torch.int64)
                tok = self._mtp_bcast_tp(tok)
                # After the TP broadcast the drawn token may come from rank 0, so
                # look its probability up by id rather than by column.
                qd = (qv * (qi == tok.unsqueeze(1)).to(qv.dtype)).sum(dim=1)
                qv_steps.append(qv)
                qi_steps.append(qi)
                qd_steps.append(qd)
                return tok
        else:
            def sample_step(logits):
                q = self._mtp_probs_from_logits(logits, decode_seqs)  # [nd, vocab]
                tok = (
                    torch.multinomial(q, num_samples=1, generator=gen)
                    .squeeze(1)
                    .to(torch.int64)
                )
                tok = self._mtp_bcast_tp(tok)
                q_steps.append(q)
                return tok

        for j in range(k):
            for i, s in enumerate(decode_seqs):
                s.computed_token_num = (
                    len(orig_tokens[i]) + j + MTP_DRAFT_POS_OFFSET
                )
                s.to_compute_token_num = 1
                s.to_compute_tokens = [x1[i] if j == 0 else drafts_cols[i][-1]]
            self.prepare_input(decode_seqs)
            # ``mtp.forward`` reaches the same attention backend as an ordinary
            # model forward.  The graph path prepares its static draft plan in
            # ``_draft_step_forward*``; eager fallback must prepare the dynamic
            # plan after every ``prepare_input`` as well.
            self._prepare_attention_metadata(self.input_data)
            out_hidden = mtp.forward(self.input_data, cur_hidden, tok)
            logits = mtp.logits_from_hidden(out_hidden)
            if penalties is not None:
                apply_speculative_penalties(
                    logits, penalty_history, penalties.values, tok[:, None], update_history=True,
                )
            # Sampled mode: draw one draft token per seq from q (TP-synced
            # generator), then broadcast TP-rank-0's picks so every rank feeds
            # the SAME token into the next draft forward (multinomial isn't
            # TP-deterministic).
            tok = sample_step(logits)
            tok_cpu = tok.tolist()
            for i in range(nd):
                drafts_cols[i].append(tok_cpu[i])
            cur_hidden = out_hidden
        if not sampled:
            return drafts_cols
        if sparse:
            return drafts_cols, self._q_sparse(
                torch.stack(qv_steps, dim=1),
                torch.stack(qi_steps, dim=1),
                torch.stack(qd_steps, dim=1),
            )
        return drafts_cols, self._q_dense(torch.stack(q_steps, dim=1))

    def _draft_chain_eager_sampled(
        self, decode_seqs, orig_tokens, x1, hidden, k, nd, gen, sparse=False
    ):
        """Sampled-mode shorthand for :meth:`_draft_chain_eager` (kept for
        callers/tests that predate the merge of the two eager chains)."""
        # Dispatch through the mixin class (not ``self._draft_chain_eager``):
        # tests drive this with a duck-typed ``SimpleNamespace`` runner.
        return MtpMixin._draft_chain_eager(
            self, decode_seqs, orig_tokens, x1, hidden, k, nd, gen=gen, sparse=sparse
        )

    @torch.inference_mode()
    def _draft_chain_graph(
        self, decode_seqs, orig_tokens, x1, hidden, k, nd, sampled=False, sparse=False
    ):
        """CUDA-graph k-step MTP draft chain.

        Replays the per-bucket draft-step graph (captured lazily by
        ``_capture_draft_graphs``) k times. Between replays, tok/hidden/
        positions/slot_mapping/seq_lens are advanced **in place on the GPU**
        (no Python/H2D/.item()), so the whole chain has zero per-step host
        overhead. The captured graph runs ``mtp.forward`` with the MTP head's
        ``prev_hidden``/``input_ids`` aliased to the static buffers
        ``self._d_hidden``/``self._d_tok``; its output token lands in
        ``self._d_next_tok`` and post-block hidden in ``self._d_out_hidden``.

        Two modes, sharing the bucket pick, the static-buffer fill and the
        replay-advance loop (all host-side code; the captured graph content
        was fixed at capture time):

        * **greedy** (``sampled=False``): replays the argmax draft step.
          Returns ``None`` and stashes the GPU draft tensor in
          ``self._drafts_gpu`` -- materializing it here is a blocking D2H
          (~0.55 ms at nd=64); host-side readers go through
          :meth:`_drafts_host`.
        * **sampled** (rejection mode): replays the Gumbel-max sampled draft
          step (top-k-sparse variant when ``sparse``). Between replays the
          drawn token is broadcast across TP (host side, OUTSIDE the graph):
          the captured default-generator RNG is not guaranteed identical
          across ranks. Returns ``(drafts, q)`` -- per-seq ``[d1..dk]`` CPU
          ints (the rejection accept walks drafts host-side) and an
          :class:`MtpQDist`.

        KV pages for the whole speculative window were pre-allocated once by
        ``_mtp_decode``, so the page tables are frozen for this step. Falls
        back to the eager chain when no captured bucket fits.
        """
        dev = hidden.device
        page_sz = self.memory_manager.page_size

        # Smallest captured bucket >= nd (sorted() ascending, take the first
        # match). ``sparse``: every request restricts top_k, so the batch can
        # use the captured top-k-sparse draft step (one topk instead of a
        # full-vocab softmax + two renorm passes per step).
        penalties = getattr(self, "_mtp_penalties", None)
        if not sampled:
            graphs = self._draft_size_to_graph
            if penalties is not None:
                graphs = self._draft_penalty_graphs["greedy"]
        else:
            graphs = (
                self._draft_size_to_graph_sampled_sparse
                if sparse
                else self._draft_size_to_graph_sampled
            )
            if penalties is not None:
                graphs = self._draft_penalty_graphs["sparse" if sparse else "dense"]
        bucket = None
        for b in sorted(graphs):
            if b >= nd:
                bucket = b
                break
        if bucket is None:
            # Fall back to eager if this batch size wasn't captured at init.
            if not sampled:
                return self._draft_chain_eager(
                    decode_seqs, orig_tokens, x1, hidden, k, nd
                )
            gen = self._mtp_rng_step(dev)
            return self._draft_chain_eager(
                decode_seqs, orig_tokens, x1, hidden, k, nd, gen=gen, sparse=sparse
            )
        g = graphs[bucket]
        if penalties is not None:
            self._stage_draft_penalties(penalties, nd, bucket)

        # Fill the static draft-input buffers IN PLACE for this step (the captured
        # graph reads these exact buffers). Padded rows [nd:bucket] are written
        # as dummy rows by the GPU prep (no throwaway ``GenerationSequence`` objects); the
        # CPU fallback still builds dummy decode seqs.
        gp = self._mtp_gpu_prep_batch(decode_seqs, orig_tokens, x1, bucket)
        if gp is not None:
            ForwardMetadataPlan.uniform_gpu(
                num_rows=bucket,
                qlen=1,
                is_mtp_verify=False,
            ).materialize(
                self._draft_input,
                gp.draft_materializer(seqs=decode_seqs),
            )
        else:
            # CPU fallback: the builders read the seq state, so put it in the
            # draft shape first (one new token at ``ctx``). ``_mtp_decode`` left
            # the seqs in the *verify* shape (1+k speculative tokens) for the
            # one-shot page allocation.
            for i, s in enumerate(decode_seqs):
                s.computed_token_num = len(orig_tokens[i]) + MTP_DRAFT_POS_OFFSET
                s.to_compute_token_num = 1
                s.to_compute_tokens = [x1[i]]
            pad_seqs = (
                self.create_dummy_seqs(bucket - nd, runtime=True) if bucket > nd else []
            )
            graph_seqs = list(decode_seqs) + pad_seqs
            self._draft_input.cal_and_set_input(graph_seqs)
        self._d_nd = bucket
        if gp is not None:
            # x1 already sits in the staged metadata on the device -> D2D copy
            # instead of another pageable H2D of the host ``x1`` list.
            self._d_tok[:nd].copy_(gp.x1_gpu(nd))
        else:
            self._d_tok[:nd].copy_(torch.tensor(x1, device=dev, dtype=torch.int64))
        if bucket > nd:
            self._d_tok[nd:bucket].zero_()
        self._d_hidden[:nd].copy_(hidden)
        if bucket > nd:
            self._d_hidden[nd:bucket].zero_()
        if sampled:
            # Fill per-seq sampling params into the static buffers the graph
            # reads. Via the pinned staging (D2D copies here) -- the previous
            # ``torch.tensor(list, device=cuda)`` form was three pageable H2Ds,
            # i.e. three implicit stream syncs per draft chain.
            _temps, _top_ks, _top_ps = self._mtp_sample_params(decode_seqs, dev)
            self._d_temp[:nd, 0].copy_(_temps.squeeze(1))
            self._d_topk[:nd].copy_(_top_ks)
            self._d_topp[:nd].copy_(_top_ps)
            if bucket > nd:  # padded rows: harmless greedy-ish params
                self._d_temp[nd:bucket].fill_(1.0)
                # ``top_k = 1`` (not ``vocab``): on the sparse path an
                # unrestricted ``top_k`` clamps the tie threshold to the last
                # column of the window, which always trips the tie-overflow
                # counter. Padded rows' output is discarded, so pick the value
                # that keeps the diagnostic meaningful.
                self._d_topk[nd:bucket].fill_(1)
                self._d_topp[nd:bucket].fill_(1.0)

        base_pos = self._d_base_pos[:bucket]
        base_pos.copy_(self._draft_input.positions[:bucket])
        block_table = self._draft_input.block_table[:bucket]
        row_idx = self._d_row_idx[:bucket]

        def _slot_for(pos):
            blk = block_table[row_idx, (pos // page_sz)]
            return blk.to(torch.int64) * page_sz + (pos % page_sz)

        step_q, step_qv, step_qi, step_qd = [], [], [], []
        for j in range(k):
            if j > 0:
                self._d_tok[:bucket].copy_(self._d_next_tok[:bucket])
                self._d_hidden[:bucket].copy_(self._d_out_hidden[:bucket])
                new_pos = base_pos + j
                self._draft_input.positions[:bucket].copy_(new_pos)
                self._draft_input.slot_mapping[:bucket].copy_(_slot_for(new_pos))
                # ``decode_seq_lens`` is MLA-only metadata (set in
                # ``_cal_mla_metadata``); non-MLA models advance only
                # ``seq_lens``, which the GDN/full-attn decode kernels read.
                if self.use_mla:
                    self._draft_input.decode_seq_lens[:bucket].add_(1)
                self._draft_input.seq_lens[:bucket].add_(1)
            g.replay()
            if sampled:
                # TP-sync the drawn token (Gumbel RNG isn't guaranteed identical
                # across ranks); broadcast BEFORE it seeds the next step's
                # forward.
                self._mtp_bcast_tp(self._d_next_tok[:bucket])
            self._d_drafts[:nd, j].copy_(self._d_next_tok[:nd])
            if sampled:
                if sparse:
                    step_qv.append(self._d_qv[:nd].clone())
                    step_qi.append(self._d_qi[:nd].clone())
                    # The broadcast above can replace this rank's drawn token
                    # with rank 0's, so re-derive the drawn probability by token
                    # id rather than trusting the in-graph ``_d_qd`` column
                    # lookup.
                    step_qd.append(
                        (
                            self._d_qv[:nd]
                            * (self._d_qi[:nd] == self._d_next_tok[:nd].unsqueeze(1))
                        ).sum(dim=1)
                    )
                else:
                    step_q.append(self._d_q[:nd].clone())

        # Stash the GPU draft tensor so the verify prep / accept step can take
        # the tokens straight from the device.
        self._drafts_gpu = self._d_drafts[:nd, :k]
        if not sampled:
            return None
        if sparse:
            q = self._q_sparse(
                torch.stack(step_qv, dim=1),  # [nd, k, k_pad]
                torch.stack(step_qi, dim=1),
                torch.stack(step_qd, dim=1),  # [nd, k]
            )
        else:
            q = self._q_dense(torch.stack(step_q, dim=1))  # [nd, k, vocab]
        mat = self._drafts_gpu.tolist()
        return [mat[i] for i in range(nd)], q

    @torch.inference_mode()
    def _ensure_draft_buffers(self):
        """Allocate the static MTP-draft-step buffers + aliasing InputData once."""
        if self._draft_input is not None:
            return
        dev = torch.cuda.current_device()
        B = max(self.capture_sizes)
        H = self.hidden_size
        dt = self.output_hidden_states.dtype
        self._d_tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self._d_hidden = torch.zeros((B, H), dtype=dt, device=dev)
        self._d_next_tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self._d_out_hidden = torch.zeros((B, H), dtype=dt, device=dev)
        # Host orchestration replays one captured draft step ``k`` times. Keep
        # the chain outputs and position helpers persistent so the hot path does
        # not allocate k clones, a stack output, a base-position clone, or a new
        # arange tensor on every speculative step.
        self._d_drafts = torch.empty((B, self._mtp_k), dtype=torch.int64, device=dev)
        self._d_base_pos = torch.empty(B, dtype=torch.long, device=dev)
        self._d_row_idx = torch.arange(B, dtype=torch.int64, device=dev)
        self._d_penalty_history = torch.ones(
            (B, self.memory_manager.vocab_size), dtype=self.memory_manager.dtype, device=dev,
        )
        self._d_penalty_values = torch.ones(B, dtype=self.memory_manager.dtype, device=dev)
        # Sampled-draft (rejection sampling) static buffers: per-seq sampling
        # params + the drawn q distribution the accept step reads. Vocab-wide q
        # is the only large one (B*vocab); allocated once, reused across replays.
        if self._mtp_can_sample:
            V = self.memory_manager.vocab_size
            self._d_temp = torch.ones((B, 1), dtype=torch.float32, device=dev)
            self._d_topk = torch.full((B,), V, dtype=torch.int32, device=dev)
            self._d_topp = torch.ones((B,), dtype=torch.float32, device=dev)
            self._d_q = torch.zeros((B, V), dtype=torch.float32, device=dev)
            # Sparse (top-k) draft distribution: ``[B, k_pad]`` values + token ids
            # + the probability of the token actually drawn. Replaces the
            # ``[B, vocab]`` dense ``q`` for top_k-restricted batches; see
            # ``_draft_step_forward_sampled_sparse``. 1.5 MB vs 63 MB.
            kp = self._sparse_kpad_capture
            self._d_qv = torch.zeros((B, kp), dtype=torch.float32, device=dev)
            self._d_qi = torch.zeros((B, kp), dtype=torch.int64, device=dev)
            self._d_qd = torch.zeros(B, dtype=torch.float32, device=dev)
        self._draft_input = InputData(
            max_running_seqs=self.max_running_seqs,
            max_seq_length=self.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )

    @torch.inference_mode()
    def _capture_draft_graphs(self, memory_pool, stream):
        """Capture one MTP draft-step graph per decode bucket (init-time).

        Mirrors the decode-graph capture: per bucket, set up the draft-input with
        dummy decode seqs (KV -> dummy page), seed the static ``_d_*`` buffers,
        warm up eager once (DeepGEMM/FlashInfer JIT), then capture
        ``_draft_step_forward`` on the shared capture ``stream``/``pool``. Replay
        (``_draft_chain_graph``) later updates the same buffers in place.
        """
        self._ensure_draft_buffers()
        iterator = self.capture_sizes
        if get_local_rank() == 0:
            logger.info(
                f"Capturing MTP draft CUDA graphs for bucket sizes: "
                f"{list(reversed(self.capture_sizes))}"
            )
            iterator = tqdm(
                self.capture_sizes, desc="Capturing MTP Draft Graphs", ncols=100
            )
        for bucket in iterator:
            seqs = self.create_dummy_seqs(bucket)
            self._draft_input.cal_and_set_input(seqs)
            self._d_nd = bucket
            # seed dummy head inputs
            self._d_tok[:bucket].zero_()
            self._d_hidden[:bucket].zero_()
            # warm up JIT outside capture
            self._draft_step_forward()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                self._draft_step_forward()
            self._draft_size_to_graph[bucket] = g

            # Also capture the sampled (rejection) draft step when enabled, so
            # sampling requests get a graphed draft too (Gumbel-max, graph-safe).
            if self._mtp_can_sample:
                self._d_temp[:bucket].fill_(1.0)
                self._d_topk[:bucket].fill_(self.memory_manager.vocab_size)
                self._d_topp[:bucket].fill_(1.0)
                self._draft_step_forward_sampled()
                torch.cuda.synchronize()
                gs = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=gs, pool=memory_pool, stream=stream):
                    self._draft_step_forward_sampled()
                self._draft_size_to_graph_sampled[bucket] = gs
                # Sparse (top-k) sampled variant, used whenever every request in
                # the batch restricts top_k (the common serving case). Captured
                # separately because ``k_pad`` is baked in; the dense graph above
                # stays as the fallback for unrestricted-top_k batches.
                self._d_topk[:bucket].fill_(
                    self._sparse_kpad_capture - self._SPARSE_TIE_MARGIN
                )
                self._draft_step_forward_sampled_sparse()
                torch.cuda.synchronize()
                gsp = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=gsp, pool=memory_pool, stream=stream):
                    self._draft_step_forward_sampled_sparse()
                self._draft_size_to_graph_sampled_sparse[bucket] = gsp
                self._d_topk[:bucket].fill_(self.memory_manager.vocab_size)

            # Separate graphs leave neutral requests on the original fast path.
            # The penalized variants read fixed-address scratch masks, updated
            # entirely on-device between draft steps and reset per chain.
            variants = [("greedy", self._draft_step_forward)]
            if self._mtp_can_sample:
                variants += [("dense", self._draft_step_forward_sampled),
                             ("sparse", self._draft_step_forward_sampled_sparse)]
            self._capture_draft_penalty = True
            try:
                for mode, forward in variants:
                    self._d_penalty_history[:bucket].fill_(1.0)
                    if mode == "sparse":
                        self._d_topk[:bucket].fill_(self._sparse_kpad_capture - self._SPARSE_TIE_MARGIN)
                    forward()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cuda_graph=graph, pool=memory_pool, stream=stream):
                        forward()
                    self._draft_penalty_graphs[mode][bucket] = graph
            finally:
                self._capture_draft_penalty = False

    def _stage_draft_penalties(self, penalties, nd, bucket):
        self._d_penalty_history[:nd].copy_(penalties.history)
        self._d_penalty_values[:nd].copy_(penalties.values)
        if bucket > nd:
            self._d_penalty_history[nd:bucket].fill_(1.0)
            self._d_penalty_values[nd:bucket].fill_(1.0)

    def _apply_draft_penalties(self, logits):
        if self._capture_draft_penalty:
            nd = self._d_nd
            apply_speculative_penalties(
                logits, self._d_penalty_history[:nd], self._d_penalty_values[:nd],
                self._d_tok[:nd, None], update_history=True,
            )

    @torch.inference_mode()
    def _draft_step_forward(self):
        """One MTP draft step over the static draft buffers (captured/replayed).

        Reads ``self._draft_input`` (positions/slot/seq_lens/block already set),
        ``self._d_hidden`` (prev hidden), ``self._d_tok`` (input token); writes
        argmax to ``self._d_next_tok`` and post-norm hidden to ``self._d_out_hidden``.
        """
        nd = self._d_nd
        mtp = self.model.mtp
        self._prepare_attention_metadata(self._draft_input)
        out_hidden = mtp.forward(
            self._draft_input, self._d_hidden[:nd], self._d_tok[:nd]
        )
        self._d_out_hidden[:nd].copy_(out_hidden)
        logits = mtp.logits_from_hidden(out_hidden)
        self._apply_draft_penalties(logits)
        tok = logits.argmax(dim=-1).to(torch.int64)
        self._d_next_tok[:nd].copy_(tok)

    @torch.inference_mode()
    def _draft_step_forward_sampled(self):
        """Sampled (rejection) MTP draft step over the static buffers.

        Like :meth:`_draft_step_forward` but draws the draft token from the
        temperature/top-k/top-p transformed distribution via Gumbel-max (graph-
        safe) instead of argmax, and stashes that distribution ``q`` into
        ``_d_q`` (the accept step needs it for ``min(1,p/q)`` + residual). The
        per-seq sampling params (``_d_temp``/``_d_topk``/``_d_topp``) are static
        buffers filled before each replay. RNG uses the default CUDA generator
        (advances across replays); TP consistency is enforced by broadcasting the
        drawn token between replays in :meth:`_draft_chain_graph` (host
        side, outside the graph).
        """
        nd = self._d_nd
        mtp = self.model.mtp
        self._prepare_attention_metadata(self._draft_input)
        out_hidden = mtp.forward(
            self._draft_input, self._d_hidden[:nd], self._d_tok[:nd]
        )
        self._d_out_hidden[:nd].copy_(out_hidden)
        logits = mtp.logits_from_hidden(out_hidden)
        self._apply_draft_penalties(logits)
        q = self._mtp_probs_static(
            logits, self._d_temp[:nd], self._d_topk[:nd], self._d_topp[:nd]
        )
        self._d_q[:nd].copy_(q)
        self._d_next_tok[:nd].copy_(self._gumbel_argmax(q))

    @torch.inference_mode()
    def _draft_step_forward_sampled_sparse(self):
        """Sparse (top-k) variant of :meth:`_draft_step_forward_sampled`.

        Same draw, but ``q`` is kept as its top-k support (``_d_qv`` values /
        ``_d_qi`` token ids, ``k_pad`` wide) instead of a dense ``[B, vocab]``
        row. Everything a rejection accept needs is preserved: the drawn token's
        own probability (``_d_qd``) and, for the residual ``(p-q)+``, q's values
        at any token id -- which can only be nonzero inside this support.

        Cost at 64 rows / 248k vocab: one ``topk`` + ``[B, k_pad]`` math (~0.2 ms)
        versus a full-vocab softmax + two renorm passes (~1.5 ms), and the Gumbel
        draw shrinks from ``[B, vocab]`` to ``[B, k_pad]``.
        """
        nd = self._d_nd
        mtp = self.model.mtp
        self._prepare_attention_metadata(self._draft_input)
        out_hidden = mtp.forward(
            self._draft_input, self._d_hidden[:nd], self._d_tok[:nd]
        )
        self._d_out_hidden[:nd].copy_(out_hidden)
        logits = mtp.logits_from_hidden(out_hidden)
        self._apply_draft_penalties(logits)
        qv, qi = self._mtp_sparse_probs(
            logits,
            self._d_temp[:nd],
            self._d_topk[:nd],
            self._d_topp[:nd],
            self._sparse_kpad_capture,
        )
        self._d_qv[:nd].copy_(qv)
        self._d_qi[:nd].copy_(qi)
        # Gumbel-max over the support, then map the column back to a token id.
        col = self._gumbel_argmax(qv).unsqueeze(1)
        self._d_qd[:nd].copy_(qv.gather(1, col).squeeze(1))
        self._d_next_tok[:nd].copy_(qi.gather(1, col).squeeze(1))

    def _create_dummy_verify_seqs(self, nd: int, qlen: int, ctx: int = 1):
        """Build ``nd`` dummy MTP-verify seqs (each: ``ctx`` cached + ``qlen`` new).

        Each seq references the memory manager's dummy page (all KV reads/writes
        land there harmlessly), is flagged ``_mtp_verify`` so ``cal_input``
        classifies the batch as verify (prefill-with-context, fp8 decode-sparse
        selector + kernel), and has ``computed_token_num=ctx`` /
        ``to_compute_token_num=qlen`` so ``prepare_input`` builds the exact
        uniform 1+k shape the captured graph expects.
        """
        dummy_page = self.memory_manager.dummy_page
        # Dummy SSM block table (all point at the dummy block 0) so the verify
        # metadata (2D block table) is built during graph capture with the same
        # shape as a real verify batch. num_accepted stays 1 (resume col 0).
        k1 = 1 + self._mtp_k
        # Dummy seq ids are offset high to avoid the common case of colliding
        # with a live request id, but correctness does NOT rely on that: we only
        # seed an embedding-cache stub when the id is currently ABSENT, remember
        # exactly which ids we inserted (``self._dummy_verify_cache_ids``), and
        # ``_drop_dummy_verify_cache`` removes only those after the forward. So a
        # dummy id can safely overlap a real one -- we never clobber or delete a
        # genuine entry.
        base_id = 1_000_000
        seeded_ids = []
        seqs = []
        for i in range(nd):
            sid = base_id + i
            s = GenerationSequence(sid, [1] * (ctx + qlen), [], output_len=1)
            # ``prompt_len = ctx`` => computed_prompt True => the seq is
            # decode-classified in the VL mm-prep path (reads a position-delta
            # stub from embedding_cache instead of trying to build image
            # embeddings for a text-only dummy). ``_mtp_verify`` still forces the
            # verify (prefill-with-context) ATTENTION shape independently.
            s.prompt_len = ctx
            npages = (ctx + qlen + self.page_size - 1) // self.page_size
            s.page_table = [dummy_page] * npages
            s.computed_token_num = ctx
            s.to_compute_token_num = qlen
            s._mtp_verify = True
            if self.memory_manager.ssm_segment is not None and self._mtp_k > 0:
                s.ssm_block_table = [0] * k1
                s.ssm_num_accepted = 1
            if self.memory_manager.recurrent_segment is not None:
                # The dummy row borrows the padding slot, which no live request
                # owns.
                s.recurrent_state_slot = 0
            # Seed a minimal embedding-cache stub so the VL mrope decode branch
            # (``mm_prepare_inputs``) finds a position delta for this dummy --
            # but ONLY if this id isn't already a live entry (never clobber).
            if self.use_mm and self.uses_mrope and sid not in self.embedding_cache:
                self.embedding_cache[sid] = EmbeddingInfo(
                    mrope_position_delta=torch.zeros(1, dtype=torch.long, device="cuda"),
                )
                seeded_ids.append(sid)
            seqs.append(s)
        self._dummy_verify_cache_ids = seeded_ids
        return seqs

    def _drop_dummy_verify_cache(self) -> None:
        """Remove the embedding-cache stubs seeded by the last
        ``_create_dummy_verify_seqs`` so they never leak / shadow a real seq id
        allocated later."""
        for sid in getattr(self, "_dummy_verify_cache_ids", ()):
            self.embedding_cache.pop(sid, None)
        self._dummy_verify_cache_ids = []

    @torch.inference_mode()
    def _capture_verify_graphs(self, memory_pool, stream):
        """Capture the full MTP-verify forward per decode bucket (init-time).

        The verify forward (target model over the uniform 1+k query per decode
        seq) is 99% of MTP step time and pure eager per-layer launch overhead.
        Capture ``self.forward()`` at each bucket into a graph keyed by bucket
        size; ``_mtp_decode`` pads the real verify batch up to a bucket and
        replays. Uses the fp8 decode-sparse verify kernel + batched selector,
        both graph-safe (the selector reads only static-buffer views; the kernel
        + ``get_mla_metadata`` already run inside the captured decode graph).
        """
        qlen = 1 + self._verify_k
        # ``capture_sizes`` is expressed in decode requests, while the verify
        # forward consumes ``bucket * (1+k)`` token rows. Keep verify graphs
        # within the same hard token capacity used by the scheduler and the
        # persistent activation buffers. Plain decode/draft graphs can still
        # retain the larger request buckets because their query length is one.
        verify_capture_sizes = [
            bucket
            for bucket in self.capture_sizes
            if bucket * qlen <= self.max_num_batched_tokens
        ]
        iterator = verify_capture_sizes
        if get_local_rank() == 0:
            logger.info(
                f"Capturing MTP verify CUDA graphs (qlen={qlen}) for bucket sizes: "
                f"{list(reversed(verify_capture_sizes))}"
            )
            iterator = tqdm(
                verify_capture_sizes,
                desc="Capturing MTP Verify Graphs",
                ncols=100,
            )
        for bucket in iterator:
            seqs = self._create_dummy_verify_seqs(bucket, qlen)
            self.input_data.cal_and_set_input(seqs=seqs)
            # warm up JIT (DeepGEMM/FlashInfer per-M-bucket) outside capture
            self.forward()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                self.forward()
                self._capture_verify_head_kv(bucket, qlen)
            self._verify_size_to_graph[bucket] = g
            self._drop_dummy_verify_cache()

    def _capture_verify_head_kv(self, bucket: int, qlen: int) -> None:
        """Fold the post-verify MTP head KV pass into the verify graph.

        Running it eagerly costs a full extra decoder-layer launch sequence
        plus an attention re-plan on every speculative step -- measurably more
        than the acceptance it buys.  Everything it needs is already static:
        the verify token buffer, the target hidden buffer ``self.forward()``
        just wrote, and the batch's own attention metadata.  So capture it as
        part of the same graph and pay only its arithmetic.

        The shift is exact inside a row; the value landing on each row's last
        column is a don't-care (that position is at or past the acceptance
        point and the next draft chain rewrites it before reading it).
        """
        if qlen < 2:
            return
        ntok = bucket * qlen
        if ntok > self._mtp_staging.refresh_tokens.numel():
            return
        src = self.input_data.tokens[:ntok].view(bucket, qlen)
        shifted = self._mtp_staging.refresh_tokens[:ntok].view(bucket, qlen)
        shifted[:, :-1].copy_(src[:, 1:])
        shifted[:, -1].fill_(1)
        self.model.mtp.forward(
            self.input_data,
            self.output_hidden_states[:ntok],
            shifted.reshape(-1),
        )

    @torch.inference_mode()
    def _verify_forward_graph(self, decode_seqs, orig_tokens, x1, drafts, nd, kk):
        """Run the verify forward via a captured graph if the batch fits a bucket.

        Sets up ``decode_seqs`` as the uniform ``1+kk`` verify batch, selects the
        smallest captured bucket >= nd, fills the static ``input_data`` buffers,
        replays, and returns ``(v_logits, query_start_loc, v_hidden)`` for the
        REAL rows -- matching the eager path's contract. Returns ``None`` when no
        bucket fits (caller falls back to eager).
        """
        qlen = 1 + kk
        if qlen != 1 + self._verify_k:
            return None  # only the captured qlen is supported
        bucket = None
        for b in sorted(self._verify_size_to_graph.keys()):
            if b >= nd:
                bucket = b
                break
        if bucket is None:
            return None

        # GPU-native prep (default): write the verify batch's per-token arrays
        # straight into the static graph buffers from the staged per-seq facts +
        # the GPU draft tensor -- no Python array rebuild, no dummy pad seqs, no
        # H2D of the token ids. ``drafts_gpu`` is threaded from the draft chain;
        # the eager draft path leaves it unset, so fall back to a one-shot H2D.
        gp = self._mtp_gpu_prep_batch(decode_seqs, orig_tokens, x1, bucket)
        if gp is not None:
            dg = getattr(self, "_drafts_gpu", None)
            if dg is None or tuple(dg.shape) != (nd, kk):
                dg = (
                    torch.tensor(
                        drafts, device=self.input_data.tokens.device, dtype=torch.int64
                    )
                    if kk
                    else self.input_data.tokens.new_zeros((nd, 0))
                )
            plan = ForwardMetadataPlan.uniform_gpu(
                num_rows=bucket,
                qlen=qlen,
                is_mtp_verify=True,
            )
            plan.materialize(
                self.input_data,
                gp.verify_materializer(
                    seqs=decode_seqs,
                    drafts_gpu=dg,
                ),
            )
        else:
            drafts = self._drafts_host(drafts, nd, kk)
            for i, s in enumerate(decode_seqs):
                s.computed_token_num = len(orig_tokens[i])
                s.to_compute_token_num = qlen
                s.to_compute_tokens = [x1[i]] + drafts[i]
                s._mtp_verify = True
            pad_seqs = (
                self._create_dummy_verify_seqs(bucket - nd, qlen) if bucket > nd else []
            )
            graph_seqs = list(decode_seqs) + pad_seqs
            self.prepare_input(graph_seqs)
        self._verify_size_to_graph[bucket].replay()
        # Drop the pad dummies' embedding-cache stubs immediately so they can
        # never shadow a real seq id allocated on a later step.
        self._drop_dummy_verify_cache()
        num_real = nd * qlen
        v_hidden = self.output_hidden_states[:num_real]
        # Return the raw verify LOGITS, not the argmax: the greedy accept wants
        # ``argmax`` while the rejection accept wants the transformed prob dist,
        # and computing them from one lm-head pass avoids a second full
        # ``[nd*qlen, vocab]`` GEMM + write (the rejection path used to call
        # ``logits_from_hidden`` again on the same hidden). Everything downstream
        # stays on the GPU; only a tiny ``[nd]``-shaped result is ever D2H'd.
        v_logits = self.model.logits_from_hidden(v_hidden)
        # query_start_loc over the REAL seqs only (uniform qlen).
        qsl = [i * qlen for i in range(nd + 1)]
        return v_logits, qsl, v_hidden

    @torch.inference_mode()
    def _mtp_bootstrap_padded_verify(self, decode_seqs):
        """Seed relay-miss seqs with a uniform verify-shaped target forward.

        A missing relay needs the ordinary target decode result: the token
        predicted after the seq's one uncached input and that input's hidden
        state. Instead of introducing a batch-x-1 forward shape, represent the
        bootstrap as the same fixed ``1+k`` query used by verify:

        * relay miss: ``[real uncached token, pad, ..., pad]``;
        * relay hit: a full dummy ``1+k`` query (its existing relay is kept).

        Causal execution makes row zero identical to a one-token decode; later
        rows cannot affect it. Their speculative KV/state writes are either to
        dummy storage or are overwritten immediately by the real MTP verify.
        This keeps every real MTP target forward on one uniform qlen and is the
        shape needed by a future DP-wide fixed-mode implementation.
        """
        nd = len(decode_seqs)
        qlen = 1 + self._mtp_k
        missing_pos = [
            i for i, s in enumerate(decode_seqs) if s.seq_id not in self._mtp_relay
        ]
        if not missing_pos:
            relay = [self._mtp_relay[s.seq_id] for s in decode_seqs]
            return torch.stack([r[1] for r in relay], dim=0), [r[0] for r in relay]

        # Reuse a captured verify graph when possible. Padding is expressed in
        # sequence buckets; every row still carries the fixed qlen above.
        graph_bucket = None
        if self._mtp_verify_graph and self._verify_size_to_graph:
            for candidate in sorted(self._verify_size_to_graph):
                if candidate >= nd:
                    graph_bucket = candidate
                    break
        bucket = graph_bucket or nd

        # Build all dummy rows in one call so their ids/cache stubs are unique.
        # Relay-hit positions select their corresponding dummy; relay misses
        # replace that row with the real seq mutated to a padded verify query.
        dummy_seqs = self._create_dummy_verify_seqs(bucket, qlen)
        graph_seqs = list(dummy_seqs)
        saved = {}
        missing_seqs = []
        for pos in missing_pos:
            seq = decode_seqs[pos]
            saved[pos] = (
                seq.token_ids,
                seq.computed_token_num,
                seq.to_compute_token_num,
                getattr(seq, "_mtp_verify", False),
            )
            # A decode bootstrap must have exactly one committed-but-uncached
            # input. The scheduler's decode contract provides it; accepting a
            # different shape here would make row zero no longer equivalent to
            # the ordinary decode path.
            uncached = len(seq.token_ids) - seq.computed_token_num
            if uncached != 1:
                raise RuntimeError(
                    f"MTP relay miss for seq {seq.seq_id} has {uncached} "
                    "uncached tokens; expected exactly one decode token"
                )
            seq.token_ids = seq.token_ids + [0] * (qlen - 1)
            seq.to_compute_token_num = qlen
            seq._mtp_verify = True
            self.memory_manager.pre_allocate_page([seq], cacheable=False)
            graph_seqs[pos] = seq
            missing_seqs.append(seq)

        try:
            self.prepare_input(graph_seqs)
            # VL prepare installs placeholder embeddings for computed-prompt
            # rows. A verify-shaped bootstrap has bucket*qlen such rows, not
            # merely ``bucket`` one-token decode rows.
            if self.use_mm and is_first_pp_rank():
                self._fixup_vl_decode_embeddings(bucket * qlen)
            if graph_bucket is not None:
                self._verify_size_to_graph[graph_bucket].replay()
            else:
                self.forward()

            num_missing = len(missing_pos)
            self._mtp_staging.bootstrap_rows_host_np[:num_missing] = [
                pos * qlen for pos in missing_pos
            ]
            first_rows = self._mtp_staging.bootstrap_rows_gpu[:num_missing]
            first_rows.copy_(
                self._mtp_staging.bootstrap_rows_host[:num_missing], non_blocking=True
            )
            # Clone before the following real verify overwrites the persistent
            # output buffer.
            bootstrap_hidden = self.output_hidden_states[: bucket * qlen]
            missing_hidden = bootstrap_hidden.index_select(0, first_rows).clone()
            missing_logits = self.model.logits_from_hidden(missing_hidden)
        finally:
            for pos, (tokens, computed, to_compute, was_verify) in saved.items():
                seq = decode_seqs[pos]
                seq.token_ids = tokens
                seq.computed_token_num = computed
                seq.to_compute_token_num = to_compute
                seq._mtp_verify = was_verify
            self._drop_dummy_verify_cache()

        # Sample only the relay misses, using their real histories and sampling
        # parameters. Then restore the full batch bookkeeping for _mtp_decode.
        self.input_data.seqs = missing_seqs
        self.input_data.prepare_sample()
        missing_x1_gpu = self.sampler.forward_gpu(missing_logits, self.input_data)
        if get_tp_size() > 1:
            self._mtp_bcast_tp(missing_x1_gpu)
        self.sampler.stage_structured_feedback(missing_x1_gpu, missing_seqs)
        missing_x1 = missing_x1_gpu.tolist()
        self.prepare_input_mtp(decode_seqs)

        hidden_rows = []
        x1 = []
        miss_idx = 0
        for seq in decode_seqs:
            relay = self._mtp_relay.get(seq.seq_id)
            if relay is not None:
                x1.append(relay[0])
                hidden_rows.append(relay[1])
            else:
                x1.append(missing_x1[miss_idx])
                hidden_rows.append(missing_hidden[miss_idx])
                miss_idx += 1
        return torch.stack(hidden_rows, dim=0), x1

    def _mtp_mrope_deltas(self, seqs) -> Optional[List[int]]:
        """Per-seq mrope position delta (Qwen-VL family), or ``None``.

        Mirrors the decode branch of ``_mm_prepare_cpu``: a decode token's mrope
        position is ``computed_token_num + delta`` on all three rows, with
        ``delta`` stashed in the embedding cache at prefill time. A text-only
        prompt has ``delta == 0``, so a missing entry is not fatal here.
        """
        if not self.uses_mrope:
            return None
        out = []
        for s in seqs:
            info = self.embedding_cache.get(s.seq_id)
            delta = getattr(info, "mrope_position_delta", None) if info else None
            out.append(int(delta) if delta is not None else 0)
        return out

    def _mtp_gpu_prep_batch(self, decode_seqs, orig_tokens, x1, bucket):
        """Stage this MTP step's per-seq facts, returning the prep helper.

        Returns ``None`` when GPU-native prep is disabled (``GLLM_MTP_GPUPREP=0``)
        so callers fall back to the CPU ``cal_input`` builders. Staging is
        memoized per (step, bucket), so calling this from both the draft and the
        verify phase costs one pass.
        """
        gp = self._mtp_gpu_prep
        if gp is None or not self._mtp_gpu_prep_on:
            return None
        async_state = None
        if torch.is_tensor(x1):
            candidate = self._mtp_async_state
            if candidate is not None and candidate.matches(
                [s.seq_id for s in decode_seqs]
            ):
                async_state = candidate
        gp.push_meta(
            decode_seqs,
            bucket,
            epoch=self._mtp_prep_epoch,
            ctx_lens=[len(t) for t in orig_tokens],
            # Host values are only staging fallbacks for a chained step; the
            # authoritative context/x1/resume column is overwritten D2D below.
            x1=([0] * len(decode_seqs) if torch.is_tensor(x1) else x1),
            dummy_page=self.memory_manager.dummy_page,
            mrope_deltas=self._mtp_mrope_deltas(decode_seqs),
            ctx_lens_gpu=(async_state.context_lens if async_state else None),
            x1_gpu=(
                async_state.relay_tokens
                if async_state
                else (x1 if torch.is_tensor(x1) else None)
            ),
            num_accepted_gpu=(async_state.resume_num_accepted if async_state else None),
        )
        return gp

    # ------------------------------------------------------------------
    # MTP draft-head KV synchronization
    # ------------------------------------------------------------------
    # The Qwen3.5 MTP head is a full-attention decoder layer that owns one KV
    # layer (``kv_layer_id == num_kv_layers``) of the SHARED paged arena and
    # attends over the sequence's whole context.  The arena is allocated with
    # ``torch.empty`` and its pages are recycled between requests, so any
    # position the head never wrote returns a previous tenant's KV -- a state
    # validity bug, not merely a weak draft.  Every target forward therefore
    # replays the head over the tokens it just consumed (vLLM's drafter "first
    # pass"), which also rewrites the entries of accepted draft tokens from the
    # target's hidden states instead of the draft chain's own.
    #
    # Entry convention (EAGLE / vLLM): the entry at position ``p`` is the pair
    # ``(target_hidden[p], embed(token[p+1]))`` and its output predicts
    # ``token[p+2]``.  The head therefore consumes the token stream shifted
    # left by one, and a draft step for a context of length ``ctx`` writes at
    # ``ctx-1`` (``MTP_DRAFT_POS_OFFSET``).

    def _mtp_kv_sync_active(self, input_data) -> bool:
        """Whether a head KV pass may run over ``input_data`` this forward."""
        if not self.mtp_enabled or not is_last_pp_rank():
            return False
        if is_dp_attn():
            # DP-attention never enters the speculative path, so nothing ever
            # reads the head's KV -- and the idle-group dummy rows here are not
            # real requests.
            return False
        if getattr(self.model, "mtp", None) is None:
            return False
        if any(getattr(s, "mm_contents", None) is not None for s in input_data.seqs):
            # A placeholder image token id does not embed to the feature the
            # target actually consumed, so replaying the head over it would
            # write KV the draft must not read.  Skip (and say so once).
            if not self._mtp_kv_sync_mm_warned:
                self._mtp_kv_sync_mm_warned = True
                logger.warning(
                    "MTP head KV sync skips multimodal prompts; speculative "
                    "acceptance for those requests stays degraded."
                )
            return False
        return True

    @torch.inference_mode()
    def _mtp_sync_kv(
        self,
        input_data,
        target_hidden,
        *,
        tail_seqs=(),
        tail_next_tokens=None,
        num_verify_rows: int = 0,
    ) -> None:
        """Replay the MTP head over a target batch to refresh its KV layer.

        The head reuses the batch's existing positions / ``slot_mapping`` /
        page tables, so the only thing rebuilt is the token stream it consumes:
        a global left shift, plus a per-row boundary fixup.

        * ``num_verify_rows`` leading rows are uniform MTP verify rows.  Their
          shift is exact inside the row; the value that lands on each row's
          LAST position is a don't-care because that position is at or past the
          step's acceptance point, and the next draft chain overwrites it
          before any attention reads it.
        * ``tail_seqs`` are the ordinary (prefill / decode) rows that follow.
          Their last position takes the next prompt token at a chunk boundary,
          or the token the target just sampled once the prompt is finished.
          ``tail_next_tokens`` may be a host list or a device tensor.

        The head's logits are discarded; the sole effect is its KV write.
        """
        if not self._mtp_kv_sync_active(input_data):
            return
        plan = input_data.forward_metadata_plan
        if plan is None:
            return
        ntok = int(plan.num_tokens)
        if ntok <= 0 or ntok > self._mtp_staging.refresh_tokens.numel():
            return
        if target_hidden.shape[0] < ntok:
            return
        qsl_cpu = input_data.query_start_loc_cpu
        tail_seqs = list(tail_seqs)
        if tail_seqs:
            if qsl_cpu is None or qsl_cpu.shape[0] < num_verify_rows + len(tail_seqs) + 1:
                return
            if len(tail_seqs) > self._mtp_staging.capacity:
                return

        shifted = self._mtp_staging.refresh_tokens[:ntok]
        tokens = input_data.tokens[:ntok]
        if ntok > 1:
            shifted[:-1].copy_(tokens[1:])
        shifted[-1:].fill_(1)

        if tail_seqs:
            gpu_next = tail_next_tokens if torch.is_tensor(tail_next_tokens) else None
            host_next = None
            if gpu_next is None:
                host_next = (
                    list(tail_next_tokens) if tail_next_tokens is not None else []
                )
            patches = []
            gpu_pairs = []
            for j, seq in enumerate(tail_seqs):
                row = num_verify_rows + j
                last = int(qsl_cpu[row + 1]) - 1
                if not 0 <= last < ntok:
                    return
                end = seq.computed_token_num + seq.to_compute_token_num
                if end < seq.prompt_len:
                    # Chunk boundary: the next token is a known prompt token.
                    patches.append((last, int(seq.token_ids[end])))
                elif gpu_next is not None:
                    gpu_pairs.append((last, j))
                else:
                    patches.append((last, int(host_next[j])))
            self._mtp_staging.patch_shifted_tokens(
                shifted, patches, gpu_next, gpu_pairs
            )

        # A replayed CUDA graph leaves the plan's python-side attention object
        # empty; re-prepare it before this eager single-layer pass.
        self._prepare_attention_metadata(input_data)
        self.model.mtp.forward(input_data, target_hidden[:ntok], shifted)

    def _drafts_host(self, drafts, nd: int, kk: int):
        """Host-side ``[nd][kk]`` draft token ids, materializing on demand.

        The graph draft chain returns ``None`` and leaves its output on the GPU
        (``self._drafts_gpu``) so the hot path never syncs mid-step. Only the
        CPU-side fallbacks (eager verify) need the host copy; they pay the D2H
        here.
        """
        if drafts is not None:
            return drafts
        dg = getattr(self, "_drafts_gpu", None)
        if dg is None or nd == 0 or kk == 0:
            return [[] for _ in range(nd)]
        return dg[:nd, :kk].tolist()

    @torch.inference_mode()
    def _mtp_verify_target(
        self,
        decode_seqs,
        orig_tokens,
        x1_tokens,
        x1,
        drafts,
        nd: int,
        kk: int,
        extra_prefill_seqs,
        async_accept: bool,
    ) -> MtpVerifyResult:
        """Verify a draft batch, optionally fused with a ragged prefill suffix."""
        if (
            not extra_prefill_seqs
            and self._mtp_verify_graph
            and self._verify_size_to_graph
            and nd <= max(self._verify_size_to_graph.keys())
        ):
            with torch.profiler.record_function("gllm::mtp_target_verify_graph"):
                graph_result = self._verify_forward_graph(
                    decode_seqs, orig_tokens, x1_tokens, drafts, nd, kk
                )
            if graph_result is not None:
                logits, _, hidden = graph_result
                # Verify capture includes ``_capture_verify_head_kv``.
                return MtpVerifyResult(logits, hidden, drafts, [])

        # A chained mixed batch keeps graph-produced drafts on the GPU. CPU
        # sequence mutations below establish shapes only; the materializer
        # overwrites their token/position/KV metadata from device state.
        mixed_gpu_prep = None
        drafts_gpu_for_mixed = getattr(self, "_drafts_gpu", None)
        if (
            extra_prefill_seqs
            and drafts is None
            and drafts_gpu_for_mixed is not None
            and self._mtp_gpu_prep is not None
            and self._mtp_gpu_prep._staged[0] == self._mtp_prep_epoch
        ):
            mixed_gpu_prep = self._mtp_gpu_prep
            drafts = [[0] * kk for _ in range(nd)]
        else:
            drafts = self._drafts_host(drafts, nd, kk)

        for i, seq in enumerate(decode_seqs):
            seq.computed_token_num = len(orig_tokens[i])
            seq.to_compute_token_num = 1 + kk
            seq.to_compute_tokens = [x1[i]] + drafts[i]
            seq._mtp_verify = True

        self.memory_manager.pre_allocate_page(decode_seqs, cacheable=False)
        verify_and_prefill = list(decode_seqs) + extra_prefill_seqs
        self.prepare_input(verify_and_prefill)
        if mixed_gpu_prep is not None:
            plan = self.input_data.forward_metadata_plan
            if plan is None:
                raise RuntimeError("mixed MTP forward has no metadata plan")
            plan = plan.with_gpu_patch(num_rows=nd, qlen=1 + kk)
            plan.materialize(
                self.input_data,
                mixed_gpu_prep.mixed_patch_materializer(
                    drafts_gpu=drafts_gpu_for_mixed
                ),
            )

        total_tokens = int(self.input_data.tokens_cpu.shape[0])
        all_hidden = None
        if (
            self._piecewise_runner is not None
            and self.input_data.embedding_size == total_tokens
            and all(seq.mm_contents is None for seq in verify_and_prefill)
            and self._piecewise_runner.can_run(total_tokens)
        ):
            self._prepare_attention_metadata(self.input_data)
            num_decode_tokens = sum(
                seq.to_compute_token_num
                for seq in self.input_data.seqs
                if seq.computed_prompt
            )
            self._fixup_vl_decode_embeddings(num_decode_tokens)
            mixed_embeddings = self.input_hidden_states[:total_tokens]
            with torch.profiler.record_function("gllm::mtp_target_mixed_piecewise"):
                all_hidden = self._piecewise_runner.run(
                    self.input_data, mixed_embeddings
                )
        if all_hidden is None:
            with torch.profiler.record_function("gllm::mtp_target_mixed_eager"):
                self.forward()
            all_hidden = self.output_hidden_states[:total_tokens]

        num_verify_tokens = nd * (1 + kk)
        verify_hidden = all_hidden[:num_verify_tokens]
        prefill_tokens = []
        prefill_tokens_gpu = None
        new_state_seq_ids = ()
        new_state_context_lens = None
        new_state_tokens = None
        new_state_hidden = None

        if extra_prefill_seqs:
            num_extra = len(extra_prefill_seqs)
            last_rows = self._mtp_staging.mixed_last_rows[:num_extra]
            last_rows.copy_(
                self.input_data.query_start_loc[nd + 1 : nd + num_extra + 1]
            )
            last_rows.sub_(1)
            prefill_hidden = all_hidden.index_select(0, last_rows)

            # Project only verify rows and each prefill's final row. Projecting
            # every prompt token can otherwise allocate several GiB of logits.
            selected_hidden = torch.cat((verify_hidden, prefill_hidden), dim=0)
            selected_logits = self.model.logits_from_hidden(selected_hidden)
            verify_logits = selected_logits[:num_verify_tokens]
            prefill_logits = selected_logits[num_verify_tokens:]

            self.input_data.seqs = extra_prefill_seqs
            mixed_greedy = all(
                seq.top_k == 1 and seq.repetition_penalty == 1.0
                for seq in extra_prefill_seqs
            )
            if mixed_greedy:
                self.input_data.needs_repetition_penalty = False
            else:
                self.input_data.prepare_sample()
            prefill_tokens_gpu = self.sampler.forward_gpu(
                prefill_logits, self.input_data
            )
            if get_tp_size() > 1:
                self._mtp_bcast_tp(prefill_tokens_gpu)
            self.sampler.stage_structured_feedback(
                prefill_tokens_gpu, extra_prefill_seqs,
                stream=self.copy_stream if async_accept else None,
            )

            if async_accept:
                completed = [
                    i
                    for i, seq in enumerate(extra_prefill_seqs)
                    if seq.computed_token_num + seq.to_compute_token_num
                    >= seq.prompt_len
                ]
                if completed:
                    nc = len(completed)
                    completed_gpu = self._mtp_staging.mixed_new_idx[:nc]
                    new_state_context_lens = self._mtp_staging.mixed_new_ctx[:nc]
                    idx_host = self._mtp_staging.mixed_new_idx_host[:nc]
                    ctx_host = self._mtp_staging.mixed_new_ctx_host[:nc]
                    self._mtp_staging.mixed_new_idx_host_np[:nc] = completed
                    new_state_seq_ids = tuple(
                        extra_prefill_seqs[i].seq_id for i in completed
                    )
                    self._mtp_staging.mixed_new_ctx_host_np[:nc] = [
                        extra_prefill_seqs[i].computed_token_num
                        + extra_prefill_seqs[i].to_compute_token_num
                        for i in completed
                    ]
                    completed_gpu.copy_(idx_host, non_blocking=True)
                    new_state_context_lens.copy_(ctx_host, non_blocking=True)
                    new_state_tokens = prefill_tokens_gpu.index_select(
                        0, completed_gpu
                    )
                    new_state_hidden = prefill_hidden.index_select(0, completed_gpu)
            else:
                prefill_tokens = prefill_tokens_gpu.cpu().tolist()
        else:
            verify_logits = self.model.logits_from_hidden(verify_hidden)

        # Sampling above temporarily narrows ``input_data.seqs`` to prefill
        # rows. Restore the real mixed layout for the head KV replay.
        self.input_data.seqs = verify_and_prefill
        with torch.profiler.record_function("gllm::mtp_head_kv_sync"):
            self._mtp_sync_kv(
                self.input_data,
                all_hidden,
                tail_seqs=extra_prefill_seqs,
                tail_next_tokens=prefill_tokens_gpu,
                num_verify_rows=nd,
            )

        return MtpVerifyResult(
            logits=verify_logits,
            hidden=verify_hidden,
            drafts=drafts,
            prefill_tokens=prefill_tokens,
            prefill_tokens_gpu=prefill_tokens_gpu,
            new_state_seq_ids=new_state_seq_ids,
            new_state_context_lens=new_state_context_lens,
            new_state_tokens=new_state_tokens,
            new_state_hidden=new_state_hidden,
        )

    @torch.inference_mode()
    def _mtp_decode(
        self,
        hidden: torch.Tensor,
        x1_tokens: list,
        extra_prefill_seqs=None,
    ):
        """MTP speculative decode, optionally sharing verify with prefill.

        Returns one committed-token list per seq: ``x1`` plus the accepted draft
        prefix (1..k+1 tokens). The target bonus is relay-only and becomes the
        next step's ``x1``; it is never committed in the producing step.

        Correctness model (greedy): with a greedy target and greedy draft, the
        accepted tokens are exactly the tokens the target would have produced
        one-at-a-time, so committing them is identical to non-speculative greedy
        decoding. KV for committed tokens is written by the verify forward into
        the seqs' real slots; rejected-tail slots are simply overwritten next
        step (seq length only advances by the accepted count).
        """
        extra_prefill_seqs = list(extra_prefill_seqs or ())
        nd = self.input_data.num_decodes
        k = self._mtp_k
        mtp = self.model.mtp
        decode_seqs = self.input_data.seqs[:nd]
        dev = hidden.device
        # ``hidden`` is a view into the persistent output_hidden_states buffer;
        # clone so subsequent forwards (draft/verify) can't mutate it underfoot.
        seed_hidden = self._mtp_staging.seed_hidden[:nd]
        seed_hidden.copy_(hidden[:nd])
        hidden = seed_hidden

        x1_is_gpu = torch.is_tensor(x1_tokens)
        if x1_is_gpu:
            x1_gpu = x1_tokens[:nd].to(device=dev, dtype=torch.int64)
            # GenerationSequence mutations below exist only to reserve pages and provide
            # fallback shapes.  The graph fast path consumes x1 directly from
            # ``x1_gpu`` through MtpGpuPrep, so no D2H is needed here.
            x1 = [0] * nd
        else:
            x1 = [int(x1_tokens[i]) for i in range(nd)]
            self._mtp_staging.x1_host_np[:nd] = x1
            x1_gpu = self._mtp_staging.x1_gpu[:nd]
            x1_gpu.copy_(self._mtp_staging.x1_host[:nd], non_blocking=True)
        # x1 is already TP-synchronized by ``step_once`` (it broadcasts the
        # sampled x1 from TP-rank-0 before calling this method whenever sampling
        # is active), so no broadcast here. Runtime dispatch: if ANY seq in the
        # batch samples (temperature != 1 or top_k != 1) take the lossless
        # rejection path; an all-greedy batch takes the argmax fast path. No env
        # flag -- purely data-driven.
        _rej_active = self._mtp_can_sample and any(
            (s.temperature > 1e-5 and abs(s.temperature - 1.0) > 1e-5) or s.top_k != 1
            for s in decode_seqs
        )
        self._mtp_sampling_seen |= _rej_active
        # Greedy mixed batches use the same GPU completion as pure decode.  The
        # completion carries variable-width decode commits followed by the
        # one-token prefill samples, while freshly completed prefills are also
        # appended to the live request-id keyed relay state.
        _async_accept = self._mtp_async_publish and not _rej_active
        # Preserve the scheduler-owned sequence view while draft/verify uses
        # compact ``to_compute_tokens`` suffixes. ``token_ids`` itself remains
        # untouched: copying every request's full context each step made host
        # preparation slower as generation progressed.
        orig_tokens = [s.token_ids for s in decode_seqs]
        orig_ctn = [s.computed_token_num for s in decode_seqs]
        orig_tctn = [s.to_compute_token_num for s in decode_seqs]
        orig_compute_tokens = [s.to_compute_tokens for s in decode_seqs]
        self._mtp_penalties = SpeculativeRepetitionPenalty.prepare(self.memory_manager, decode_seqs)
        self._mtp_prep_epoch += 1
        # Pre-allocate the KV pages for the WHOLE speculative window once: the
        # draft chain writes tokens at ctx .. ctx+k-1 and the verify forward at
        # ctx .. ctx+k, so the verify shape's allocation covers both phases.
        # Doing it here (instead of once per phase) also freezes every seq's page
        # table for the rest of the step, which is what lets the GPU-native prep
        # stage the page tables a single time (see ``_mtp_gpu_prep_batch``).
        self.memory_manager.pre_allocate_page_for_lengths(
            decode_seqs, [len(tokens) + 1 + k for tokens in orig_tokens]
        )

        # --- Hybrid GDN recurrent-state: block-table column commit ---
        # The verify forward (GDN ``_forward_mtp_verify``) wrote each of the
        # ``1+k`` verify tokens' post-state into that column of every seq's SSM
        # block table (column 0 held the committed pre-x1 state on entry). After
        # the accept step decides how many drafts each seq kept (``na``), the
        # exact post-acceptance state (after ``1+na`` committed tokens) sits at
        # column ``na``. We promote it to column 0 so the plain decode /
        # snapshot paths keep reading column 0. Pure-attention models (DeepSeek
        # MTP) have no SSM segment and skip this.
        _ssm_seg = getattr(self.memory_manager, "ssm_segment", None)
        _has_gdn = _ssm_seg is not None and all(
            s.ssm_block_table is not None for s in decode_seqs
        )
        # A target bonus is not a verify INPUT, so it has no KV/GDN state yet.
        # Its draft seed, however, is the target hidden at the accepted verify
        # row, and the recurrent state *before* the bonus is the checkpoint in
        # column ``na``.  The accept path below promotes that checkpoint to
        # column zero.  Relaying (bonus, hidden) then lets the next qlen=1+k
        # verify consume the bonus exactly once. Together with the accepted
        # state-column promotion, this removes the steady-state qlen=1
        # full-model bootstrap.

        def restore():
            # NOTE: deliberately do NOT restore page_table. The verify forward
            # wrote each committed token's KV into the pages allocated during
            # verify; the scheduler will commit ``m`` tokens and must keep those
            # exact pages so the next decode step reads valid KV. Restoring the
            # page_table here would orphan the verify pages and let the scheduler
            # hand out different physical slots (garbage KV) -> divergence.
            for i, s in enumerate(decode_seqs):
                # Restore the scheduler-owned view; token_ids was never mutated.
                s.token_ids = orig_tokens[i]
                s.computed_token_num = orig_ctn[i]
                s.to_compute_token_num = orig_tctn[i]
                s.to_compute_tokens = orig_compute_tokens[i]
                s._mtp_verify = False

        # --- 1. Draft k tokens per seq by looping the MTP head. ---
        # Two paths: a CUDA-graph replay path (``_mtp_draft_graph``) that captures
        # one draft-step graph per decode bucket and replays it k times with
        # in-place GPU buffer advance (no per-step Python / H2D / .item()); and an
        # eager fallback. Both produce ``drafts`` = per-seq [d1..dk] on CPU.
        # ``_rej_active`` (computed above for the x1 broadcast) also selects the
        # sampling draft chain, which keeps the per-step draft dist ``q``.
        _use_rej = _rej_active
        q_dists = None
        # Sparse (top-k) rejection path: decided ONCE per step so the draft's ``q``
        # and the verify's ``p`` are always built by the same code (they must live
        # on the same transformed space for the accept test to be exact).
        _sparse = _rej_active and self._mtp_sparse_eligible(decode_seqs)
        # One TP-synced generator per MTP step, used by BOTH the eager sampled
        # draft chain (if taken) AND the accept-step residual/bonus draws below.
        # (The graph draft path uses the default CUDA generator internally and
        # broadcasts its tokens, so it doesn't consume ``gen``.)
        gen = self._mtp_rng_step(dev) if _use_rej else None
        # Reset the GPU-draft stash; only the greedy graph draft chain fills it
        # (the accept step reads it to skip an H2D of the host drafts list).
        self._drafts_gpu = None
        with torch.profiler.record_function("gllm::mtp_draft_chain"):
            if _use_rej:
                # Sampling draft chain: prefer the captured Gumbel-max graph
                # (draft forward + sampling in-graph, token broadcast between
                # replays); fall back to eager when graphs are off or the batch
                # exceeds the max bucket. Both draw from q and keep the per-step
                # q dist for accept.
                _sg = (
                    self._draft_size_to_graph_sampled_sparse
                    if _sparse
                    else self._draft_size_to_graph_sampled
                )
                if self._mtp_draft_graph and _sg and nd <= max(_sg.keys()):
                    drafts, q_dists = self._draft_chain_graph(
                        decode_seqs,
                        orig_tokens,
                        x1_tokens,
                        hidden,
                        k,
                        nd,
                        sampled=True,
                        sparse=_sparse,
                    )
                else:
                    drafts, q_dists = self._draft_chain_eager(
                        decode_seqs,
                        orig_tokens,
                        x1,
                        hidden,
                        k,
                        nd,
                        gen=gen,
                        sparse=_sparse,
                    )
            elif (
                self._mtp_draft_graph
                and self.capture_sizes
                and nd <= max(self.capture_sizes)
            ):
                drafts = self._draft_chain_graph(
                    decode_seqs, orig_tokens, x1_tokens, hidden, k, nd
                )
            else:
                drafts = self._draft_chain_eager(
                    decode_seqs, orig_tokens, x1, hidden, k, nd
                )
        restore()

        # --- 2. Verify: one base forward over [x1, d1..dk] per seq. ---
        kk = (k if drafts is None else len(drafts[0])) if nd else 0
        structured_inputs = None
        if any(getattr(s, "structured_output", None) is not None for s in decode_seqs):
            dg = self._drafts_gpu
            if dg is None:
                dg = torch.tensor(drafts, dtype=torch.int64, device=dev)
            contexts = (self._mtp_async_state.context_lens[:nd] if x1_is_gpu else
                        torch.tensor([len(t) for t in orig_tokens], dtype=torch.int64, device=dev))
            structured_inputs = self.sampler._structured.stage_speculative_inputs(
                contexts, x1_gpu, dg, stream=getattr(self, "copy_stream", None)
            )
        verify = self._mtp_verify_target(
            decode_seqs=decode_seqs,
            orig_tokens=orig_tokens,
            x1_tokens=x1_tokens,
            x1=x1,
            drafts=drafts,
            nd=nd,
            kk=kk,
            extra_prefill_seqs=extra_prefill_seqs,
            async_accept=_async_accept,
        )
        v_logits = verify.logits
        v_hidden = verify.hidden
        drafts = verify.drafts
        prefill_tokens = verify.prefill_tokens
        prefill_tokens_gpu = verify.prefill_tokens_gpu
        new_state_seq_ids = verify.new_state_seq_ids
        new_state_context_lens = verify.new_state_context_lens
        new_state_tokens = verify.new_state_tokens
        new_state_hidden = verify.new_state_hidden

        penalty_candidates = None
        if self._mtp_penalties is not None:
            dg = self._drafts_gpu
            if dg is None:
                dg = torch.tensor(drafts, dtype=torch.int64, device=dev)
            penalty_candidates = torch.cat((x1_gpu[:, None], dg), dim=1)
            self._mtp_penalties.verify(v_logits, penalty_candidates)

        structured_active = []
        if structured_inputs is not None:
            # Verify is already enqueued. Wait only for the earlier candidate
            # copy and predecessor acceptance, then build masks while it runs.
            with torch.profiler.record_function("gllm::mtp_grammar_ready"):
                host, ready = structured_inputs
                if ready is not None:
                    ready.synchronize()
                rows = host.tolist()
                structured_active = self.sampler._structured.mask_speculative(
                    v_logits, decode_seqs, None if x1_is_gpu else orig_tokens,
                    [row[1:] for row in rows], positions=[row[0] for row in rows],
                )

        # Target verification returns post-final-norm hidden states, matching
        # vLLM and sglang's MTP conditioning contract.
        # Rejection mode: the target dist ``p`` at each of the 1+kk verify
        # positions, in the SAME transformed space as the draft dist ``q``
        # (temperature + top-k/top-p renorm) -- that equality is what makes the
        # accept test distribution-lossless. Rows are laid out
        # [seq0: x1,d1..dk | seq1: ...]; the per-seq sampling params are expanded
        # on-device with ``repeat_interleave`` instead of building a 256-entry
        # python seq list per step.
        p_dists = None  # dense [num_v, vocab] (unrestricted-top_k batches)
        p_sparse = None  # (vals, idx) [nd, 1+kk, k_pad] (top_k-restricted)
        if _use_rej:
            num_v = nd * (1 + kk)
            temps, top_ks, top_ps = self._mtp_sample_params(decode_seqs, dev)
            rep = 1 + kk
            t_r = temps.repeat_interleave(rep, dim=0)
            k_r = top_ks.repeat_interleave(rep)
            p_r = top_ps.repeat_interleave(rep)
            if _sparse:
                # Only the top-k support can carry probability, so keep p in the
                # sparse form: one ``topk`` instead of a full-vocab softmax + two
                # renorm passes over ``[nd*(1+k), vocab]`` (3.1 ms -> 0.6 ms at
                # nd=64, and 254 MB of dense probs never materialized).
                k_pad = self._mtp_kpad(decode_seqs)
                pv, pi = self._mtp_sparse_probs(v_logits[:num_v], t_r, k_r, p_r, k_pad)
                p_sparse = (pv.view(nd, rep, -1), pi.view(nd, rep, -1))
            else:
                p_dists = self._mtp_probs_static(
                    v_logits[:num_v], t_r, k_r, p_r
                )  # [num_v, vocab]

        # --- 3. Greedy accept per seq. ---
        # Verify inputs per seq are [x1, d1..dk] at positions start..start+k.
        # v_pred[start+p] = target's greedy token AFTER consuming input p. So the
        # target's token following x1 is v_pred[start+0]; accept d_{p+1} iff it
        # equals v_pred[start+p]. Commit x1 always; append accepted drafts; the
        # bonus token is the target prediction at the first rejection (or after
        # the last accepted draft).
        # Commit x1 + the longest prefix of drafts the target agrees with. We do
        # NOT commit a trailing "bonus"/corrected token: relay it as the next
        # MTP step's x1, where it becomes a verify input and receives valid KV.
        # Every committed token here was a verify INPUT, so its KV is valid.
        # Size to ``nd`` (the real decode seqs captured at entry). The graph path
        # leaves ``self.input_data.seqs`` holding bucket-padded seqs, so it is no
        # longer a reliable count here -- use ``nd`` directly. This batch is pure
        # decode/verify (no real non-decode seqs), so there is no tail to fill.
        results = [None] * nd
        n_accepted = [0] * nd  # accepted DRAFT tokens per seq (excludes x1)
        new_relay = {}

        if _use_rej:
            # --- 3b. Rejection-sampling accept (distribution-lossless). ---
            # x1 is already a proper target sample (step_once's sampler), always
            # committed. For each draft d_p ~ q, accept with prob min(1, p/q);
            # on reject, resample the position from the residual (p-q)+ and stop;
            # if all accepted, sample a bonus from p at the last position.
            #
            # DON'T commit the bonus this step; relay (bonus_tok, bonus_hidden)
            # as the NEXT step's x1. Committing it both here and as next-step x1
            # would double-emit it. ``bonus_hidden`` is the verify hidden at the
            # last-accepted position (start+na), exactly the state the bonus was
            # sampled from, so it correctly seeds the next draft.
            # Fully vectorized on the GPU. The previous per-seq python loop read
            # ``float(q_dists[i,p,d])`` / ``float(px_all[start+p,d])`` -- each of
            # those is a device-scalar sync, up to ``2*nd*kk`` (384 at nd=64) per
            # step -- and then ran one ``torch.multinomial`` + ``.item()`` per
            # sequence. That accept phase measured 20.5 ms of a 34.6 ms sampling
            # step. Here every decision is a batched tensor op and the host learns
            # the outcome through ONE packed D2H at the end.
            qlen = 1 + kk
            seq_ar = self._mtp_staging.row_idx[:nd]
            sparse = p_sparse is not None
            if sparse:
                p3_vals, p3_idx = p_sparse  # [nd, qlen, k_pad] each
            else:
                p3 = p_dists.view(nd, qlen, p_dists.shape[-1])
            if kk > 0:
                dg = getattr(self, "_drafts_gpu", None)
                if dg is not None and tuple(dg.shape) == (nd, kk):
                    d_gpu = dg.to(torch.int64)
                else:
                    d_gpu = torch.tensor(drafts, device=dev, dtype=torch.int64)
                idx = d_gpu.unsqueeze(-1)
                if sparse:
                    # p(d_p): the drafted token either sits in p's kept support
                    # (pick its value) or outside it, where p is exactly 0 -- the
                    # same value the dense gather would return.
                    hit = (p3_idx[:, :kk] == idx).to(p3_vals.dtype)
                    p_d = (p3_vals[:, :kk] * hit).sum(dim=-1)  # [nd,kk] p(d_p)
                    # q(d_p) was recorded by the draft step that drew it.
                    q_d = q_dists.drawn
                else:
                    p_d = p3[:, :kk].gather(2, idx).squeeze(-1)  # [nd,kk] p(d_p)
                    q_d = q_dists.dense.gather(2, idx).squeeze(-1)  # [nd,kk] q(d_p)
                # accept d_p with prob min(1, p/q); q<=0 can only happen for a
                # token q never proposes, so treat it as certain acceptance
                # (matches the old scalar branch).
                ratio = torch.where(
                    q_d > 0,
                    (p_d / q_d.clamp_min(1e-30)).clamp_max(1.0),
                    torch.ones_like(p_d),
                )
                u = torch.rand((nd, kk), generator=gen, device=dev)
                accept = u < ratio  # [nd,kk] bool
                # Leading all-accept prefix length (same cumprod trick as greedy).
                na_gpu = torch.cumprod(accept.to(torch.int32), dim=1).sum(dim=1)
            else:
                d_gpu = None
                na_gpu = torch.zeros(nd, dtype=torch.int64, device=dev)
            na_l = na_gpu.to(torch.long)
            # Bonus draw. Rejected at position ``na`` (``na < kk``): draw from the
            # residual ``(p-q)+`` there. All accepted (``na == kk``): draw from p
            # at the tail position. Both are row ``na``, so one gather serves both
            # cases and a single batched multinomial does the draw.
            if sparse:
                # ``(p-q)+`` is supported inside p's kept set (outside it p == 0,
                # so the clamp is 0 there), which is why the residual only needs
                # p's ``k_pad`` columns plus q's values at those same token ids.
                p_row = p3_vals[seq_ar, na_l]  # [nd, k_pad]
                p_row_idx = p3_idx[seq_ar, na_l]  # [nd, k_pad]
            else:
                p_row = p3[seq_ar, na_l]  # [nd, V]
            if kk > 0:
                sel = na_l.clamp(max=kk - 1)
                if sparse:
                    # q's values at p's kept token ids. Everything outside q's own
                    # support is q == 0, so matching the two id lists (k_pad x
                    # k_pad per row, ~16k comparisons) is all that's needed.
                    q_row_v = q_dists.vals[seq_ar, sel]  # [nd, k_pad]
                    q_row_i = q_dists.idx[seq_ar, sel]  # [nd, k_pad]
                    match = q_row_i.unsqueeze(1) == p_row_idx.unsqueeze(2)
                    q_at_p = (match.to(q_row_v.dtype) * q_row_v.unsqueeze(1)).sum(-1)
                else:
                    q_at_p = q_dists.dense[seq_ar, sel]  # [nd, V]
                resid = torch.where(
                    (na_gpu < kk).unsqueeze(1),
                    (p_row - q_at_p).clamp_min(0),
                    p_row,
                )
                # Degenerate rows (p == q over the whole support) carry no mass;
                # fall back to sampling p, as the scalar path did.
                resid = torch.where(
                    resid.sum(dim=1, keepdim=True) > 1e-12, resid, p_row
                )
            else:
                resid = p_row
            bonus_gpu = torch.multinomial(resid, 1, generator=gen).squeeze(1)
            if sparse:
                # Map the drawn column back to a token id.
                bonus_gpu = p_row_idx.gather(1, bonus_gpu.unsqueeze(1)).squeeze(1)
            # ONE D2H: [n_accepted | bonus | drafts...] -- same packing as the
            # greedy accept, so the draft chain's tokens also arrive here.
            rows = [na_gpu.to(torch.int64), bonus_gpu.to(torch.int64)]
            if d_gpu is not None:
                rows.extend(d_gpu.t())
            packed_cpu = torch.stack(rows).cpu()  # [2+kk, nd]
            na_cpu = packed_cpu[0].tolist()
            bonus_cpu2 = packed_cpu[1].tolist()
            drafts_cpu = packed_cpu[2:].t().tolist() if d_gpu is not None else None
            # Batch-gather the per-seq draft-seed hidden (verify row
            # ``i*qlen + na``) in one op instead of nd tiny clones.
            bonus_hidden_all = v_hidden.index_select(0, seq_ar * qlen + na_l)
            for i, s in enumerate(decode_seqs):
                na = na_cpu[i]
                committed = [x1[i]] + (drafts_cpu[i][:na] if na else [])
                # Relay the bonus as next step's x1; do NOT commit it now.
                new_relay[s.seq_id] = (bonus_cpu2[i], bonus_hidden_all[i])
                n_accepted[i] = na
                results[i] = committed
            # The accept decisions used per-rank distributions (p/q differ by fp
            # all-reduce epsilon across TP ranks) + per-rank RNG draws, so
            # ``results`` / ``n_accepted`` / the relayed bonus can diverge. Make
            # TP-rank-0's decisions authoritative: broadcast a padded token grid
            # (committed lists) + the relayed bonus token, then every rank rebuilds
            # identical state. (Draft tokens were already broadcast; the accept +
            # bonus draws happen here.)
            if get_tp_size() > 1:
                # ``results`` holds [x1 + accepted_drafts]; the bonus is relay-only.
                maxlen = kk + 1
                grid = torch.full((nd, maxlen), -1, dtype=torch.int64, device=dev)
                lens = torch.zeros(nd, dtype=torch.int64, device=dev)
                bonus_t = torch.zeros(nd, dtype=torch.int64, device=dev)
                if get_tp_rank() == 0:
                    for i in range(nd):
                        c = results[i]
                        lens[i] = len(c)
                        grid[i, : len(c)] = torch.tensor(
                            c, dtype=torch.int64, device=dev
                        )
                        bonus_t[i] = new_relay[decode_seqs[i].seq_id][0]
                src = get_rank() - get_tp_rank()
                dist.broadcast(lens, src=src, group=get_ipc_tp_group())
                dist.broadcast(grid, src=src, group=get_ipc_tp_group())
                dist.broadcast(bonus_t, src=src, group=get_ipc_tp_group())
                lens_cpu = lens.cpu().tolist()
                grid_cpu = grid.cpu().tolist()
                bonus_cpu = bonus_t.cpu().tolist()
                for i in range(nd):
                    n = lens_cpu[i]
                    results[i] = grid_cpu[i][:n]
                    n_accepted[i] = max(0, n - 1)
                    # Adopt rank-0's bonus token; keep this rank's own hidden
                    # (only seeds the next draft, whose token is broadcast).
                    _, h = new_relay[decode_seqs[i].seq_id]
                    new_relay[decode_seqs[i].seq_id] = (bonus_cpu[i], h)
                # The persistent penalty history must adopt rank 0's accepted
                # prefix, just like the CPU sequence/relay state above.
                na_gpu = lens - 1
        else:
            # --- 3. Greedy accept per seq (vectorized on GPU). ---
            # Verify inputs per seq are [x1, d1..dk] at positions start..start+k.
            # v_pred[start+p] = target's greedy token AFTER consuming input p, so
            # accept d_{p+1} iff it equals v_pred[start+p]; the accepted count is
            # the length of the leading all-match prefix. ``v_pred`` is a GPU
            # tensor ``[nd*qlen]`` (qlen == 1+kk, uniform); compute ``n_accepted``
            # on-device and D2H only the tiny ``[nd]`` results (+ ``[nd]`` fused
            # bonus tokens), instead of the old blocking ``[nd*qlen]`` v_pred D2H.
            # ``results`` is rebuilt on the host from the already-known ``x1`` /
            # ``drafts`` lists sliced to ``n_accepted`` -- no full v_pred needed.
            qlen = 1 + kk
            # Greedy target: one argmax over the verify logits (the rejection
            # branch instead transforms the same logits into ``p``).
            vp = v_logits[: nd * qlen].argmax(dim=-1).view(nd, qlen)
            seq_ar = self._mtp_staging.row_idx[:nd]
            if kk > 0:
                # Prefer the GPU draft tensor threaded from the draft chain (the
                # graph chain never materializes the drafts on the host at all).
                # Fall back to a one-shot H2D only when a host-side chain ran.
                dg = getattr(self, "_drafts_gpu", None)
                if dg is not None and tuple(dg.shape) == (nd, kk):
                    drafts_gpu = dg.to(vp.dtype)
                else:
                    drafts_gpu = torch.tensor(drafts, device=dev, dtype=vp.dtype)
                match = vp[:, :kk] == drafts_gpu  # [nd,kk] bool
                # cumprod over the bool prefix: 1 until the first mismatch, 0 after
                # -> sum = length of the leading all-accept prefix.
                na_gpu = torch.cumprod(match.to(torch.int32), dim=1).sum(dim=1)  # [nd]
            else:
                drafts_gpu = None
                na_gpu = torch.zeros(nd, device=dev, dtype=torch.int32)
            na_l = na_gpu.to(torch.long)
            # Batch-gather every seq's bonus hidden in ONE op (verify row
            # ``i*qlen + na``) instead of nd separate per-seq ``.clone()``s.
            bonus_rows = seq_ar * qlen + na_l
            bonus_hidden_all = v_hidden.index_select(0, bonus_rows)  # [nd, H]
            if _async_accept:
                if self._mtp_penalties is not None:
                    self._mtp_penalties.commit(penalty_candidates, na_gpu)
                state = self._mtp_async_state
                seq_ids = tuple(s.seq_id for s in decode_seqs)
                if state is None:
                    state = MtpAsyncBatchState(
                        max_batch_size=self.max_running_seqs,
                        k=kk,
                        hidden_size=bonus_hidden_all.shape[-1],
                        hidden_dtype=bonus_hidden_all.dtype,
                        device=dev,
                    )
                    self._mtp_async_state = state
                if not state.matches(seq_ids):
                    if any(state._busy):
                        raise RuntimeError(
                            "MTP async cohort changed before pending completion was collected"
                        )
                    self._mtp_staging.install_ctx_host_np[:nd] = [
                        len(t) for t in orig_tokens
                    ]
                    state.install(
                        seq_ids,
                        self._mtp_staging.install_ctx_host[:nd],
                        x1_gpu,
                        hidden,
                    )
                with torch.profiler.record_function("gllm::mtp_accept_publish"):
                    completion = state.publish(
                        current_x1=x1_gpu,
                        drafts=(
                            drafts_gpu
                            if drafts_gpu is not None
                            else torch.empty((nd, 0), dtype=torch.int64, device=dev)
                        ),
                        num_accepted_drafts=na_gpu,
                        next_bonus=vp[seq_ar, na_l],
                        next_hidden=bonus_hidden_all,
                        producer_stream=torch.cuda.current_stream(),
                        extra_tokens=prefill_tokens_gpu,
                        extra_seq_ids=tuple(s.seq_id for s in extra_prefill_seqs),
                        new_state_seq_ids=new_state_seq_ids,
                        new_state_context_lens=new_state_context_lens,
                        new_state_tokens=new_state_tokens,
                        new_state_hidden=new_state_hidden,
                    )
                # All speculative GenerationSequence mutations are host bookkeeping only;
                # restore them now.  The completion event orders the later CPU
                # finalize after verify/accept and the D2H record.
                if structured_active:
                    self.sampler._structured.record_speculative(structured_active, completion)
                restore()
                return completion

            # Synchronous fallback: ONE D2H per step carries everything the
            # host still needs.  This must stay after the async early return;
            # doing it before that branch silently serialized overlap MTP.
            #   row 0        : n_accepted
            #   row 1        : the relayed bonus token
            #   rows 2..2+kk : the draft token grid (transposed)
            rows = [na_gpu.to(torch.int64)]
            rows.append(vp[seq_ar, na_gpu.to(torch.long)].to(torch.int64))
            if drafts_gpu is not None:
                rows.extend(drafts_gpu.to(torch.int64).t())
            packed_cpu = torch.stack(rows).cpu()  # [2+kk, nd]
            na_cpu = packed_cpu[0].tolist()
            bonus_cpu2 = packed_cpu[1].tolist()
            drafts_cpu = packed_cpu[2:].t().tolist() if drafts_gpu is not None else None
            for i, s in enumerate(decode_seqs):
                na = na_cpu[i]
                n_accepted[i] = na
                # committed = x1 + the accepted draft prefix.
                results[i] = [x1[i]] + (drafts_cpu[i][:na] if na else [])
                # Bonus token from the on-device gather; bonus hidden is row i of
                # the batched gather and only seeds the next draft.
                new_relay[s.seq_id] = (bonus_cpu2[i], bonus_hidden_all[i])

        if self._mtp_penalties is not None:
            self._mtp_penalties.commit(penalty_candidates, na_gpu)
        self._record_mtp_metrics(nd, kk, n_accepted)
        if structured_active:
            self.sampler._structured.commit_speculative(
                structured_active, results,
                [new_relay[s.seq_id][0] for s in decode_seqs],
            )

        # Update only rows that actually ran. A seq absent from this MTP batch
        # has not advanced, so its relay remains position-correct and must be
        # retained. Relay invalidation belongs to the paths that really advance
        # a seq without MTP (``_mtp_drop_relay``) and to request cleanup. The
        # old wholesale replacement turned harmless scheduling gaps into relay
        # misses; the bootstrap then consumed an already-verified token again
        # and double-advanced hybrid GDN state.
        self._mtp_relay.update(new_relay)

        # --- Hybrid GDN recurrent-state: commit the accepted column to col 0 ---
        # The verify forward wrote token t's post-state into block-table column
        # t. Each seq committed ``1+na`` tokens, so the post-acceptance state is
        # already sitting in the block at column ``na``. Rather than COPY that
        # block's 18.6MB (all 18 layers) into column 0 -- pure memory-bandwidth
        # cost, ~2.2ms at nd=64 -- we just SWAP the two block-table entries: the
        # physical block holding the committed state becomes column 0, and the
        # old column-0 block moves to column ``na`` where it's overwritten as
        # scratch by the next verify. O(1) per seq, zero data movement. The next
        # step rebuilds ``ssm_block_table_2d`` / ``recurrent_state_slot`` from the
        # (now-permuted) list, so decode/snapshot read the committed state from
        # column 0 as before. na==0 -> committed state already at column 0.
        if _has_gdn:
            for i in range(nd):
                na = n_accepted[i]
                if na > 0:
                    bt = decode_seqs[i].ssm_block_table
                    bt[0], bt[na] = bt[na], bt[0]
                    # Keep the scalar slot mirror consistent with column 0 (read
                    # by the plain decode path + prefix-cache snapshot capture).
                    decode_seqs[i].recurrent_state_slot = bt[0]
                # Reset the persisted resume column: committed state is now at
                # column 0, so next step's num_accepted is neutral (1).
                decode_seqs[i].ssm_num_accepted = 1

        restore()
        return results + prefill_tokens

    def step_once_mtp_async(self) -> MtpAsyncCompletion:
        """Enqueue one greedy fused-MTP step without waiting for its D2H result.

        Sampling/rejection MTP intentionally falls back to the synchronous path:
        its TP-authoritative accept broadcast currently materializes CPU lists.
        """
        seqs = self.input_data.seqs[: self.input_data.num_decodes]
        if any(
            (s.temperature > 1e-5 and abs(s.temperature - 1.0) > 1e-5) or s.top_k != 1
            for s in seqs
        ):
            raise RuntimeError("async MTP currently requires greedy requests")
        self._mtp_async_publish = True
        try:
            completion = self.step_once()
        finally:
            self._mtp_async_publish = False
        if not isinstance(completion, MtpAsyncCompletion):
            raise RuntimeError("fused MTP did not produce an async completion")
        return completion

    @torch.inference_mode()
    def step_once_mtp_mixed(
        self, decode_seqs, prefill_seqs, *, async_publish: bool = False
    ):
        """Run one mixed verify+prefill step with exactly one input prepare.

        The overlap worker calls this directly from its prefetched sequence
        list.  Going through ordinary ``prepare_input`` first would build and
        release Qwen-VL-compatible prompt embeddings, only for ``_mtp_decode``
        to request the same prefill rows again.
        """
        decode_seqs = list(decode_seqs)
        prefill_seqs = list(prefill_seqs)
        if not decode_seqs or not prefill_seqs:
            raise ValueError("mixed MTP requires both decode and prefill rows")
        if not all(s.computed_prompt for s in decode_seqs):
            raise ValueError("mixed MTP decode prefix contains a prefill row")
        async_state = self._mtp_async_state
        if (
            async_publish
            and async_state is not None
            and async_state.matches([s.seq_id for s in decode_seqs])
        ):
            # The predecessor has published its acceptance/relay entirely on
            # the forward stream.  Consume those tensors directly; collecting
            # its host completion before this launch would break the pipeline.
            x1 = async_state.relay_tokens[: len(decode_seqs)]
            hidden = async_state.relay_hidden[: len(decode_seqs)]
            self.prepare_input_mtp(decode_seqs)
        elif all(s.seq_id in self._mtp_relay for s in decode_seqs):
            relay = [self._mtp_relay[s.seq_id] for s in decode_seqs]
            x1 = [r[0] for r in relay]
            hidden = torch.stack([r[1] for r in relay], dim=0)
            self.prepare_input_mtp(decode_seqs)
        else:
            # Bootstrap finishes by restoring the fused decode bookkeeping.
            hidden, x1 = self._mtp_bootstrap_padded_verify(decode_seqs)
        self._last_logprobs = None
        old_publish = self._mtp_async_publish
        self._mtp_async_publish = async_publish
        try:
            result = self._mtp_decode(
                hidden, x1, extra_prefill_seqs=prefill_seqs
            )
        finally:
            self._mtp_async_publish = old_publish
        if async_publish and not isinstance(result, MtpAsyncCompletion):
            raise RuntimeError("mixed MTP did not produce an async completion")
        return result

    def finalize_mtp_async(
        self,
        completion: MtpAsyncCompletion,
        seqs,
        *,
        materialize_state=True,
        materialize_seqs=None,
    ):
        """Collect one result; publish CPU relay/GDN mirrors only at a drain.

        Stable overlap cohorts collect older user-visible outputs after their
        successor has already launched.  Mutating the CPU block table for such
        an intermediate completion would make it describe an obsolete GPU
        checkpoint.  The latest completion is materialized only when the MTP
        pipeline drains or changes cohort.
        """
        if self.sampler._structured is not None:
            self.sampler._structured.finish_speculative(seqs, completion)
        valid, committed = completion.collect()
        if tuple(s.seq_id for s in seqs) != completion.seq_ids:
            raise RuntimeError("MTP async completion no longer matches its cohort")
        state = completion.owner
        if materialize_state:
            # One batched copy is paid only at a real drain boundary, not once
            # per speculative iteration.
            state_n = state.batch_size
            relay_hidden = state.relay_hidden[:state_n].clone()
            relay_tokens = state.relay_tokens[:state_n].cpu().tolist()
            resume = state.resume_num_accepted[:state_n].cpu().tolist()
            candidates = list(materialize_seqs or seqs)
            by_id = {s.seq_id: s for s in candidates}
            missing = [seq_id for seq_id in state.seq_ids if seq_id not in by_id]
            if missing:
                raise RuntimeError(
                    f"cannot materialize MTP async state; missing seq ids {missing}"
                )
            for i, seq_id in enumerate(state.seq_ids):
                seq = by_id[seq_id]
                if getattr(seq, "_overlap_freed", False):
                    continue
                self._mtp_relay[seq.seq_id] = (int(relay_tokens[i]), relay_hidden[i])
                if self.sampler._structured is not None:
                    self.sampler._structured.stage_relay(seq, int(relay_tokens[i]))
                na = int(resume[i]) - 1
                if seq.ssm_block_table is not None:
                    if na > 0:
                        bt = seq.ssm_block_table
                        bt[0], bt[na] = bt[na], bt[0]
                        seq.recurrent_state_slot = bt[0]
                    seq.ssm_num_accepted = 1
            # The GPU state used the pre-materialization block-table ordering.
            # Invalidate it so a later async run starts from the CPU column-0
            # view installed above instead of mixing the two conventions.
            state.reset()
        self._record_mtp_metrics(len(seqs), self._mtp_k, [v - 1 for v in valid])
        return committed


def __getattr__(name: str):
    # ``EmbeddingInfo`` is defined in ``gllm.multimodal.mixin`` (it is shared
    # with the multimodal embedding cache there). Resolve it lazily so this
    # module does not import the mixin at module level, keeping the import
    # graph one-directional.
    if name == "EmbeddingInfo":
        from gllm.multimodal.mixin import EmbeddingInfo

        return EmbeddingInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
