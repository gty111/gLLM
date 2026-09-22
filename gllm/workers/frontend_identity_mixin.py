"""Worker-side handling of frontend-facing request identity.

A standalone frontend is stateless and restarts freely; its id pool
re-issues 0, 1, 2, ... on every incarnation, while a SURVIVING fleet
may still be running the dead frontend's request 0. Two live
(session, id) pairs can therefore carry the SAME client seq_id, and a
bare id cannot name a unique request anywhere inside the worker
(scheduling, aborts, FollowerSeq mirrors, KV, output stamping).

Remapping at admission fixes this structurally: each admitted seq gets
a FLEET-UNIQUE monotonic internal id (``seq_id``); the dispatching
frontend's original id is preserved as ``client_seq_id`` next to the
session stamp, and OUTPUT packages translate internal ids back to
client ids with per-row session stamps before hitting the wire. The
frontend (running_maps / async_streams) keeps seeing only client ids.

The host :class:`~gllm.workers.worker.Worker` mixes this in (see
``class Worker(FrontendMixin, TorchProfilerMixin)``). Everything here
is a no-op on the monolith path: the guard
:meth:`_standalone_remap_enabled` keys off the deployment mode, and
each method early-returns when the worker does not sit behind a
standalone frontend."""
from gllm.runtime.id_allocator import INTERNAL_ID_START


class FrontendMixin:
    """Translate between frontend client ids and fleet-internal ids.

    State consulted (host-provided):

    * ``self.standalone_worker`` -- deployment-mode flag (see
      :meth:`_standalone_remap_enabled`);
    * ``self.comm`` -- a `zmqComm`; the identity registry
      (``_session_identity``, internal id -> (client id, session)) and
      the deferred-reclaim set (``_identity_reclaim``) live on it so
      the output hook can share them across the wire layer;
    * ``self.scheduler`` -- live-request source (seqs_to_prefill /
      seqs_to_decode / batch_running) for abort routing and fallback
      identity lookups.

    Host touchpoints (all one-liners, no-ops on the monolith path):

    * admission: ``_remap_client_ids`` on every admit path;
    * aborts: ``_route_frontend_aborts`` before the scheduler
      sees them;
    * output: ``comm._output_committer =
      self.translate_output_for_frontend`` installed once in
      ``Worker.init``.
    """

    def _standalone_remap_enabled(self) -> bool:
        """True iff this worker sits behind a STANDALONE frontend.

        A monolith worker also has ``comm.frontend == False`` (it talks
        to its in-process frontend), so the remap must key off the
        deployment mode, not the socket role: monolith client ids are
        already unique per fleet (one frontend), remapping there is
        pointless churn. ``standalone_worker`` is set on the worker
        parent and inherited by its spawned child.
        """
        return bool(getattr(self, "comm", None)) \
            and bool(getattr(self, "standalone_worker", False))

    def _remap_client_ids(self, seqs):
        """Assign fleet-unique internal ids to freshly admitted seqs.

        Ids start past the allocator's client range (0..99999) so an
        internal id can never collide with a bare client id on the wire.
        Every PP-0 TP rank runs this with the IDENTICAL input order (the
        aggregated package is broadcast pre-admission), so all columns mint
        the same internal ids in lockstep; monolith workers skip it.
        """
        if not self._standalone_remap_enabled():
            return
        counter = getattr(self, "_next_internal_seq", None)
        if counter is None:
            counter = self._next_internal_seq = INTERNAL_ID_START
        # Register the identity AT ADMISSION: the scheduler may free a seq
        # (first token == EOS / max_tokens=1) before it ever produces a
        # non-terminal output row, so identity recorded only at translate
        # time would be lost for exactly those requests. Retention is
        # bounded by the output drain below.
        reg = self.comm._session_identity
        for seq in seqs or ():
            seq.client_seq_id = seq.seq_id
            seq.seq_id = counter
            reg[seq.seq_id] = (seq.client_seq_id,
                               getattr(seq, "frontend_session", None))
            counter += 1
        self._next_internal_seq = counter

    def _route_frontend_aborts(self, abort_ids, abort_sessions):
        """Translate frontend aborts (CLIENT ids + session stamps) to the
        INTERNAL ids of the requests those stamps actually own.

        A stamped abort (standalone frontend) resolves only to live seqs
        whose (client id, session) pair matches exactly -- a bare id
        naming a request of ANOTHER session is ignored. An unstamped
        abort (monolith; the standalone frontend always stamps) resolves
        to every live holder of that id, preserving monolith semantics.
        Aborts that match no live seq are dropped: the request already
        finished (or the stamp is unknown) and nothing is left to free.
        """
        ids = list(abort_ids or ())
        if not ids or not self._standalone_remap_enabled():
            return ids
        stamps = list(abort_sessions) if abort_sessions is not None \
            else [None] * len(ids)
        try:
            batches = [
                list(self.scheduler.seqs_to_prefill),
                list(self.scheduler.seqs_to_decode),
            ]
            batches.extend(self.scheduler.batch_running)
        except Exception:
            return ids
        # Live (client_id, session) holders of any requested client id.
        live = {}
        for batch in batches:
            for seq in batch:
                cid = getattr(seq, "client_seq_id", None)
                if cid is not None and cid in ids:
                    live.setdefault(cid, []).append(
                        (getattr(seq, "frontend_session", None),
                         seq.seq_id))
        routed = []
        for cid, st in zip(ids, stamps):
            holders = live.get(cid) or []
            if st is None:
                # Unstamped (monolith): every holder of that id.
                routed.extend(internal for _, internal in holders)
            else:
                # Stamped: only the request of THIS session.
                routed.extend(internal for held, internal in holders
                              if held == st)
        return routed

    def _output_row_identity(self, seq_id):
        """(client_id, session) for one OUTPUT row's internal id, or None.

        The admission-time registry answers first: it covers rows of
        requests the scheduler already freed (first token == EOS /
        max_tokens=1, aborted pre-first-output -- the seq is gone from
        every queue by the time its terminal row is translated) and rows
        of overlap/MTP requests deferred across ticks. A live seq is
        consulted when the registry missed (defensive; admission
        registers every remapped seq). Unknown ids degrade to None and
        are dropped by the frontend (fail-closed).
        """
        reg = getattr(self.comm, "_session_identity", None)
        if reg is not None and seq_id in reg:
            return reg[seq_id]
        try:
            batches = [
                list(self.scheduler.seqs_to_prefill),
                list(self.scheduler.seqs_to_decode),
            ]
            batches.extend(self.scheduler.batch_running)
            for batch in batches:
                for seq in batch:
                    if seq.seq_id == seq_id:
                        return getattr(seq, "client_seq_id", seq_id), \
                            getattr(seq, "frontend_session", None)
        except Exception:
            pass
        return None

    def translate_output_for_frontend(self, ipc_package) -> None:
        """In-place: internal ids -> client ids, with per-row session stamps.

        Invoked from zmqComm.send_output (installed via
        ``comm._output_committer``) so EVERY frontend-facing emission --
        plain process_output, check_abort_seqs replies, overlap finalize,
        MTP paths -- is translated. Monolith packages (no stamps minted)
        pass through byte-identical: internal id equals client id and the
        stamp lists come back all-None, which the monolith frontend never
        consults.
        """
        if not self._standalone_remap_enabled():
            return
        # Identity retention: an admission-time mapping is reclaimable only
        # AFTER the request's TERMINAL rows have been translated -- free
        # rows (abort / capacity-error replies) and the final act row of a
        # finishing seq (which the scheduler ships together with its free
        # row in the same package). Ordinary non-terminal act rows -- one
        # PER DECODE STEP for a max_tokens=N request -- must NOT trigger
        # reclamation: the request keeps producing rows (and the scheduler
        # drops it from every live queue at its final step), so its
        # mapping must survive until the terminal rows go out. Deferral
        # bounds retention: reclaimed on the NEXT drain after the terminal
        # rows, i.e. at most one drain's worth of finished requests.
        reg = self.comm._session_identity
        pending = self.comm._identity_reclaim
        for stale in pending:
            reg.pop(stale, None)
        pending.clear()

        # Act rows of ids freed in THIS package are the finishing step
        # (token + free ship together); they join the free rows as
        # reclaimable.
        terminal = set(ipc_package.free_ids)

        def _row(internal_id, reclaim):
            ident = self._output_row_identity(internal_id)
            if ident is not None:
                if reclaim:
                    self.comm._identity_reclaim.add(internal_id)
                return ident
            return None

        act = list(ipc_package.act_schedule_ids)
        free = list(ipc_package.free_ids)
        if act:
            rows = [_row(i, i in terminal) for i in act]
            ipc_package.act_schedule_ids = [
                r[0] if r is not None else i for r, i in zip(rows, act)]
            ipc_package.sessions = [r[1] if r is not None else None
                                    for r in rows]
        if free:
            rows = [_row(i, True) for i in free]
            ipc_package.free_ids = [
                r[0] if r is not None else i for r, i in zip(rows, free)]
            ipc_package.free_sessions = [r[1] if r is not None else None
                                         for r in rows]
        errors = getattr(ipc_package, "request_errors", None)
        if errors:
            new_errors = {}
            for internal_id, err in errors.items():
                # Terminal capacity errors free the request this tick.
                r = _row(internal_id, True)
                new_errors[r[0] if r is not None else internal_id] = err
            ipc_package.request_errors = new_errors
        plc = getattr(ipc_package, "prompt_logprobs", None)
        if plc:
            new_plc = {}
            for internal_id, val in plc.items():
                # Prefill-side sidecar (seq still running unless freed in
                # this same package): reclaim only for terminal ids.
                r = _row(internal_id, internal_id in terminal)
                new_plc[r[0] if r is not None else internal_id] = val
            ipc_package.prompt_logprobs = new_plc
