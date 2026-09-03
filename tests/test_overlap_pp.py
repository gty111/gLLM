from types import SimpleNamespace

import torch

import gllm.distributed.parallel_state as parallel_state
import gllm.workers.overlap as overlap_worker
from gllm.scheduling.distributed import FollowerSeq, SeqRegister
from gllm.scheduling.scheduler import OverlapScheduler
from gllm.runtime.async_runtime import FutureMap


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _set_pp_rank(monkeypatch, *, rank, pp_rank, pp_size=3, tp_size=2):
    monkeypatch.setattr(parallel_state, "_RANK", rank)
    monkeypatch.setattr(parallel_state, "_PP_RANK", pp_rank)
    monkeypatch.setattr(parallel_state, "_PP_SIZE", pp_size)
    monkeypatch.setattr(parallel_state, "_TP_SIZE", tp_size)
    monkeypatch.setattr(parallel_state, "_DP_SIZE", 1)


def test_pp_token_feedback_uses_same_column_and_waits(monkeypatch):
    _set_pp_rank(monkeypatch, rank=5, pp_rank=2)
    broadcasts = []

    def fake_broadcast(tensor, src, group, async_op):
        work = _Work()
        broadcasts.append((tensor, src, group, async_op, work))
        return work

    monkeypatch.setattr(parallel_state.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(parallel_state, "_PP_GROUP", "pp-column")
    tokens = torch.tensor([3, 7])
    parallel_state.send_pp_tokens_to_previous_stages(tokens)

    assert len(broadcasts) == 1
    tensor, src, group, async_op, work = broadcasts[0]
    assert tensor is tokens
    assert src == 5
    assert group == "pp-column"
    assert async_op
    assert work.waited


def test_pp_token_feedback_recv_comes_from_last_stage(monkeypatch):
    _set_pp_rank(monkeypatch, rank=3, pp_rank=1)
    received = []

    def fake_broadcast(tensor, src, group, async_op):
        work = _Work()
        received.append((tensor, src, group, async_op, work))
        return work

    monkeypatch.setattr(parallel_state.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(parallel_state, "_PP_GROUP", "pp-column")
    target = torch.empty(2, dtype=torch.long)
    parallel_state.recv_pp_tokens_from_last_stage(target)

    assert received[0][0] is target
    assert received[0][1] == 5
    assert received[0][2] == "pp-column"
    assert received[0][3]
    assert received[0][4].waited


def test_pp_follower_has_local_repetition_penalty_state():
    seq = FollowerSeq(
        SeqRegister(
            seq_id=1,
            prompt_token_ids=[1, 2],
            prompt_len=2,
            finish_tokens=[0],
            ignore_eos=False,
            output_len=8,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.1,
        )
    )
    assert seq.rep_slot is None
    assert seq.rep_filled == 0


def test_overlap_mtp_is_not_selected_across_pp(monkeypatch):
    monkeypatch.setattr(overlap_worker, "get_pp_size", lambda: 2)
    worker = overlap_worker.OverlapWorker.__new__(overlap_worker.OverlapWorker)
    worker._dp = False
    worker.model_runner = SimpleNamespace(mtp_enabled=True)
    input_data = SimpleNamespace(seqs=[SimpleNamespace(computed_prompt=True)])
    assert worker._mtp_decode_prefix(input_data) == []


class _DeferredSeq:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.computed_token_num = 4
        self.to_compute_token_num = 1
        self.prompt_len = 4
        self.token_ids = [1, 2, 3, 4, 5]
        self._overlap_freed = False

    @property
    def computed_prompt(self):
        return self.computed_token_num >= self.prompt_len

    def append(self, token):
        self.token_ids.append(token)


def test_pp_overlap_rotates_completed_microbatch_to_decode_tail():
    first = _DeferredSeq(1)
    waiting = _DeferredSeq(2)
    scheduler = OverlapScheduler.__new__(OverlapScheduler)
    scheduler.pp_size = 2
    scheduler.batch_running = overlap_worker.deque([[first]])
    scheduler.seqs_to_decode = overlap_worker.deque([waiting])

    deferred = scheduler.process_output_deferred([7])

    assert deferred == [(0, first, 5)]
    assert [seq.seq_id for seq in scheduler.seqs_to_decode] == [2, 1]
    assert first.token_ids[-1] == -7


def test_future_map_fast_path_detects_only_negative_placeholders():
    assert not FutureMap.has_futures(torch.tensor([0, 1, 7], dtype=torch.long))
    assert FutureMap.has_futures(torch.tensor([0, -3, 7], dtype=torch.long))


def test_retire_helper_returns_batches_in_launch_order(monkeypatch):
    """The driver pops ``_ready_q`` assuming FIFO; assert the helper keeps it.

    Also guards the wiring itself: ``init`` starts ``_retire_loop`` in a
    thread, so a missing or renamed method only surfaces at serving time --
    a cleanup pass once deleted both this method and ``_settle`` with every
    unit test still green.
    """
    import queue as _queue
    import threading

    from gllm.workers import overlap as overlap_worker

    worker = overlap_worker.OverlapWorker.__new__(overlap_worker.OverlapWorker)
    worker._wait_q = _queue.Queue()
    worker._ready_q = _queue.Queue()
    # ``tokens`` already set makes ``_settle`` a no-op, so no GPU is needed.
    # The helper still arms its CUDA context on the first batch, which this
    # CPU-only test stubs out.
    import torch

    monkeypatch.setattr(torch.cuda, "set_device", lambda _dev: None)
    worker.model_runner = SimpleNamespace(
        forward_stream=SimpleNamespace(device="cpu")
    )
    sent = [
        overlap_worker._PendingBatch(
            copy_done=None,
            batch_size=1,
            buf_idx=i,
            future_slot_ids=[i],
            deferred=None,
            input_data=None,
            is_dummy=True,
            lp_k=None,
            tokens=[i],
        )
        for i in range(5)
    ]
    t = threading.Thread(target=worker._retire_loop, daemon=True)
    t.start()
    for b in sent:
        worker._wait_q.put(b)
    got = [worker._ready_q.get(timeout=10) for _ in sent]
    worker._wait_q.put(None)
    t.join(timeout=5)
    assert [b.buf_idx for b in got] == [b.buf_idx for b in sent]
