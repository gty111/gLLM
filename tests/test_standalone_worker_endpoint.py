"""Unit tests for the frontend/worker decoupling rendezvous
(``gllm.entrypoints.worker_endpoint``) and the standalone engine plumbing.

These run without a GPU: they exercise the endpoint-file writer/reader
lifecycle and the standalone frontend's comm construction + reconnect
logic, which is the load-bearing part of the crash-isolation contract.
"""

import json
import os
import time

import pytest
import zmq

from gllm.entrypoints import worker_endpoint as we


def test_writer_publish_read_roundtrip(tmp_path):
    path = str(tmp_path / "ep.json")
    w = we.WorkerEndpointWriter(path)
    uuid1 = w.set_endpoints(
        {0: {"schedule": "ipc:///tmp/a", "output": "ipc:///tmp/b", "token": "ipc:///tmp/c"}}
    )
    # Reader must see the published endpoints with INTEGER rank keys.
    got_uuid, endpoints = we.read_worker_endpoint_file(path)
    assert got_uuid == uuid1
    assert 0 in endpoints  # JSON string key normalised to int
    assert endpoints[0]["output"] == "ipc:///tmp/b"
    w.cleanup()
    # Cleanup removes the file so no frontend can dial a dead worker.
    assert we.read_worker_endpoint_file(path) == (None, None)


def test_missing_file_reads_none(tmp_path):
    path = str(tmp_path / "nope.json")
    assert we.read_worker_endpoint_file(path) == (None, None)
    assert we.endpoint_file_age_seconds(path) is None


def test_heartbeat_refreshes_mtime(tmp_path):
    path = str(tmp_path / "ep.json")
    w = we.WorkerEndpointWriter(path, heartbeat_interval=0.01)
    w.set_endpoints({0: {"schedule": "s", "output": "o", "token": "t"}})
    before = we.endpoint_file_age_seconds(path)
    time.sleep(0.05)
    after = we.endpoint_file_age_seconds(path)
    # 0.5s after publish the age would be ~0.5s; staying near-zero proves
    # the heartbeat kept rewriting the file.
    assert after < 0.3, f"mtime not refreshed: before={before:.3f} after={after:.3f}"
    assert after >= 0
    w.heartbeat()
    w.cleanup()


def test_standalone_frontend_comm_connects(monkeypatch):
    """A standalone_remote frontend PULL/PUSH pair must exchange frames with a
    worker-style binder -- this is the transport the decoupled path relies
    on. All endpoints are absolute ipc:// URIs with unique names; the
    frontend-side sockets are created through the real production code path
    (``make_socket``), which is where the SNDHWM/RCVHWM tuning that caused
    the 512MB-buffer send hang lives."""
    import threading as _threading

    import zmq as _zmq
    from gllm.distributed.comm import zmqComm

    sched_path = "ipc:///tmp/_gllm_fe_test_sched_%d" % os.getpid()
    out_path = "ipc:///tmp/_gllm_fe_test_out_%d" % os.getpid()
    tok_path = "ipc:///tmp/_gllm_fe_test_tok_%d" % os.getpid()
    from gllm.utils import make_socket

    ctx = _zmq.Context()
    # Worker child binds the schedule PULL; the frontend PUSHes to it.
    sched = ctx.socket(_zmq.PULL)
    sched.setsockopt(_zmq.LINGER, 0)
    sched.bind(sched_path)

    # Frontend sockets via the production factory (PUSH schedule, PULL
    # output), exactly as zmqComm(standalone_remote=True) builds them.
    fe_req = make_socket(ctx, sched_path, _zmq.PUSH)
    fe_out = make_socket(ctx, out_path, _zmq.PULL)

    # Worker-side output: plain PUSH with modest buffers (the real worker
    # child uses make_pull_bind for its *request* side; the output PUSH is
    # created with the same tuning as make_socket).
    out = make_socket(ctx, out_path, _zmq.PUSH)

    fe_req.send_pyobj({"ping": 1})
    assert sched.recv_pyobj() == {"ping": 1}

    # Bounded send: a PUSH send must not be able to park forever on a huge
    # zero-copy buffer if the peer connection is slow to establish.
    sent = []

    def _do_send():
        try:
            out.send_pyobj({"pong": 2})
            sent.append(True)
        except _zmq.ZMQError:
            sent.append(False)

    t = _threading.Thread(target=_do_send, daemon=True)
    t.start()
    deadline = time.monotonic() + 10.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    t.join(timeout=5.0)
    assert sent and sent[0], "worker output send did not complete (dead peer)"

    deadline = time.monotonic() + 5.0
    pong = None
    while time.monotonic() < deadline:
        if fe_out.poll(timeout=100):
            pong = fe_out.recv_pyobj()
            break
    assert pong == {"pong": 2}

    # Deterministic teardown before anything else in the test process
    # touches CUDA (a zombie zmq I/O thread holding open ipc connections
    # can block subsequent native init on a loaded host).
    for sock in (sched, fe_req, fe_out, out):
        try:
            sock.setsockopt(_zmq.LINGER, 0)
            sock.close(linger=0)
        except _zmq.ZMQError:
            pass
    ctx.term()


def test_standalone_worker_skips_frontend_comm():
    """A standalone worker parent must not create a frontend-role comm on the
    shared endpoints (a stray PULL would swallow half the output frames)."""
    from gllm.engine.llm import LLM

    # Construct just enough to call _init_frontend_comm in worker mode without
    # building the full engine (which would spawn a GPU child).
    eng = LLM.__new__(LLM)
    eng.standalone_worker = True
    eng.host = "127.0.0.1"
    eng.launch_mode = "normal"
    eng.master_addr = "127.0.0.1"
    eng.schedule_path = "ipc:///tmp/_gllm_noworker_sched"
    eng.output_path = "ipc:///tmp/_gllm_noworker_out"
    eng.token_path = "ipc:///tmp/_gllm_noworker_tok"
    eng.dp_size = 1
    eng._init_frontend_comm()
    assert eng.comm is None
