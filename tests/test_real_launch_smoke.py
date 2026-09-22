"""Real-launch smoke tests for the decoupled deployment.

CPU-only unit tests cannot see launch-path breakage (e.g. a synchronous
watchdog blocking ``worker.init()`` forever, or a NameError in the
worker entrypoint), so these spin up REAL processes on the smallest
cached model (Qwen3-0.6B). They are marked ``requires_gpu`` and SKIPPED
when no model is available, so CI without GPU access stays green.

Run explicitly:
    pytest tests/test_real_launch_smoke.py -m requires_gpu -v
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time

import pytest

# The smoke subprocess runs with a PYTHON that matches the box's GPU
# DRIVER: the default interpreter of this checkout (torch cu130) fails
# cuda init on a 12.x driver even though the production service runs
# fine (it is a cu124 build). Override with GLLM_SMOKE_PYTHON.
# Default: the cu130 env of this checkout (its torch needs the
# cuda-compat lib, applied via _compat_env below).
PY = os.environ.get("GLLM_SMOKE_PYTHON", sys.executable)
_CUDA_HOME = "/mnt/sdb/home/gty/miniconda3/envs/gllm-cu130"
_compat = os.path.join(_CUDA_HOME, "opt/cuda-compat-13-0/compat")
_cuda_lib = os.path.join(_CUDA_HOME, "targets/x86_64-linux/lib")
MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")

# 81920 MiB GPUs; leave generous headroom, never touch GPU 0 (production).
GPU = "1"
GPU_MEMORY_UTIL = "0.3"
STARTUP_TIMEOUT = 240  # 0.6B load + cuda init + bind


def _model_path():
    cands = sorted(glob.glob(MODEL))
    return cands[-1].rstrip("/") if cands else None


requires_model = pytest.mark.skipif(
    not _model_path(),
    reason="Qwen3-0.6B snapshot not found locally; real-launch smoke needs a model",
)


def _compat_env():
    """Environment for the smoke subprocess: HF offline + the cuda-compat
    LD_LIBRARY_PATH that lets the cu130 torch talk to the box's 12.x
    driver (same setup the production launcher uses)."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_HOME"] = _CUDA_HOME
    env["PATH"] = _CUDA_HOME + "/bin:" + env.get("PATH", "")
    ld = _compat + ":" + _cuda_lib
    if env.get("LD_LIBRARY_PATH"):
        ld += ":" + env["LD_LIBRARY_PATH"]
    env["LD_LIBRARY_PATH"] = ld
    lp = _cuda_lib + ":" + _cuda_lib + "/stubs"
    if env.get("LIBRARY_PATH"):
        lp += ":" + env["LIBRARY_PATH"]
    env["LIBRARY_PATH"] = lp
    return env


def _drain_pipe(pipe):
    """Non-blocking drain of a subprocess PIPE (for failure diagnostics)."""
    import select
    out = ""
    while True:
        try:
            r, _, _ = select.select([pipe], [], [], 2.0)
        except Exception:
            break
        if not r:
            break
        try:
            chunk = os.read(pipe.fileno(), 65536)
        except Exception:
            break
        if not chunk:
            break
        out += chunk.decode(errors="replace")
    return out


def _wait_for(pred, timeout, interval=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _start_registry():
    """An in-process endpoint-registry middleware on a free loopback port;
    returns (server, "HOST:PORT")."""
    import socket
    import threading

    from gllm.disagg.discovery import DiscoveryServer

    sock = socket.socket(); sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]; sock.close()
    addr = f"127.0.0.1:{port}"
    server = DiscoveryServer(addr)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, addr


def _stop_registry(server):
    try:
        server.stop()
    except Exception:
        pass


def _registry_has_member(addr):
    """True once the worker fleet has registered in the registry."""
    try:
        from gllm.disagg.discovery import make_discovery
        d = make_discovery(addr)
        try:
            return len(d.list("gllm-worker")) > 0
        finally:
            d.close()
    except Exception:
        return False


def _registry_first_row(addr):
    """The rank-0 transport row the worker published (for assertions)."""
    from gllm.disagg.discovery import make_discovery
    d = make_discovery(addr)
    try:
        members = d.list("gllm-worker")
        assert members, "no gllm-worker member in registry"
        return members[0]["payload"]["endpoints"]["0"]
    finally:
        d.close()


def _kill(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)


@requires_model
def test_monolith_worker_launches_and_serves():
    """BLOCKER regression: run_worker's parent watchdog must not block
    worker.init() (a synchronous infinite loop parked every non-overlap
    child before its first CUDA call; wait_workers deadlocked and NO
    deployment -- monolith included -- could start).

    Launch the real worker_server entrypoint (which spawns a real GPU
    child through run_worker) and require the fleet to REGISTER with the
    endpoint registry -- registration happens AFTER child init + bind, so
    a hung child means no registration within the timeout.
    """
    server, addr = _start_registry()
    model = _model_path()
    env = _compat_env()
    # worker_server pins CVV from --worker-gpu (physical); do not preset.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = [
        PY, "-m", "gllm.entrypoints.worker_server",
        "--model-path", model,
        "--worker-gpu", GPU,  # PHYSICAL ordinal; worker_server pins CVV to it
        "--endpoint-registry-addr", addr,
        "--tp", "1",
        "--gpu-memory-util", GPU_MEMORY_UTIL,
        "--master-addr", "127.0.0.1",
        "--master-port", "29631",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        ok = _wait_for(lambda: _registry_has_member(addr), STARTUP_TIMEOUT)
        assert ok, (
            "worker fleet never registered with the endpoint registry -- "
            "the spawned child never finished init (watchdog/launch regression)\n"
            + (proc.stdout.read().decode(errors="replace")[-4000:] if proc.poll() is not None else "")
        )
        # Local ipc:// transport (no base port): rows must be ipc:// paths.
        row = _registry_first_row(addr)
        assert all(k in row for k in ("schedule", "output", "token"))
        assert all(v.startswith("ipc://") for v in row.values())
    finally:
        _kill(proc)
        _stop_registry(server)


@requires_model
def test_standalone_frontend_connects_and_generates():
    """End-to-end: real worker fleet + real standalone frontend + real
    completion through the decoupled transport."""
    server, addr = _start_registry()
    model = _model_path()
    env = _compat_env()
    # worker_server pins CVV from --worker-gpu (physical); do not preset.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    wcmd = [
        PY, "-m", "gllm.entrypoints.worker_server",
        "--model-path", model,
        "--worker-gpu", GPU,
        "--endpoint-registry-addr", addr,
        "--tp", "1",
        "--gpu-memory-util", GPU_MEMORY_UTIL,
        "--master-addr", "127.0.0.1",
        "--master-port", "29632",
    ]
    fcmd = [
        PY, "-m", "gllm.entrypoints.api_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", "18123",
        "--standalone-frontend",
        "--endpoint-registry-addr", addr,
    ]
    worker = subprocess.Popen(wcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    frontend = None
    try:
        ok = _wait_for(lambda: _registry_has_member(addr), STARTUP_TIMEOUT)
        if not ok:
            out = _drain_pipe(worker.stdout)
            raise AssertionError(
                "worker fleet never registered with the endpoint registry\n" + out[-6000:]
            )
        frontend = subprocess.Popen(
            fcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        import urllib.request

        def health_ok():
            try:
                with urllib.request.urlopen("http://127.0.0.1:18123/health", timeout=2) as r:
                    return r.status == 200
            except Exception:
                return False

        assert _wait_for(health_ok, 120), "frontend /health never came up"
        # Real completion over the decoupled transport.
        import urllib.request

        req = urllib.request.Request(
            "http://127.0.0.1:18123/v1/completions",
            data=json.dumps({
                # api_server validates the model id against the served set
                # (model_path basename); use it, not an arbitrary string.
                "model": os.path.basename(model.rstrip("/")),
                "prompt": "Hello, world!",
                "max_tokens": 8,
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        assert body.get("choices"), body
        text = body["choices"][0].get("text", "")
        assert isinstance(text, str)
    finally:
        if frontend is not None:
            _kill(frontend)
        _kill(worker)


@requires_model
def test_standalone_via_in_memory_registry_proxy():
    """End-to-end THROUGH the in-memory registry proxy middleware: the
    control plane (worker register / frontend discover) is served by a real
    DiscoveryServer, and the data plane stays frontend<->worker point-to-point
    zmq (the proxy never forwards tokens). Verifies the registry-backed
    standalone pair actually launches + generates."""
    import urllib.request

    server, addr = _start_registry()
    model = _model_path()
    env = _compat_env()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    wcmd = [
        PY, "-m", "gllm.entrypoints.worker_server",
        "--model-path", model,
        "--worker-gpu", GPU,
        "--endpoint-registry-addr", addr,
        "--tp", "1",
        "--gpu-memory-util", GPU_MEMORY_UTIL,
        "--master-addr", "127.0.0.1",
        "--master-port", "29641",
    ]
    fcmd = [
        PY, "-m", "gllm.entrypoints.api_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", "18133",
        "--standalone-frontend",
        "--endpoint-registry-addr", addr,
    ]
    worker = subprocess.Popen(wcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    frontend = None
    try:
        ok = _wait_for(lambda: _registry_has_member(addr), STARTUP_TIMEOUT)
        if not ok:
            out = _drain_pipe(worker.stdout)
            raise AssertionError(
                "worker never registered with the proxy registry\n" + out[-6000:]
            )
        frontend = subprocess.Popen(
            fcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )

        def health_ok():
            try:
                with urllib.request.urlopen("http://127.0.0.1:18133/health", timeout=2) as r:
                    return r.status == 200
            except Exception:
                return False

        assert _wait_for(health_ok, 120), "frontend /health never came up via proxy"
        req = urllib.request.Request(
            "http://127.0.0.1:18133/v1/completions",
            data=json.dumps({
                "model": os.path.basename(model.rstrip("/")),
                "prompt": "Hello, world!",
                "max_tokens": 8,
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        assert body.get("choices"), body
        assert isinstance(body["choices"][0].get("text", ""), str)
    finally:
        if frontend is not None:
            _kill(frontend)
        _kill(worker)
        server.stop()
