import gzip
import os
import shutil
import time

import torch
from logger import logger

from gllm.distributed.parallel_state import get_world_size


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class TorchProfilerMixin:
    def init_profiler_state(self):
        self.profiler = None
        self.profile_start_ts = None
        self.profile_output_dir = os.getenv("GLLM_TORCH_PROFILER_DIR", "/tmp")
        # ``key_averages()`` materializes an additional in-memory view of all
        # recorded events.  For long profiles that view can be many times
        # larger than the chrome trace itself, so avoid building it once the
        # raw trace is already large.  The compressed chrome trace remains the
        # source of truth and can still be aggregated with streaming tools.
        self.profile_summary_max_trace_bytes = int(
            os.getenv(
                "GLLM_TORCH_PROFILER_SUMMARY_MAX_TRACE_BYTES",
                str(2 * 1024**3),
            )
        )
        self.profile_record_shapes = _env_flag(
            "GLLM_TORCH_PROFILER_RECORD_SHAPES", True
        )
        self.profile_with_stack = _env_flag("GLLM_TORCH_PROFILER_WITH_STACK", True)
        self.profile_session_dir = None

    def _start_profiler(self, profile_session_dir=None):
        if self.profiler is not None:
            logger.warning("Torch profiler is already running")
            return

        os.makedirs(self.profile_output_dir, exist_ok=True)

        if profile_session_dir:
            self.profile_session_dir = profile_session_dir
            session_name = os.path.basename(profile_session_dir)
            if session_name.startswith("trace_session_"):
                self.profile_start_ts = int(session_name[len("trace_session_") :])
            else:
                self.profile_start_ts = int(time.time())
        else:
            self.profile_start_ts = int(time.time())
            self.profile_session_dir = os.path.join(
                self.profile_output_dir,
                f"trace_session_{self.profile_start_ts}",
            )

        os.makedirs(self.profile_session_dir, exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self.profile_record_shapes,
            profile_memory=False,
            with_stack=self.profile_with_stack,
        )
        self.profiler.start()
        logger.info("Torch profiler started")

    def _stop_profiler(self):
        if self.profiler is None:
            logger.warning("Torch profiler is not running")
            return

        output_dir = self.profile_session_dir or self.profile_output_dir
        trace_path = os.path.join(
            output_dir,
            f"trace_rank{self.rank}_{self.profile_start_ts}.json",
        )
        trace_gz_path = f"{trace_path}.gz"
        self.profiler.stop()
        self.profiler.export_chrome_trace(trace_path)
        trace_size = os.path.getsize(trace_path)
        with open(trace_path, "rb") as src, gzip.open(trace_gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(trace_path)
        # Also dump a human-readable ``key_averages`` summary next to the
        # trace so bottlenecks can be inspected without parsing the (often
        # hundreds of MB) chrome trace. Best-effort: never let summary
        # generation break the trace export above.
        if trace_size <= self.profile_summary_max_trace_bytes:
            try:
                summary_path = os.path.join(
                    output_dir,
                    f"summary_rank{self.rank}_{self.profile_start_ts}.txt",
                )
                ka = self.profiler.key_averages()
                with open(summary_path, "w") as f:
                    f.write("== sorted by self_cuda_time_total ==\n")
                    f.write(ka.table(sort_by="self_cuda_time_total", row_limit=40))
                    f.write("\n\n== sorted by self_cpu_time_total ==\n")
                    f.write(ka.table(sort_by="self_cpu_time_total", row_limit=40))
                logger.info(f"Torch profiler summary saved to {summary_path}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to write profiler summary: {e}")
        else:
            logger.warning(
                "Skipping torch profiler key_averages summary because the raw "
                f"trace is {trace_size / 1024**3:.2f} GiB, above the "
                f"{self.profile_summary_max_trace_bytes / 1024**3:.2f} GiB limit"
            )
        self.profiler = None
        self.profile_start_ts = None
        self.profile_session_dir = None
        logger.info(f"Torch profiler stopped, trace saved to {trace_gz_path}")

    def _apply_control_cmd(self, cmd_code: int, profile_session_dir=None):
        if cmd_code == 1:
            self._start_profiler(profile_session_dir=profile_session_dir)
        elif cmd_code == 2:
            self._stop_profiler()

    def sync_control_cmd(self, control_cmd):
        cmd_to_send = 0
        profile_session_dir = None
        if self.rank == 0 and control_cmd is not None:
            if control_cmd == "start_profile":
                cmd_to_send = 1
                start_ts = int(time.time())
                profile_session_dir = os.path.join(
                    self.profile_output_dir,
                    f"trace_session_{start_ts}",
                )
            elif control_cmd == "stop_profile":
                cmd_to_send = 2

        if cmd_to_send != 0:
            if get_world_size() > 1:
                # Broadcast command over existing schedule sockets to avoid dist sync stalls.
                self.comm.broadcast_control_cmd(cmd_to_send, profile_session_dir)
            self._apply_control_cmd(cmd_to_send, profile_session_dir)
