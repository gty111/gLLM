import gzip
import os
import shutil
import time

import torch
from logger import logger

from gllm.dist_utils import get_pp_size


class TorchProfilerMixin:
    def init_profiler_state(self):
        self.profiler = None
        self.profile_start_ts = None
        self.profile_output_dir = os.getenv("GLLM_TORCH_PROFILER_DIR", "/tmp")
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
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
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
        with open(trace_path, "rb") as src, gzip.open(trace_gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(trace_path)
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
            if get_pp_size() > 1:
                # Broadcast command over existing schedule sockets to avoid dist sync stalls.
                self.comm.send_control_cmd(cmd_to_send, profile_session_dir)
            self._apply_control_cmd(cmd_to_send, profile_session_dir)
