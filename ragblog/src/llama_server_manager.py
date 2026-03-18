"""LlamaServer management for VRAM-aware model switching.

Handles lifecycle of llama.cpp server processes to ensure only one model
resides in GPU memory at a time, critical for consumer GPUs with limited VRAM.
"""

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.const import LLM
from src.logger_custom import LOGGER, log_init


@log_init
@dataclass(frozen=True)
class ConfServer:
    """Configuration for llama-server process management.

    Environment variable LLAMA_SERVER_PATH overrides default binary location.
    """

    path_bin: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "LLAMA_SERVER_PATH", "/home/mat/llama.cpp/build/bin/llama-server"
            )
        )
    )
    port: int = 8080
    ctx_size: int = 32768
    sec_wait_load: int = 5
    sec_wait_free: int = 2
    sec_timeout_stop: int = 10


class LlamaServerManager:
    """Manages llama-server lifecycle for single-model VRAM residency.

    Consumer GPUs (e.g., RTX 5060 Ti 16GB) cannot host multiple large
    quantized models simultaneously. This manager ensures sequential
    loading: start -> evaluate -> stop -> next model.
    """

    def __init__(self, conf: Optional[ConfServer] = None):
        self.conf = conf or ConfServer()
        self.process_active: Optional[subprocess.Popen] = None
        self.map_model_hf = {
            LLM.QWEN_3_5_9B_Q8: "unsloth/Qwen3.5-9B-GGUF:Q8_0",
            LLM.GPT_OSS_20B: "unsloth/gpt-oss-20b-GGUF:Q8_0",
            LLM.QWEN_3_5_27B_Q3: "unsloth/Qwen3.5-27B-GGUF:Q3_K_S",
        }

    def start(self, model: LLM) -> subprocess.Popen:
        """Start llama-server with specified model, blocking until ready.

        Args:
            model: LLM enum identifying which quantized model to load.

        Returns:
            Running subprocess.Popen instance.

        Raises:
            ValueError: If model not in HF mapping.
            RuntimeError: If server fails to start.
        """
        if self.process_active:
            LOGGER.warning("Server already running, stopping first")
            self.stop()

        # Kill any existing server on port to avoid conflicts
        self._kill_port(self.conf.port)

        hf_model = self.map_model_hf.get(model)
        if not hf_model:
            raise ValueError(f"No HF mapping for model: {model}")

        cmd = [
            str(self.conf.path_bin),
            "-hf",
            hf_model,
            "-c",
            str(self.conf.ctx_size),
            "--port",
            str(self.conf.port),
        ]

        LOGGER.info(f"Starting llama-server: {' '.join(cmd)}")

        preexec_fn = os.setsid if os.name != "nt" else None
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec_fn,
        )

        self.process_active = process
        LOGGER.info(f"Waiting {self.conf.sec_wait_load}s for model load into VRAM")
        time.sleep(self.conf.sec_wait_load)

        if process.poll() is not None:
            raise RuntimeError("llama-server exited prematurely")

        return process

    def stop(self) -> None:
        """Terminate active server and release VRAM.

        Sends SIGTERM, waits for graceful shutdown, then SIGKILL if needed.
        Includes brief pause to ensure GPU memory reclamation.
        """
        if not self.process_active:
            return

        process = self.process_active
        LOGGER.info(f"Stopping llama-server (PID: {process.pid})")

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=self.conf.sec_timeout_stop)
            LOGGER.info("llama-server stopped gracefully")
        except subprocess.TimeoutExpired:
            LOGGER.warning("Force killing unresponsive server")
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()

        self.process_active = None
        time.sleep(self.conf.sec_wait_free)
        LOGGER.info("VRAM released")

    def _kill_port(self, port: int) -> None:
        """Kill any process listening on the given port.

        Uses lsof to find and terminate processes to avoid orphaned servers.
        """
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        LOGGER.info(f"Killing orphaned process on port {port}: {pid}")
                        os.kill(int(pid), signal.SIGTERM)
                    except (OSError, ValueError):
                        pass
                time.sleep(1)
        except FileNotFoundError:
            LOGGER.warning("lsof not found, cannot kill orphaned port processes")

    def is_running(self) -> bool:
        """Check if server process is active and responsive."""
        if not self.process_active:
            return False
        return self.process_active.poll() is None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit ensures cleanup."""
        self.stop()
        return False
