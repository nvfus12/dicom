"""The worker side of the lab control system, without UI or HTTP.

Holds the approval state machine, runs `gpubench run` as a child process once the owner
has approved that exact run, reads progress from the JSONL log the trainer writes, and
packs the results for the controller.

    IDLE -> ASKING -> APPROVED -> RUNNING -> FINISHED | FAILED | STOPPED
              |deny       |expired/stopped
              v           v
             IDLE        IDLE
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml

from ..app.config_loader import CONFIGS_DIR, PROJECT_DIR, RESULTS_DIR, load_machine
from .protocol import (AGENT_VERSION, APPROVAL_VALID_S, ProtocolError, RunParams, build_experiment,
                       validate_params)

Approver = Callable[[dict], "tuple[str, str]"]
Launcher = Callable[[list, Path, Path], Any]

FREE_STATES = ("IDLE", "FINISHED", "FAILED", "STOPPED")
ERROR_TAIL_CHARS = 2000


class AgentError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def default_launcher(argv: list, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log_path.open("w", encoding="utf-8") as log:
        # The child keeps its own copy of the handle; ours can close right away.
        return subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=flags)


def read_progress(path: Path) -> dict:
    """Summary of a trainer JSONL log. Tolerates a half-written last line."""
    progress = {"epoch": None, "step": None, "test_acc": None, "points": [], "total_epochs": None,
                "finished": False, "error": None}
    if not path.is_file():
        return progress
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if event == "train_start":
            per_epoch, total = record.get("steps_per_epoch"), record.get("total_steps")
            if per_epoch and total:
                progress["total_epochs"] = total // per_epoch
        elif event in ("step", "eval"):
            if record.get("epoch") is not None:     # the trainer counts from 0; show 1-based
                progress["epoch"] = record["epoch"] + 1
            progress["step"] = record.get("step", progress["step"])
            if event == "eval" and record.get("test_acc") is not None:
                progress["test_acc"] = record["test_acc"]
                progress["points"].append([record.get("train_time_s"), record["test_acc"]])
        elif event == "train_end":
            progress["finished"] = True
        elif event == "error":
            progress["error"] = record.get("traceback")
    return progress


def detect_gpu(device: str, cuda_index: int) -> dict:
    gpu = {"name": None, "vram_mb": None, "cuda_available": False}
    if device != "cuda":
        return gpu
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(cuda_index)
            gpu.update(name=props.name, vram_mb=props.total_memory // 2**20, cuda_available=True)
    except Exception:  # a broken torch install must not kill the agent
        pass
    return gpu


def torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


class AgentCore:
    def __init__(self, machine_name: str, approver: Approver, launcher: Launcher = default_launcher,
                 configs_dir: Path = CONFIGS_DIR, results_dir: Path = RESULTS_DIR,
                 project_dir: Path = PROJECT_DIR, clock: Callable[[], float] = time.monotonic):
        self.machine = load_machine(machine_name, configs_dir)
        self.name = self.machine.name
        self._approver = approver
        self._launcher = launcher
        self._configs_dir = configs_dir
        self._results_dir = results_dir
        self._project_dir = project_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._gpu: dict | None = None
        self._torch: str | None = None
        self._reset("IDLE")

    def _reset(self, state: str) -> None:
        self._state = state
        self._run_id: str | None = None
        self._params: RunParams | None = None
        self._workers: list[str] = []
        self._approved_at = 0.0
        self._process = None
        self._exit_code: int | None = None
        self._stopped_by: str | None = None

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh_locked()
            return self._state

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def params(self) -> RunParams | None:
        return self._params

    @property
    def workers(self) -> list[str]:
        return list(self._workers)

    def info(self) -> dict:
        if self._gpu is None:
            self._gpu = detect_gpu(self.machine.device, self.machine.cuda_index)
            self._torch = torch_version()
        with self._lock:
            self._refresh_locked()
            return {"name": self.name, "owner": self.machine.owner, "address": self.machine.address,
                    "gpu": dict(self._gpu), "torch": self._torch, "state": self._state,
                    "run_id": self._run_id, "agent_version": AGENT_VERSION}

    # ---------------------------------------------------------------- approval

    def _parse(self, payload: dict) -> tuple[str, RunParams, list[str]]:
        try:
            run_id, workers = payload["run_id"], payload["workers"]
            params = validate_params(payload["params"])
        except KeyError as e:
            raise AgentError(400, f"Thiếu trường {e}") from None
        except ProtocolError as e:
            raise AgentError(400, str(e)) from None
        if not isinstance(run_id, str) or not run_id:
            raise AgentError(400, "run_id không hợp lệ")
        if not isinstance(workers, list) or not all(isinstance(w, str) for w in workers):
            raise AgentError(400, "workers phải là danh sách tên máy")
        if self.name not in workers:
            raise AgentError(400, f"Máy '{self.name}' không có trong danh sách {workers}")
        return run_id, params, workers

    def request(self, payload: dict) -> dict:
        run_id, params, workers = self._parse(payload)
        with self._lock:
            self._refresh_locked()
            if self._state not in FREE_STATES:
                return {"decision": "deny", "reason": f"Máy đang bận với lượt {self._run_id}",
                        "machine": self.name, "run_id": run_id}
            self._reset("ASKING")
            self._run_id, self._params, self._workers = run_id, params, workers

        decision, reason = self._approver(payload)  # blocks until the owner decides

        with self._lock:
            if self._state != "ASKING" or self._run_id != run_id:  # cancelled meanwhile
                decision, reason = "deny", reason or "Yêu cầu đã bị huỷ"
            elif decision == "allow":
                self._state, self._approved_at = "APPROVED", self._clock()
            else:
                decision = "deny"
                self._reset("IDLE")
        return {"decision": decision, "reason": reason, "machine": self.name, "run_id": run_id}

    # ---------------------------------------------------------------- training

    def start(self, payload: dict) -> dict:
        run_id, params, workers = self._parse(payload)
        master_addr = payload.get("master_addr")
        if not isinstance(master_addr, str) or not master_addr:
            raise AgentError(400, "Thiếu master_addr")
        with self._lock:
            if self._state != "APPROVED" or self._run_id != run_id:
                raise AgentError(409, f"Lượt {run_id} chưa được chủ máy cho phép "
                                      f"(trạng thái hiện tại: {self._state})")
            if self._clock() - self._approved_at > APPROVAL_VALID_S:
                self._reset("IDLE")
                raise AgentError(409, "Sự cho phép đã hết hạn; hãy gửi lại yêu cầu")
            if params != self._params or workers != self._workers:
                raise AgentError(409, "Cấu hình khác với lúc xin phép; hãy gửi lại yêu cầu")

            runs_dir = self._configs_dir / "experiments" / "_runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            experiment = runs_dir / f"{run_id}.yaml"
            experiment.write_text(yaml.safe_dump(build_experiment(params, workers, run_id),
                                                 allow_unicode=True, sort_keys=True), encoding="utf-8")
            argv = [sys.executable, "-m", "gpubench", "run", "-m", self.name, "-e", str(experiment),
                    "--run-id", run_id, "--master-addr", master_addr,
                    "--results-dir", str(self._results_dir)]
            self._process = self._launcher(argv, self._project_dir, self._log_path(run_id))
            self._state = "RUNNING"
        return self.status(run_id)

    def _log_path(self, run_id: str) -> Path:
        return self._results_dir / "_agent" / f"{run_id}.log"

    def _jsonl_path(self, run_id: str) -> Path:
        rank = self._workers.index(self.name)
        return self._results_dir / run_id / f"rank{rank}-{self.name}.jsonl"

    def _refresh_locked(self) -> None:
        if self._state != "RUNNING" or self._process is None:
            return
        code = self._process.poll()
        if code is None:
            return
        self._exit_code = code
        if self._stopped_by:
            self._state = "STOPPED"
        elif code == 0 and read_progress(self._jsonl_path(self._run_id))["finished"]:
            self._state = "FINISHED"
        else:
            self._state = "FAILED"

    def refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    def status(self, run_id: str) -> dict:
        with self._lock:
            if run_id != self._run_id:
                raise AgentError(404, f"Máy '{self.name}' không có lượt {run_id}")
            self._refresh_locked()
            progress = read_progress(self._jsonl_path(run_id))
            error = progress["error"]
            if error is None and self._state == "FAILED":
                log = self._log_path(run_id)
                error = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else None
            return {"state": self._state, "run_id": run_id, "machine": self.name,
                    "epoch": progress["epoch"], "step": progress["step"],
                    "test_acc": progress["test_acc"], "points": progress["points"],
                    "total_epochs": progress["total_epochs"], "exit_code": self._exit_code,
                    "error": error[-ERROR_TAIL_CHARS:] if error else None,
                    "stopped_by": self._stopped_by}

    def stop(self, run_id: str, reason: str, by: str) -> dict:
        with self._lock:
            if run_id != self._run_id:
                raise AgentError(404, f"Máy '{self.name}' không có lượt {run_id}")
            self._refresh_locked()
            if self._state in ("ASKING", "APPROVED"):
                self._reset("IDLE")
                return {"state": "IDLE", "run_id": run_id, "machine": self.name}
            process = self._process if self._state == "RUNNING" else None
            if process is not None:
                self._stopped_by = by
        if process is not None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is None:
                process.kill()
                process.wait(10)
        return self.status(run_id)

    def results_zip(self, run_id: str) -> bytes:
        run_dir = self._results_dir / run_id
        if not run_dir.is_dir():
            raise AgentError(404, f"Máy '{self.name}' không có kết quả của lượt {run_id}")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir).as_posix())
            log = self._log_path(run_id)
            if log.is_file():
                archive.write(log, f"agent-{self.name}.log")
        return buffer.getvalue()

    def shutdown(self) -> None:
        with self._lock:
            self._refresh_locked()
            running = self._state == "RUNNING"
            run_id = self._run_id
        if running:
            self.stop(run_id, "Agent đóng", by="agent")
