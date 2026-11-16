"""The controller side of the lab control system, without UI: one training run end to end.

    check() -> request_all() -> start() -> poll_once() ... done() -> collect() -> analyze()

A run can be picked up again with RunSession.resume() after the controller was closed:
the agents keep training, and controller.json in the run folder remembers the run.
"""
from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from .protocol import (APPROVAL_TIMEOUT_S, DEFAULT_PORT, TOKEN_HEADER, ProtocolError, RunParams,
                       new_run_id, rank_order, valid_run_id, validate_params)

FREE_STATES = ("IDLE", "FINISHED", "FAILED", "STOPPED")
DONE_STATES = ("FINISHED", "FAILED", "STOPPED", "LOST")
# A machine that stops answering is given up on only when every other machine is done:
# after gloo's own timeout (timeout_s 300) plus a margin, or sooner once a stop was sent.
UNREACHABLE_LIMIT_S = 360
STOPPING_GRACE_S = 60
ANALYZE_ACC = ["0.5", "0.6", "0.7", "0.8"]
ANALYZE_MINUTES = ["10", "30"]


class AgentUnavailable(Exception):
    def __init__(self, name: str, message: str):
        super().__init__(f"Không kết nối được với {name}: {message}")
        self.name = name


class AgentRejected(Exception):
    def __init__(self, name: str, status: int, message: str):
        super().__init__(f"{name} từ chối ({status}): {message}")
        self.name, self.status, self.message = name, status, message


@dataclass(frozen=True)
class AgentRef:
    name: str
    address: str
    port: int = DEFAULT_PORT


def load_lab(path: Path) -> tuple[list[AgentRef], str]:
    if not path.is_file():
        raise ProtocolError(f"Không tìm thấy {path}. Chép configs/lab.yaml.example thành "
                            f"configs/lab.yaml rồi sửa địa chỉ các agent.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        agents = [AgentRef(a["name"], a["address"], int(a.get("port", DEFAULT_PORT)))
                  for a in data["agents"]]
        return agents, str(data["token"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProtocolError(f"{path}: cần 'token' và danh sách 'agents' có name, address ({e})") from None


class AgentClient:
    def __init__(self, ref: AgentRef, token: str, timeout: float = 10):
        self.ref = ref
        self.name = ref.name
        self._token = token
        self._timeout = timeout
        # LAN addresses must never go through a system proxy or VPN proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout: float | None = None) -> tuple[bytes, str]:
        url = f"http://{self.ref.address}:{self.ref.port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={TOKEN_HEADER: self._token,
                                              "Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=timeout or self._timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except (ValueError, AttributeError):
                message = str(e.reason)
            raise AgentRejected(self.name, e.code, message) from None
        except (urllib.error.URLError, OSError) as e:
            raise AgentUnavailable(self.name, str(getattr(e, "reason", e))) from None

    def _json(self, *args, **kwargs) -> dict:
        return json.loads(self._call(*args, **kwargs)[0])

    def info(self) -> dict:
        return self._json("GET", "/info")

    def request(self, payload: dict, timeout: float = APPROVAL_TIMEOUT_S + 15) -> dict:
        return self._json("POST", "/request", payload, timeout=timeout)

    def start(self, payload: dict) -> dict:
        return self._json("POST", "/start", payload, timeout=60)

    def status(self, run_id: str) -> dict:
        return self._json("GET", f"/status?run_id={run_id}")

    def stop(self, run_id: str, reason: str) -> dict:
        return self._json("POST", "/stop", {"run_id": run_id, "reason": reason}, timeout=30)

    def results(self, run_id: str) -> bytes:
        return self._call("GET", f"/results?run_id={run_id}", timeout=120)[0]


def _parallel(clients: list[AgentClient], fn: Callable[[AgentClient], object]) -> dict:
    """{name: result or the exception it raised}."""
    with ThreadPoolExecutor(max_workers=max(1, len(clients))) as pool:
        futures = {c.name: pool.submit(fn, c) for c in clients}
    out = {}
    for name, future in futures.items():
        try:
            out[name] = future.result()
        except Exception as e:
            out[name] = e
    return out


def _safe_extract(blob: bytes, target: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            path = (target / member).resolve()
            if target.resolve() not in path.parents:
                raise ProtocolError(f"Tệp kết quả có đường dẫn không hợp lệ: {member}")
        archive.extractall(target)


class RunSession:
    def __init__(self, agents: list[AgentClient], params: RunParams, results_root: Path,
                 quick_root: Path, prototype_dir: Path,
                 on_event: Callable[[str, dict], None] = lambda kind, data: None,
                 clock: Callable[[], float] = time.time):
        self.agents = list(agents)
        self.params = params
        self.results_root = results_root
        self.quick_root = quick_root
        self.prototype_dir = prototype_dir
        self.on_event = on_event
        self.clock = clock
        self.run_id: str | None = None
        self.workers: list[str] = []
        self.master_addr: str | None = None
        self.infos: dict[str, dict] = {}
        self.decisions: dict[str, tuple[str, str]] = {}
        self.approved = False
        self.started = False
        self.stop_reason: str | None = None
        self.last_statuses: dict[str, dict] = {}
        self._unreachable_since: dict[str, float] = {}

    @property
    def run_dir(self) -> Path:
        root = self.quick_root if self.params.quick else self.results_root
        return root / self.run_id

    def _client(self, name: str) -> AgentClient:
        return next(c for c in self.agents if c.name == name)

    def _active(self) -> list[AgentClient]:
        return [self._client(name) for name in self.workers]

    # ---------------------------------------------------------------- before training

    def check(self) -> list[dict]:
        if len(self.agents) < 2:
            raise ProtocolError("Cần chọn ít nhất 2 máy")
        replies = _parallel(self.agents, lambda c: c.info())
        problems = []
        for name, reply in replies.items():
            if isinstance(reply, Exception):
                problems.append(str(reply))
            elif reply.get("state") not in FREE_STATES:
                problems.append(f"{name} đang bận (lượt {reply.get('run_id')})")
            elif not self.params.quick and not (reply.get("gpu") or {}).get("cuda_available"):
                problems.append(f"{name} không có GPU CUDA dùng được")
            elif not reply.get("address"):
                problems.append(f"{name} chưa có address trong file máy")
        if problems:
            raise ProtocolError("; ".join(problems))
        self.infos = replies
        self.workers = rank_order(list(replies.values()))
        self.master_addr = replies[self.workers[0]]["address"]
        self.run_id = new_run_id(self.params, datetime.fromtimestamp(self.clock()))
        self.on_event("checked", {"run_id": self.run_id, "workers": self.workers,
                                  "infos": replies})
        return [replies[name] for name in self.workers]

    def _payload(self) -> dict:
        return {"run_id": self.run_id, "controller": socket.gethostname(),
                "params": self.params.to_dict(), "workers": self.workers}

    def request_all(self) -> dict[str, tuple[str, str]]:
        payload = self._payload()

        def ask(client: AgentClient):
            try:
                reply = client.request(payload)
                result = (reply["decision"], reply.get("reason", ""))
            except (AgentUnavailable, AgentRejected) as e:
                result = ("deny", str(e))
            self.on_event("decision", {"machine": client.name, "decision": result[0],
                                       "reason": result[1]})
            return result

        self.decisions = _parallel(self._active(), ask)
        self.approved = all(d == "allow" for d, _ in self.decisions.values())
        if not self.approved:   # release the machines that did allow
            self.cancel("Không đủ máy đồng ý; lượt bị huỷ")
        return dict(self.decisions)

    def cancel(self, reason: str) -> None:
        """Releases every machine of a run that has not started. Errors are ignored."""
        _parallel(self._active(), lambda c: c.stop(self.run_id, reason))
        self.approved = False
        self.on_event("cancelled", {"reason": reason})

    def _write_record(self, **updates) -> None:
        path = self.run_dir / "controller.json"
        record = json.loads(path.read_text("utf-8")) if path.is_file() else {}
        record.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def start(self) -> None:
        if not self.approved:
            raise ProtocolError("Chưa đủ máy đồng ý; không thể bắt đầu")
        self._write_record(run_id=self.run_id, params=self.params.to_dict(), workers=self.workers,
                           master_addr=self.master_addr, quick=self.params.quick,
                           agents=[asdict(c.ref) for c in self._active()],
                           decisions={k: list(v) for k, v in self.decisions.items()},
                           started_at=datetime.fromtimestamp(self.clock()).isoformat(timespec="seconds"))
        payload = {**self._payload(), "master_addr": self.master_addr}
        for client in self._active():
            try:
                client.start(payload)
            except (AgentUnavailable, AgentRejected) as e:
                # Stop everyone, the failing machine too: it may have started and lost the reply.
                self.cancel(f"{client.name} không bắt đầu được")
                self._write_record(outcome={"state": "failed", "reason": str(e)})
                raise ProtocolError(str(e)) from None
        self.started = True
        self.on_event("started", {"run_id": self.run_id})

    # ---------------------------------------------------------------- during training

    def _reason(self, name: str, status: dict) -> str:
        if status["state"] == "LOST":
            if status.get("unreachable"):
                return f"Mất liên lạc với {name} quá lâu; không gom được kết quả của máy này"
            return f"{name} không còn giữ lượt chạy này (agent đã khởi động lại?)"
        if status["state"] == "STOPPED":
            who = status.get("stopped_by")
            if who == "owner":
                return f"Chủ máy {name} đã dừng huấn luyện"
            if who == "agent":
                return f"Agent trên {name} đã bị đóng"
            return f"{name} đã dừng"
        lines = [line for line in (status.get("error") or "").splitlines() if line.strip()]
        return f"{name} gặp lỗi: {lines[-1] if lines else 'không rõ nguyên nhân'}"

    def poll_once(self) -> dict[str, dict]:
        def ask(client: AgentClient):
            return client.status(self.run_id)
        statuses = {}
        for name, reply in _parallel(self._active(), ask).items():
            if isinstance(reply, AgentRejected) and reply.status == 404:
                statuses[name] = {"state": "LOST", "machine": name}
            elif isinstance(reply, Exception):
                since = self._unreachable_since.setdefault(name, self.clock())
                statuses[name] = {"state": "UNREACHABLE", "machine": name, "since": since,
                                  "error": str(reply)}
            else:
                self._unreachable_since.pop(name, None)
                statuses[name] = reply
        self._give_up_on_unreachable(statuses)
        self.last_statuses = statuses

        broken = [n for n, s in statuses.items() if s["state"] in ("FAILED", "STOPPED", "LOST")]
        running = [n for n, s in statuses.items() if s["state"] in ("RUNNING", "UNREACHABLE")]
        if broken and running and self.stop_reason is None:
            self.stop_all(self._reason(broken[0], statuses[broken[0]]))
        self.on_event("progress", statuses)
        return statuses

    def _give_up_on_unreachable(self, statuses: dict[str, dict]) -> None:
        silent = [n for n, s in statuses.items() if s["state"] == "UNREACHABLE"]
        others_done = all(s["state"] in DONE_STATES for n, s in statuses.items() if n not in silent)
        if not silent or not others_done:
            return
        limit = STOPPING_GRACE_S if self.stop_reason else UNREACHABLE_LIMIT_S
        for name in silent:
            if self.clock() - statuses[name]["since"] > limit:
                statuses[name] = {"state": "LOST", "machine": name, "unreachable": True,
                                  "error": statuses[name].get("error")}

    def done(self, statuses: dict[str, dict]) -> bool:
        return all(s["state"] in DONE_STATES for s in statuses.values())

    def stop_all(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self.on_event("stopping", {"reason": reason})
        for name, reply in _parallel(self._active(), lambda c: c.stop(self.run_id, reason)).items():
            if isinstance(reply, Exception):
                self.on_event("stop_failed", {"machine": name, "error": str(reply)})

    # ---------------------------------------------------------------- after training

    def collect(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for name, blob in _parallel(self._active(), lambda c: c.results(self.run_id)).items():
            try:
                if isinstance(blob, Exception):
                    raise blob
                _safe_extract(blob, self.run_dir)
            except Exception as e:
                missing.append(name)
                self.on_event("collect_failed", {"machine": name, "error": str(e)})
        states = [s["state"] for s in self.last_statuses.values()]
        if states and all(s == "FINISHED" for s in states):
            state = "finished"
        elif "STOPPED" in states:
            state = "stopped"
        else:
            state = "failed"
        reason = self.stop_reason or next(
            (self._reason(n, s) for n, s in self.last_statuses.items() if s["state"] != "FINISHED"), "")
        self._write_record(missing=sorted(missing), outcome={
            "state": state, "reason": reason,
            "finished_at": datetime.fromtimestamp(self.clock()).isoformat(timespec="seconds")})
        self.on_event("collected", {"run_dir": str(self.run_dir), "missing": missing, "state": state})
        return self.run_dir

    def analyze(self) -> Path | None:
        if self.params.quick:
            return None
        out = self.results_root / "report-lab"
        argv = [sys.executable, "analyze/analyze.py", "lab-", "--results", str(self.results_root),
                "--out", str(out), "--acc", *ANALYZE_ACC, "--at-minutes", *ANALYZE_MINUTES]
        try:
            done = subprocess.run(argv, cwd=self.prototype_dir, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.on_event("analyze_failed", {"output": str(e)})
            return None
        if done.returncode != 0:
            self.on_event("analyze_failed", {"output": (done.stdout or "") + (done.stderr or "")})
            return None
        return out

    # ---------------------------------------------------------------- reconnect

    @classmethod
    def resume(cls, agents: list[AgentClient], results_root: Path, quick_root: Path,
               prototype_dir: Path, on_event: Callable[[str, dict], None] = lambda kind, data: None,
               clock: Callable[[], float] = time.time) -> "RunSession | None":
        for name, info in _parallel(agents, lambda c: c.info()).items():
            if isinstance(info, Exception) or not info.get("run_id"):
                continue
            if info.get("state") not in ("RUNNING", "FINISHED", "FAILED", "STOPPED"):
                continue
            run_id = info["run_id"]
            if not valid_run_id(run_id):    # it becomes a folder name here
                on_event("resume_invalid", {"machine": name, "run_id": run_id})
                continue
            root =quick_root if run_id.startswith("thu-") else results_root
            path = root / run_id / "controller.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text("utf-8"))
            if "outcome" in record:
                continue
            session = cls([c for c in agents if c.name in record["workers"]],
                          validate_params(record["params"]), results_root, quick_root, prototype_dir,
                          on_event, clock)
            session.run_id, session.workers = run_id, list(record["workers"])
            session.master_addr = record.get("master_addr")
            session.decisions = {k: tuple(v) for k, v in record.get("decisions", {}).items()}
            session.approved = session.started = True
            missing = [w for w in session.workers if w not in {c.name for c in session.agents}]
            if missing:
                on_event("resume_missing", {"machines": missing})
                session.workers = [w for w in session.workers if w not in missing]
            on_event("resumed", {"run_id": run_id})
            return session
        return None
