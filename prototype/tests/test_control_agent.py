"""Tests for the lab agent core, with a fake training process (no torch run, no network)."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # shared log samples in test_control_protocol

from gpubench.control import agent_core, protocol  # noqa: E402
from gpubench.control.agent_core import AgentCore, AgentError  # noqa: E402
from test_control_protocol import NATIVE_CRASH  # noqa: E402

PROTOTYPE = Path(__file__).resolve().parents[1]
RUN = "thu-localsgd-tiny_cnn-20261116-100000"
PARAMS = {"mode": "localsgd", "quick": True, "dataset": "synthetic", "model": "tiny_cnn", "local_steps": 4}
WORKERS = ["sim-0", "sim-1"]


def make_configs(root: Path) -> Path:
    configs = root / "configs"
    (configs / "machines").mkdir(parents=True)
    for name in WORKERS:
        shutil.copy(PROTOTYPE / "configs" / "machines" / f"{name}.yaml", configs / "machines")
    return configs


def write_events(path: Path, *events: dict, partial: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write(partial)


class FakeProcess:
    def __init__(self):
        self.exit_code = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = 1

    def kill(self):
        self.killed = True
        self.exit_code = 1

    def wait(self, timeout=None):
        return self.exit_code


class FakeLauncher:
    def __init__(self, log_text: str = "nhật ký huấn luyện\n"):
        self.calls = []
        self.process = FakeProcess()
        self.log_text = log_text

    def __call__(self, argv, cwd, log_path):
        self.calls.append((argv, cwd, log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(self.log_text, encoding="utf-8")
        return self.process


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class AgentCoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.configs = make_configs(self.tmp)
        self.results = self.tmp / "results"
        self.launcher = FakeLauncher()
        self.clock = Clock()
        self.decision = ("allow", "")
        self.asked = 0
        self.core = AgentCore("sim-1", self.approve, self.launcher, configs_dir=self.configs,
                              results_dir=self.results, project_dir=PROTOTYPE, clock=self.clock)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def approve(self, payload):
        self.asked += 1
        return self.decision

    def request(self, run_id=RUN, params=PARAMS):
        return self.core.request({"run_id": run_id, "controller": "C", "params": params,
                                  "workers": WORKERS})

    def start(self, run_id=RUN, params=PARAMS):
        return self.core.start({"run_id": run_id, "params": params, "workers": WORKERS,
                                "master_addr": "127.0.0.1"})

    @property
    def log_file(self) -> Path:
        return self.results / RUN / "rank1-sim-1.jsonl"

    def test_request_dong_y_roi_start_chay_dung_lenh(self):
        reply = self.request()
        self.assertEqual((reply["decision"], reply["machine"], reply["run_id"]), ("allow", "sim-1", RUN))
        self.start()
        self.assertEqual(self.core.state, "RUNNING")
        argv, cwd, _ = self.launcher.calls[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:6], ["-m", "gpubench", "run", "-m", "sim-1"])
        experiment = Path(argv[argv.index("-e") + 1])
        self.assertEqual(experiment, self.configs / "experiments" / "_runs" / f"{RUN}.yaml")
        self.assertTrue(experiment.is_file())
        self.assertEqual(argv[argv.index("--run-id") + 1], RUN)
        self.assertEqual(argv[argv.index("--master-addr") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--results-dir") + 1], str(self.results))
        self.assertEqual(cwd, PROTOTYPE)

    def test_request_tu_choi_ve_idle(self):
        self.decision = ("deny", "đang bận việc riêng")
        self.assertEqual(self.request()["decision"], "deny")
        self.assertEqual(self.core.state, "IDLE")
        with self.assertRaises(AgentError) as caught:
            self.start()
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(self.launcher.calls, [])

    def test_start_hai_lan_chi_mot_tien_trinh(self):
        self.request()
        self.start()
        with self.assertRaises(AgentError) as caught:
            self.start()
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(len(self.launcher.calls), 1)

    def test_request_khi_ban_bi_tu_choi_khong_hoi_chu_may(self):
        self.request()
        reply = self.request(run_id="thu-ddp-tiny_cnn-20261116-100500")
        self.assertEqual(reply["decision"], "deny")
        self.assertIn("bận", reply["reason"])
        self.assertEqual(self.asked, 1)

    def test_request_dong_thoi_chi_hoi_mot_lan(self):
        gate = threading.Event()

        def slow(payload):
            self.asked += 1
            gate.wait(5)
            return ("allow", "")
        self.core._approver = slow
        first = threading.Thread(target=self.request)
        first.start()
        while self.core.state != "ASKING":
            pass
        self.assertEqual(self.request(run_id="khac")["decision"], "deny")
        gate.set()
        first.join(5)
        self.assertEqual(self.asked, 1)
        self.assertEqual(self.core.state, "APPROVED")

    def test_request_sai_tham_so(self):
        with self.assertRaises(AgentError) as caught:
            self.request(params={"mode": "ddp", "command": "x"})
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(self.asked, 0)

    def test_run_id_nguy_hiem_bi_tu_choi(self):
        for bad in (r"..\machines\sim-1", "../x", r"C:\Users", "a/b", "", ".hidden", "x" * 120, "a:b"):
            with self.subTest(run_id=bad), self.assertRaises(AgentError) as caught:
                self.request(run_id=bad)
            self.assertEqual(caught.exception.status, 400)
        self.assertEqual(self.asked, 0)

    def test_results_chi_cho_luot_da_chay_tren_agent(self):
        (self.results / "khac").mkdir(parents=True)
        (self.results / "khac" / "bi-mat.txt").write_text("x", encoding="utf-8")
        for run_id in ("khac", "..", r"..\..", r"C:\Users"):
            with self.subTest(run_id=run_id), self.assertRaises(AgentError):
                self.core.results_zip(run_id)

    def test_status_va_stop_tu_choi_run_id_la(self):
        with self.assertRaises(AgentError) as caught:
            self.core.status(r"..\..")
        self.assertIn(caught.exception.status, (400, 404))

    def test_request_may_khong_co_trong_danh_sach(self):
        with self.assertRaises(AgentError):
            self.core.request({"run_id": RUN, "controller": "C", "params": PARAMS, "workers": ["sim-0"]})

    def test_phe_duyet_het_han(self):
        self.request()
        self.clock.now += protocol.APPROVAL_VALID_S + 1
        with self.assertRaises(AgentError) as caught:
            self.start()
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(self.core.state, "IDLE")

    def test_phe_duyet_het_han_tu_ve_ranh(self):
        self.request()
        self.clock.now += protocol.APPROVAL_VALID_S + 1
        self.assertEqual(self.core.state, "IDLE")
        self.assertEqual(self.request(run_id="thu-ddp-tiny_cnn-20261116-120000")["decision"], "allow")

    def test_params_khac_luc_xin_phep(self):
        self.request()
        with self.assertRaises(AgentError) as caught:
            self.start(params={**PARAMS, "local_steps": 8})
        self.assertEqual(caught.exception.status, 409)

    def test_ket_thuc_binh_thuong(self):
        self.request()
        self.start()
        write_events(self.log_file,
                     {"event": "train_start", "steps_per_epoch": 10, "total_steps": 20},
                     {"event": "step", "step": 10, "epoch": 0},
                     {"event": "eval", "step": 10, "epoch": 0, "train_time_s": 3.0, "test_acc": 0.4},
                     {"event": "eval", "step": 20, "epoch": 1, "train_time_s": 6.0, "test_acc": 0.6},
                     {"event": "train_end", "stop_reason": "max_epochs"})
        self.launcher.process.exit_code = 0
        st = self.core.status(RUN)
        self.assertEqual(st["state"], "FINISHED")
        self.assertEqual((st["epoch"], st["test_acc"], st["total_epochs"], st["exit_code"]), (2, 0.6, 2, 0))
        self.assertEqual(st["points"], [[3.0, 0.4], [6.0, 0.6]])

    def test_ket_thuc_loi(self):
        self.request()
        self.start()
        write_events(self.log_file, {"event": "error", "traceback": "Traceback ...\nRuntimeError: gloo"})
        self.launcher.process.exit_code = 1
        st = self.core.status(RUN)
        self.assertEqual(st["state"], "FAILED")
        self.assertIn("RuntimeError: gloo", st["error"])

    def test_ket_thuc_loi_khong_co_jsonl_dung_log(self):
        self.request()
        self.start()
        self.launcher.process.exit_code = 2
        st = self.core.status(RUN)
        self.assertEqual(st["state"], "FAILED")
        self.assertIn("nhật ký huấn luyện", st["error"])

    def test_dung_tu_controller(self):
        self.request()
        self.start()
        st = self.core.stop(RUN, "Người điều phối dừng", by="controller")
        self.assertEqual((st["state"], st["stopped_by"]), ("STOPPED", "controller"))
        self.assertTrue(self.launcher.process.terminated)

    def test_dung_khi_moi_duoc_duyet(self):
        self.request()
        self.core.stop(RUN, "huỷ", by="controller")
        self.assertEqual(self.core.state, "IDLE")

    def test_luot_moi_sau_khi_xong(self):
        self.request()
        self.start()
        self.launcher.process.exit_code = 0
        write_events(self.log_file, {"event": "train_end"})
        self.core.status(RUN)
        self.assertEqual(self.request(run_id="thu-ddp-tiny_cnn-20261116-110000")["decision"], "allow")

    def test_status_sai_run_id(self):
        self.request()
        with self.assertRaises(AgentError) as caught:
            self.core.status("khac")
        self.assertEqual(caught.exception.status, 404)

    def test_results_zip(self):
        self.request()
        self.start()
        write_events(self.log_file, {"event": "train_end"})
        self.launcher.process.exit_code = 0
        names = zipfile.ZipFile(io.BytesIO(self.core.results_zip(RUN))).namelist()
        self.assertIn("rank1-sim-1.jsonl", names)
        self.assertIn("agent-sim-1.log", names)

    def test_results_khi_hong_truoc_luc_huan_luyen_van_co_log(self):
        # A run that dies before the trainer creates its folder still has the agent's log.
        self.request()
        self.start()
        self.launcher.process.exit_code = 1
        self.assertEqual(self.core.status(RUN)["state"], "FAILED")
        names = zipfile.ZipFile(io.BytesIO(self.core.results_zip(RUN))).namelist()
        self.assertEqual(names, ["agent-sim-1.log"])

    def test_loi_la_cac_dong_co_nghia_khong_phai_stack_trace(self):
        self.launcher.log_text = NATIVE_CRASH
        self.request()
        self.start()
        self.launcher.process.exit_code = 3
        error = self.core.status(RUN)["error"]
        self.assertIn("Unhandled exception caught", error)
        self.assertIn("failed to connect", error)
        self.assertNotIn("THPGenerator", error)

    def test_netcheck_chay_lenh_netcheck_va_xong_khi_thoat_0(self):
        params = {"mode": "netcheck"}
        run_id = "lab-netcheck-20261118-090000"
        self.request(run_id=run_id, params=params)
        self.core.start({"run_id": run_id, "params": params, "workers": WORKERS,
                         "master_addr": "127.0.0.1"})
        argv = self.launcher.calls[0][0]
        self.assertEqual(argv[1:6], ["-m", "gpubench", "netcheck", "-m", "sim-1"])
        self.assertEqual(argv[argv.index("--run-id") + 1], run_id)
        self.assertEqual(argv[argv.index("--results-dir") + 1], str(self.results))
        self.launcher.process.exit_code = 0
        self.assertEqual(self.core.status(run_id)["state"], "FINISHED")

    def test_results_zip_khong_co(self):
        with self.assertRaises(AgentError) as caught:
            self.core.results_zip("khong-co")
        self.assertEqual(caught.exception.status, 404)

    def test_shutdown_khi_dang_chay(self):
        self.request()
        self.start()
        self.core.shutdown()
        self.assertTrue(self.launcher.process.terminated)
        self.assertEqual(self.core.status(RUN)["stopped_by"], "agent")

    def test_info_khong_cuda(self):
        info = self.core.info()
        self.assertEqual((info["name"], info["address"], info["state"]), ("sim-1", "127.0.0.1", "IDLE"))
        self.assertEqual(info["gpu"]["cuda_available"], False)
        self.assertIsNone(info["gpu"]["vram_mb"])
        self.assertEqual(info["agent_version"], protocol.AGENT_VERSION)


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class DefaultLauncherTest(unittest.TestCase):
    def test_dung_giet_ca_tien_trinh_python_that(self):
        # With a venv on Windows, sys.executable is a launcher that starts the real interpreter
        # as a child. Stopping must end the interpreter that actually trains, not just the launcher.
        tmp = Path(tempfile.mkdtemp())
        try:
            pid_file = tmp / "pid.txt"
            script = f"import os, time; open(r'{pid_file}', 'w').write(str(os.getpid())); time.sleep(120)"
            process = agent_core.default_launcher([sys.executable, "-c", script], tmp, tmp / "log.txt")
            deadline = time.monotonic() + 30
            while not (pid_file.is_file() and pid_file.read_text()):
                self.assertLess(time.monotonic(), deadline, "tiến trình không khởi động")
                time.sleep(0.1)
            real_pid = int(pid_file.read_text())
            process.terminate()
            process.wait(15)
            time.sleep(0.5)
            self.assertIsNotNone(process.poll())
            self.assertFalse(pid_alive(real_pid), "tiến trình Python thật vẫn còn chạy")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ReadProgressTest(unittest.TestCase):
    def test_bo_dong_hong(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "r.jsonl"
            # The trainer numbers epochs from 0; progress reports epochs completed.
            write_events(path, {"event": "eval", "epoch": 0, "step": 5, "train_time_s": 1.0, "test_acc": 0.3},
                         partial='{"event": "eval", "epo')
            progress = agent_core.read_progress(path)
            self.assertEqual((progress["epoch"], progress["test_acc"], progress["finished"]), (1, 0.3, False))
            self.assertEqual(agent_core.read_progress(tmp / "khong-co.jsonl")["points"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
