"""End-to-end tests of the lab controller: two real agents on localhost, real `gpubench run` on CPU.

Each full run starts two Python processes that train a tiny CNN on synthetic data over gloo,
so this file takes a few minutes.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # shared fakes in test_control_agent

from gpubench import cli  # noqa: E402
from gpubench.control import controller_core  # noqa: E402
from gpubench.control.agent_core import AgentCore  # noqa: E402
from gpubench.control.agent_server import make_server  # noqa: E402
from gpubench.control.controller_core import (AgentClient, AgentRef, RunSession,  # noqa: E402
                                              load_lab)
from gpubench.control.protocol import ProtocolError, validate_params  # noqa: E402
from test_control_agent import PROTOTYPE, FakeLauncher, make_configs, write_events  # noqa: E402

TOKEN = "e2e-token"


def quick(mode: str, **extra) -> dict:
    return {"mode": mode, "quick": True, "dataset": "synthetic", "model": "tiny_cnn",
            "local_steps": 4, "max_epochs": 2, "max_train_minutes": 2, **extra}


class Lab:
    """Two agents (sim-0, sim-1) with their own result folders, plus a controller folder."""

    def __init__(self, launcher_factory=None, decisions=None):
        self.tmp = Path(tempfile.mkdtemp())
        self.configs = make_configs(self.tmp)
        self.decisions = decisions or {}
        self.cores, self.servers, self.clients, self.launchers = {}, {}, [], {}
        for name in ("sim-0", "sim-1"):
            kwargs = {}
            if launcher_factory:
                self.launchers[name] = kwargs["launcher"] = launcher_factory()
            core = AgentCore(name, self._approver(name), configs_dir=self.configs,
                             results_dir=self.tmp / f"agent-{name}", project_dir=PROTOTYPE, **kwargs)
            server = make_server(core, "127.0.0.1", 0, TOKEN)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.cores[name], self.servers[name] = core, server
            self.clients.append(AgentClient(AgentRef(name, "127.0.0.1", server.server_address[1]), TOKEN))
        self.results = self.tmp / "results"
        self.quick_results = self.tmp / "results-thu"
        self.events: list[tuple[str, dict]] = []

    def _approver(self, name):
        return lambda payload: self.decisions.get(name, ("allow", ""))

    def session(self, params: dict, clock=time.time) -> RunSession:
        return RunSession(self.clients, validate_params(params), self.results, self.quick_results,
                          PROTOTYPE, on_event=lambda kind, data: self.events.append((kind, data)),
                          clock=clock)

    def close(self):
        for core in self.cores.values():
            core.shutdown()
        for server in self.servers.values():
            server.shutdown()
            server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def poll_until_done(session: RunSession, timeout: float, every: float = 0.5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = session.poll_once()
        if session.done(statuses):
            return statuses
        time.sleep(every)
    raise AssertionError(f"lượt chạy chưa xong sau {timeout} s: {session.poll_once()}")


def outcome(run_dir: Path) -> dict:
    return json.loads((run_dir / "controller.json").read_text("utf-8"))["outcome"]


class FullRunTest(unittest.TestCase):
    def setUp(self):
        self.lab = Lab()

    def tearDown(self):
        self.lab.close()

    def run_mode(self, mode: str) -> Path:
        s = self.lab.session(quick(mode))
        s.check()
        self.assertEqual(s.workers, ["sim-0", "sim-1"])
        self.assertTrue(all(d == "allow" for d, _ in s.request_all().values()))
        self.assertTrue(s.approved)
        s.start()
        statuses = poll_until_done(s, timeout=180)
        self.assertTrue(all(st["state"] == "FINISHED" for st in statuses.values()), statuses)
        run_dir = s.collect()
        self.assertEqual(run_dir.parent, self.lab.quick_results)
        self.assertEqual(sorted(p.name for p in run_dir.glob("rank*.jsonl")),
                         ["rank0-sim-0.jsonl", "rank1-sim-1.jsonl"])
        self.assertEqual(outcome(run_dir)["state"], "finished")
        report = s.analyze()        # real analyze.py on the real quick run
        self.assertIsNotNone(report, [e for e in self.lab.events if e[0] == "analyze_failed"])
        self.assertTrue((report / "runs.csv").is_file())
        self.assertIn(s.run_id, (report / "runs.csv").read_text("utf-8"))
        return run_dir

    def test_chay_thu_localsgd_tron_vong(self):
        self.run_mode("localsgd")

    def test_chay_thu_ddp_tron_vong(self):
        self.run_mode("ddp")

    def test_chay_thu_sync_tron_vong(self):
        self.run_mode("sync")

    def test_chu_may_dung_giua_chung(self):
        s = self.lab.session(quick("ddp", max_epochs=300, max_train_minutes=2))
        s.check()
        s.request_all()
        s.start()
        deadline = time.monotonic() + 120
        while self.lab.cores["sim-1"].status(s.run_id)["step"] is None:
            self.assertLess(time.monotonic(), deadline, "huấn luyện không bắt đầu")
            time.sleep(0.5)
        self.lab.cores["sim-1"].stop(s.run_id, "Chủ máy dừng", by="owner")
        statuses = poll_until_done(s, timeout=120)
        self.assertEqual(statuses["sim-0"]["state"], "STOPPED")
        result = outcome(s.collect())
        self.assertEqual(result["state"], "stopped")
        self.assertIn("sim-1", result["reason"])

    def test_noi_lai_sau_khi_controller_tat(self):
        s = self.lab.session(quick("localsgd"))
        s.check()
        s.request_all()
        s.start()
        run_id = s.run_id
        del s
        resumed = RunSession.resume(self.lab.clients, self.lab.results, self.lab.quick_results, PROTOTYPE)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.run_id, run_id)
        poll_until_done(resumed, timeout=180)
        run_dir = resumed.collect()
        self.assertEqual(len(list(run_dir.glob("rank*.jsonl"))), 2)
        self.assertIsNone(RunSession.resume(self.lab.clients, self.lab.results, self.lab.quick_results,
                                            PROTOTYPE))


class FakeTrainingTest(unittest.TestCase):
    """Same controller logic, training replaced by FakeLauncher so each case takes a second."""

    def setUp(self):
        self.lab = Lab(launcher_factory=FakeLauncher)

    def tearDown(self):
        self.lab.close()

    def started(self, params=None) -> RunSession:
        s = self.lab.session(params or quick("ddp"))
        s.check()
        s.request_all()
        s.start()
        return s

    def finish(self, name: str, run_id: str) -> None:
        rank = ["sim-0", "sim-1"].index(name)
        write_events(self.lab.tmp / f"agent-{name}" / run_id / f"rank{rank}-{name}.jsonl",
                     {"event": "train_end"})
        self.lab.launchers[name].process.exit_code = 0

    def test_mot_may_tu_choi_huy_luot(self):
        self.lab.decisions["sim-1"] = ("deny", "đang dùng máy")
        s = self.lab.session(quick("ddp"))
        s.check()
        decisions = s.request_all()
        self.assertEqual(decisions["sim-1"], ("deny", "đang dùng máy"))
        self.assertFalse(s.approved)
        with self.assertRaises(ProtocolError):
            s.start()
        self.assertEqual(self.lab.launchers["sim-0"].calls, [])
        self.assertIn(("decision", {"machine": "sim-1", "decision": "deny", "reason": "đang dùng máy"}),
                      self.lab.events)
        self.assertEqual(self.lab.cores["sim-0"].state, "IDLE")   # sim-0 had allowed; released

    def test_huy_sau_khi_dong_y(self):
        s = self.lab.session(quick("ddp"))
        s.check()
        s.request_all()
        s.cancel("Người điều phối huỷ")
        self.assertEqual([c.state for c in self.lab.cores.values()], ["IDLE", "IDLE"])

    def test_start_loi_o_rank0_tha_may_con_lai(self):
        s = self.lab.session(quick("ddp"))
        s.check()
        s.request_all()
        self.lab.servers["sim-0"].shutdown()
        self.lab.servers["sim-0"].server_close()
        with self.assertRaises(ProtocolError):
            s.start()
        self.assertEqual(self.lab.cores["sim-1"].state, "IDLE")
        self.assertEqual(self.lab.launchers["sim-1"].calls, [])

    def test_noi_lai_khi_mot_may_xong_mot_may_con_chay(self):
        s = self.started()
        self.finish("sim-0", s.run_id)
        resumed = RunSession.resume(self.lab.clients, self.lab.results, self.lab.quick_results, PROTOTYPE)
        statuses = resumed.poll_once()
        self.assertFalse(resumed.done(statuses))
        self.finish("sim-1", s.run_id)
        self.assertTrue(resumed.done(resumed.poll_once()))
        run_dir = resumed.collect()
        self.assertEqual(len(list(run_dir.glob("rank*.jsonl"))), 2)
        self.assertEqual(outcome(run_dir)["state"], "finished")

    def test_gom_khi_mot_may_mat(self):
        s = self.started()
        self.finish("sim-0", s.run_id)
        self.finish("sim-1", s.run_id)
        s.done(s.poll_once())
        self.lab.servers["sim-1"].shutdown()
        self.lab.servers["sim-1"].server_close()
        run_dir = s.collect()
        record = json.loads((run_dir / "controller.json").read_text("utf-8"))
        self.assertEqual(record["missing"], ["sim-1"])
        self.assertTrue((run_dir / "rank0-sim-0.jsonl").is_file())

    def test_may_khong_tra_loi_khong_dung_luot(self):
        s = self.started()
        self.lab.servers["sim-1"].shutdown()
        self.lab.servers["sim-1"].server_close()
        statuses = s.poll_once()
        self.assertEqual(statuses["sim-1"]["state"], "UNREACHABLE")
        self.assertEqual(statuses["sim-0"]["state"], "RUNNING")
        self.assertFalse(self.lab.launchers["sim-0"].process.terminated)

    def test_may_chet_han_thi_luot_van_ket_thuc(self):
        now = [1000.0]
        s = self.lab.session(quick("ddp"), clock=lambda: now[0])
        s.check()
        s.request_all()
        s.start()
        self.finish("sim-0", s.run_id)
        self.lab.servers["sim-1"].shutdown()
        self.lab.servers["sim-1"].server_close()
        self.assertFalse(s.done(s.poll_once()))           # just unreachable: keep waiting
        now[0] += controller_core.UNREACHABLE_LIMIT_S + 1
        statuses = s.poll_once()
        self.assertEqual(statuses["sim-1"]["state"], "LOST")
        self.assertTrue(s.done(statuses))
        record = json.loads((s.collect() / "controller.json").read_text("utf-8"))
        self.assertEqual(record["missing"], ["sim-1"])
        self.assertEqual(record["outcome"]["state"], "failed")
        self.assertIn("sim-1", record["outcome"]["reason"])

    def test_may_mat_lien_lac_sau_khi_bam_dung_ket_thuc_som_hon(self):
        now = [1000.0]
        s = self.lab.session(quick("ddp"), clock=lambda: now[0])
        s.check()
        s.request_all()
        s.start()
        self.lab.servers["sim-1"].shutdown()
        self.lab.servers["sim-1"].server_close()
        s.stop_all("Người điều phối dừng")
        self.assertFalse(s.done(s.poll_once()))
        now[0] += controller_core.STOPPING_GRACE_S + 1
        self.assertTrue(s.done(s.poll_once()))

    def test_dung_tu_controller(self):
        s = self.started()
        s.stop_all("Người điều phối dừng")
        statuses = s.poll_once()
        self.assertTrue(all(st["state"] == "STOPPED" for st in statuses.values()))
        self.assertEqual(outcome(s.collect())["state"], "stopped")

    def test_diem_gap_la_ip_controller_dung_khong_phai_ip_trong_file_may(self):
        # DHCP may give rank 0 a different address than its machine file says. The controller
        # reached the agent at the address typed in, so that is where the others must meet.
        from dataclasses import replace
        for name in ("sim-0", "sim-1"):
            core = self.lab.cores[name]
            core.machine = replace(core.machine, address="10.9.9.9")
        s = self.lab.session(quick("ddp"))
        s.check()
        self.assertEqual(s.workers[0], "sim-0")
        self.assertEqual(s.master_addr, "127.0.0.1")
        s.request_all()
        s.start()
        argv = self.lab.launchers["sim-1"].calls[0][0]
        self.assertEqual(argv[argv.index("--master-addr") + 1], "127.0.0.1")

    def test_agent_khong_co_address_van_chay(self):
        from dataclasses import replace
        core = self.lab.cores["sim-1"]
        core.machine = replace(core.machine, address=None)
        self.lab.session(quick("ddp")).check()

    def test_khong_du_hai_may(self):
        s = RunSession(self.lab.clients[:1], validate_params(quick("ddp")), self.lab.results,
                       self.lab.quick_results, PROTOTYPE)
        with self.assertRaises(ProtocolError):
            s.check()

    def test_may_ban_khong_duoc_chon(self):
        self.started()
        s = self.lab.session(quick("localsgd"))
        with self.assertRaises(ProtocolError) as caught:
            s.check()
        self.assertIn("bận", str(caught.exception))

    def test_luot_that_can_cuda(self):
        s = self.lab.session({"mode": "ddp"})
        with self.assertRaises(ProtocolError) as caught:
            s.check()
        self.assertIn("CUDA", str(caught.exception))

    def test_analyze_luot_that_dung_lenh(self):
        s = self.lab.session({"mode": "ddp"})
        s.run_id = "lab-ddp-resnet20-20261116-100000"
        with mock.patch.object(controller_core.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            report = s.analyze()
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:], ["analyze/analyze.py", "lab-", "--results", str(self.lab.results),
                                    "--out", str(self.lab.results / "report-lab"),
                                    "--acc", "0.5", "0.6", "0.7", "0.8", "--at-minutes", "10", "30"])
        self.assertEqual(run.call_args.kwargs["cwd"], PROTOTYPE)
        self.assertEqual(report, self.lab.results / "report-lab")

    def test_luot_chay_thu_phan_tich_rieng(self):
        # Quick runs get their own report, kept apart from the lab report used in the paper.
        s = self.started()
        with mock.patch.object(controller_core.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            report = s.analyze()
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:6], ["analyze/analyze.py", "thu-", "--results", str(self.lab.quick_results),
                                     "--out"])
        self.assertEqual(argv[6], str(self.lab.quick_results / "report-thu"))
        self.assertEqual(report, self.lab.quick_results / "report-thu")


class LabFileTest(unittest.TestCase):
    def test_doc_lab_yaml(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "lab.yaml"
            path.write_text("token: abc\nagents:\n  - {name: doan-tuan, address: 192.168.0.103}\n"
                            "  - {name: trinh-duc, address: 192.168.0.102, port: 8800}\n", encoding="utf-8")
            agents, token = load_lab(path)
            self.assertEqual(token, "abc")
            self.assertEqual(agents, [AgentRef("doan-tuan", "192.168.0.103", 8765),
                                      AgentRef("trinh-duc", "192.168.0.102", 8800)])
            with self.assertRaises(ProtocolError) as caught:
                load_lab(tmp / "khong-co.yaml")
            self.assertIn("lab.yaml.example", str(caught.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_file_mau_doc_duoc(self):
        agents, token = load_lab(PROTOTYPE / "configs" / "lab.yaml.example")
        self.assertEqual([a.name for a in agents], ["doan-tuan", "trinh-duc"])


class RunResultsDirTest(unittest.TestCase):
    def test_lenh_run_ghi_vao_results_dir(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            code = cli.main(["run", "-m", "sim-0", "-e", "smoke_single", "--run-id", "kiem-tra-results-dir",
                             "--set", "max_epochs=1", "--results-dir", str(tmp)])
            self.assertEqual(code, 0)
            self.assertTrue((tmp / "kiem-tra-results-dir" / "rank0-sim-0.jsonl").is_file())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
