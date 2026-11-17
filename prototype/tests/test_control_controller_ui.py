"""Tests for the lab controller window, with fake agents and a fake run session."""
from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpubench import cli  # noqa: E402
from gpubench.control import controller_ui  # noqa: E402
from gpubench.control.protocol import RunParams  # noqa: E402

PROTOTYPE = Path(__file__).resolve().parents[1]


def pump_until(root, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return False


class FakeClient:
    def __init__(self, ref, token):
        self.ref, self.name = ref, ref.name

    def info(self):
        vram = {"doan-tuan": 6141, "trinh-duc": 4096}[self.name]
        return {"name": self.name, "address": self.ref.address, "state": "IDLE", "run_id": None,
                "gpu": {"name": "GPU " + self.name, "vram_mb": vram, "cuda_available": True}}


class FakeSession:
    instances: list["FakeSession"] = []
    approve = True
    resume_with: "FakeSession | None" = None

    def __init__(self, agents, params, results_root, quick_root, prototype_dir, on_event=None):
        self.agents, self.params, self.results_root = agents, params, results_root
        self.on_event = on_event or (lambda kind, data: None)
        self.run_id = "lab-localsgd-resnet20-20261116-100000"
        self.workers = ["doan-tuan", "trinh-duc"]
        self.approved = False
        self.calls: list[str] = []
        self.polls = 0
        FakeSession.instances.append(self)

    def check(self):
        self.calls.append("check")

    def request_all(self):
        self.calls.append("request_all")
        decision = ("allow", "") if FakeSession.approve else ("deny", "bận")
        for name in self.workers:
            self.on_event("decision", {"machine": name, "decision": decision[0], "reason": decision[1]})
        self.approved = FakeSession.approve
        return {name: decision for name in self.workers}

    def start(self):
        self.calls.append("start")

    def poll_once(self):
        self.polls += 1
        state = "FINISHED" if self.polls >= 2 else "RUNNING"
        return {name: {"state": state, "epoch": self.polls, "total_epochs": 30,
                       "points": [[60.0, 0.4], [120.0, 0.55]]} for name in self.workers}

    def done(self, statuses):
        return all(s["state"] == "FINISHED" for s in statuses.values())

    def stop_all(self, reason):
        self.calls.append("stop_all")

    def cancel(self, reason):
        self.calls.append("cancel")

    def collect(self):
        self.calls.append("collect")
        return self.results_root / self.run_id

    def analyze(self):
        self.calls.append("analyze")
        report = self.results_root / "report-lab"
        report.mkdir(parents=True, exist_ok=True)
        (report / "runs.csv").write_text("run_id,mode,best_acc\nlab-a,ddp,0.81\nlab-b,localsgd,0.79\n",
                                         encoding="utf-8")
        return report

    @classmethod
    def resume(cls, agents, results_root, quick_root, prototype_dir, on_event=None):
        return cls.resume_with


class ControllerWindowTest(unittest.TestCase):
    def setUp(self):
        FakeSession.instances, FakeSession.approve, FakeSession.resume_with = [], True, None
        self.tmp = Path(tempfile.mkdtemp())
        self.lab = self.tmp / "lab.yaml"
        self.lab.write_text("token: t\nagents:\n  - {name: doan-tuan, address: 192.168.0.103}\n"
                            "  - {name: trinh-duc, address: 192.168.0.102}\n", encoding="utf-8")
        self.win = self.make_window()

    def make_window(self):
        win = controller_ui.ControllerWindow(self.lab, PROTOTYPE, session_factory=FakeSession,
                                             client_factory=FakeClient,
                                             results_root=self.tmp / "results",
                                             quick_root=self.tmp / "results-thu", poll_s=0.05)
        win.start_background()
        return win

    def tearDown(self):
        try:
            self.win.quit()
        except tk.TclError:
            pass
        gc.collect()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def session(self) -> FakeSession:
        self.assertTrue(pump_until(self.win.root, lambda: bool(FakeSession.instances)))
        return FakeSession.instances[-1]

    def test_bang_worker_hien_gpu_va_rank(self):
        self.assertTrue(pump_until(self.win.root, lambda: len(self.win.worker_tree.get_children()) == 2))
        self.assertTrue(pump_until(self.win.root, lambda: "GPU doan-tuan" in str(
            self.win.worker_tree.item("doan-tuan", "values"))))
        self.assertEqual(self.win.worker_tree.set("doan-tuan", "rank"), "0")
        self.assertEqual(self.win.worker_tree.set("trinh-duc", "rank"), "1")

    def address_of(self, name):
        return next(c for c in self.win.clients if c.name == name).ref.address

    def test_sua_ip_cap_nhat_client_va_luu_lab_yaml(self):
        self.assertTrue(self.win.set_worker_address("trinh-duc", " 192.168.0.101 "))
        self.assertEqual(self.address_of("trinh-duc"), "192.168.0.101")
        self.assertEqual(self.address_of("doan-tuan"), "192.168.0.103")
        agents, token = controller_ui.load_lab(self.lab)
        self.assertEqual(token, "t")
        self.assertEqual([(a.name, a.address) for a in agents],
                         [("doan-tuan", "192.168.0.103"), ("trinh-duc", "192.168.0.101")])
        self.win.root.update()
        self.assertEqual(self.win.worker_tree.set("trinh-duc", "address"), "192.168.0.101:8765")

    def test_ip_sai_bi_tu_choi(self):
        for bad in ("192.168.0.999", "abc", "", "192.168.0"):
            with mock.patch.object(controller_ui.messagebox, "showerror") as showerror:
                self.assertFalse(self.win.set_worker_address("trinh-duc", bad))
            showerror.assert_called_once()
        self.assertEqual(self.address_of("trinh-duc"), "192.168.0.102")

    def test_khong_sua_ip_khi_dang_co_luot(self):
        self.win._busy = True
        with mock.patch.object(controller_ui.messagebox, "showerror") as showerror:
            self.assertFalse(self.win.set_worker_address("trinh-duc", "192.168.0.101"))
        showerror.assert_called_once()
        self.assertEqual(self.address_of("trinh-duc"), "192.168.0.102")

    def test_nut_sua_ip_hoi_ip_cho_may_dang_chon(self):
        self.win.worker_tree.selection_set("doan-tuan")
        with mock.patch.object(controller_ui.simpledialog, "askstring", return_value="192.168.0.110") as ask:
            self.win.edit_ip_button.invoke()
        self.assertIn("192.168.0.103", str(ask.call_args))       # current IP offered for editing
        self.assertEqual(self.address_of("doan-tuan"), "192.168.0.110")

    def test_nut_sua_ip_huy_thi_giu_nguyen(self):
        self.win.worker_tree.selection_set("doan-tuan")
        with mock.patch.object(controller_ui.simpledialog, "askstring", return_value=None):
            self.win.edit_ip_button.invoke()
        self.assertEqual(self.address_of("doan-tuan"), "192.168.0.103")

    def test_h_chi_bat_khi_local_sgd(self):
        self.win.mode_var.set("ddp")
        self.win.root.update()
        self.assertEqual(str(self.win.h_entry["state"]), "disabled")
        self.win.mode_var.set("localsgd")
        self.win.root.update()
        self.assertEqual(str(self.win.h_entry["state"]), "normal")

    def test_chay_thu_khoa_dataset_model(self):
        self.win.quick_var.set(True)
        self.win.root.update()
        self.assertEqual(str(self.win.dataset_box["state"]), "disabled")
        self.assertEqual(self.win.model_box.get(), "Tiny CNN")
        params = self.win.collect_params()
        self.assertEqual((params.quick, params.dataset, params.model), (True, "synthetic", "tiny_cnn"))
        self.win.quick_var.set(False)
        self.win.root.update()
        self.assertEqual(self.win.collect_params().model, "resnet20")

    def test_params_gui_di_dung_form(self):
        self.win.mode_var.set("localsgd")
        self.win.h_var.set("128")
        self.win.minutes_var.set("20")
        self.win.send_button.invoke()
        s = self.session()
        self.assertEqual(s.params, RunParams(mode="localsgd", local_steps=128, max_train_minutes=20))
        self.assertTrue(pump_until(self.win.root, lambda: "request_all" in s.calls))

    def test_sai_tham_so_hien_loi(self):
        self.win.mode_var.set("localsgd")
        self.win.h_var.set("0")
        with mock.patch.object(controller_ui.messagebox, "showerror") as showerror:
            self.win.send_button.invoke()
        showerror.assert_called_once()
        self.assertEqual(FakeSession.instances, [])

    def test_o_so_khong_hop_le_hien_loi(self):
        self.win.minutes_var.set("ba mươi")
        with mock.patch.object(controller_ui.messagebox, "showerror") as showerror:
            self.win.send_button.invoke()
        showerror.assert_called_once()

    def test_start_chi_bat_khi_moi_may_dong_y(self):
        self.assertEqual(str(self.win.start_button["state"]), "disabled")
        FakeSession.approve = False
        self.win.send_button.invoke()
        s = self.session()
        self.assertTrue(pump_until(self.win.root, lambda: "request_all" in s.calls))
        self.win.root.update()
        self.assertEqual(str(self.win.start_button["state"]), "disabled")
        FakeSession.approve = True
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.start_button["state"]) == "normal"))

    def test_gui_lai_truoc_start_huy_phien_cu(self):
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.start_button["state"]) == "normal"))
        first = FakeSession.instances[-1]
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: "cancel" in first.calls))

    def test_dong_cua_so_truoc_start_huy_phien(self):
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.start_button["state"]) == "normal"))
        session = FakeSession.instances[-1]
        self.win.quit()
        self.assertIn("cancel", session.calls)

    def test_ket_thuc_luot_nap_bang_ket_qua(self):
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.start_button["state"]) == "normal"))
        self.assertEqual(str(self.win.stop_button["state"]), "disabled")
        self.win.start_button.invoke()
        s = self.session()
        self.assertTrue(pump_until(self.win.root, lambda: "analyze" in s.calls, timeout=15))
        self.assertTrue(pump_until(self.win.root, lambda: len(self.win.results_tree.get_children()) == 2))
        self.assertLess(s.calls.index("collect"), s.calls.index("analyze"))
        self.assertEqual(str(self.win.stop_button["state"]), "disabled")
        self.assertEqual(str(self.win.send_button["state"]), "normal")

    def test_nut_dung_goi_stop_all(self):
        self.win.send_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.start_button["state"]) == "normal"))
        s = self.session()
        s.poll_once = lambda: {n: {"state": "RUNNING", "points": []} for n in s.workers}
        self.win.start_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: str(self.win.stop_button["state"]) == "normal"))
        with mock.patch.object(controller_ui.messagebox, "askyesno", return_value=True):
            self.win.stop_button.invoke()
        self.assertTrue(pump_until(self.win.root, lambda: "stop_all" in s.calls))

    def test_noi_lai_luot_khi_mo(self):
        self.win.quit()
        resumed = FakeSession([], RunParams(mode="ddp"), self.tmp / "results", self.tmp / "results-thu",
                              PROTOTYPE)
        FakeSession.instances.clear()
        FakeSession.resume_with = resumed
        self.win = self.make_window()
        self.assertTrue(pump_until(self.win.root, lambda: "Nối lại lượt" in self.win.log.get("1.0", "end")))
        self.assertTrue(pump_until(self.win.root, lambda: "collect" in resumed.calls, timeout=15))


class ControllerCliTest(unittest.TestCase):
    def test_cli_controller_co_trong_help(self):
        with self.assertRaises(SystemExit) as caught, mock.patch("sys.stdout"):
            cli.main(["controller", "--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
