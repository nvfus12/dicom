"""Tests for the lab agent window. Drives buttons with .invoke() and pumps the Tk loop.

Windows flash briefly while this runs; it needs a display.
"""
from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # shared fakes in test_control_agent

from gpubench import cli  # noqa: E402
from gpubench.control import agent_ui  # noqa: E402
from gpubench.control.agent_core import AgentCore  # noqa: E402
from test_control_agent import PROTOTYPE, RUN, WORKERS, FakeLauncher, make_configs  # noqa: E402

REAL_PARAMS = {"mode": "localsgd", "local_steps": 256}


def find_widgets(parent, cls) -> list:
    found = []
    for child in parent.winfo_children():
        if isinstance(child, cls):
            found.append(child)
        found.extend(find_widgets(child, cls))
    return found


def texts(window) -> str:
    out = []
    for widget in find_widgets(window, tk.Widget):
        try:
            out.append(str(widget.cget("text")))
        except tk.TclError:
            continue
    return " | ".join(out)


def button(window, text: str):
    for widget in find_widgets(window, tk.Widget):
        try:
            if widget.winfo_class() == "TButton" and widget.cget("text") == text:
                return widget
        except tk.TclError:
            continue
    raise AssertionError(f"không tìm thấy nút '{text}'")


def pump_until(root, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return False


class AgentWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.launcher = FakeLauncher()
        configs = make_configs(self.tmp)

        def factory(approver):
            return AgentCore("sim-1", approver, self.launcher, configs_dir=configs,
                             results_dir=self.tmp / "results", project_dir=PROTOTYPE)
        self.app = agent_ui.AgentWindow(factory, "127.0.0.1", 0, "t")
        self.app.serve()

    def tearDown(self):
        try:
            self.app.quit()
        except tk.TclError:
            pass
        gc.collect()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ask(self, params=REAL_PARAMS) -> dict:
        """Sends a request from another thread; returns the dict its reply will land in."""
        reply: dict = {}
        payload = {"run_id": RUN, "controller": "Máy điều phối", "params": params, "workers": WORKERS}
        threading.Thread(target=lambda: reply.update(self.app.core.request(payload)), daemon=True).start()
        self.assertTrue(pump_until(self.app.root, self.dialog), "hộp thoại không hiện ra")
        return reply

    def dialog(self):
        windows = find_widgets(self.app.root, tk.Toplevel)
        return windows[0] if windows else None

    def answer(self, text: str, reply: dict) -> None:
        button(self.dialog(), text).invoke()
        self.assertTrue(pump_until(self.app.root, lambda: bool(reply)), "không có câu trả lời")

    def run_training(self) -> None:
        reply = self.ask()
        self.answer("Cho phép", reply)
        self.app.core.start({"run_id": RUN, "params": REAL_PARAMS, "workers": WORKERS,
                             "master_addr": "127.0.0.1"})
        self.assertTrue(pump_until(self.app.root,
                                   lambda: str(self.app.stop_button["state"]) == "normal"))

    def test_hop_thoai_hien_noi_dung_luot(self):
        self.ask()
        shown = texts(self.dialog())
        for expected in ("Máy điều phối", "Local SGD", "256 bước", "30 phút", "sim-0"):
            self.assertIn(expected, shown)
        button(self.dialog(), "Từ chối").invoke()

    def test_cho_phep_tra_allow(self):
        reply = self.ask()
        self.answer("Cho phép", reply)
        self.assertEqual(reply["decision"], "allow")
        self.assertEqual(self.app.core.state, "APPROVED")

    def test_tu_choi_tra_deny(self):
        reply = self.ask()
        self.answer("Từ chối", reply)
        self.assertEqual(reply["decision"], "deny")

    def test_dong_hop_thoai_la_tu_choi(self):
        reply = self.ask()
        dialog = self.dialog()
        dialog.tk.call(dialog.protocol("WM_DELETE_WINDOW"))
        self.assertTrue(pump_until(self.app.root, lambda: bool(reply)))
        self.assertEqual(reply["decision"], "deny")

    def test_nut_dung_chi_bat_khi_dang_chay(self):
        self.assertEqual(str(self.app.stop_button["state"]), "disabled")
        self.run_training()
        self.assertIn("Đang được dùng để huấn luyện", self.app.status_label.cget("text"))

    def test_nut_dung_goi_stop_owner(self):
        self.run_training()
        with mock.patch.object(agent_ui.messagebox, "askyesno", return_value=True):
            self.app.stop_button.invoke()
        status = self.app.core.status(RUN)
        self.assertEqual((status["state"], status["stopped_by"]), ("STOPPED", "owner"))
        self.assertTrue(self.launcher.process.terminated)

    def test_nut_dung_huy_xac_nhan_thi_van_chay(self):
        self.run_training()
        with mock.patch.object(agent_ui.messagebox, "askyesno", return_value=False):
            self.app.stop_button.invoke()
        self.assertEqual(self.app.core.state, "RUNNING")

    def test_tat_agent_khi_dang_chay_ket_thuc_tien_trinh(self):
        self.run_training()
        self.app.quit()
        self.assertTrue(self.launcher.process.terminated)

    def test_tat_agent_khi_dang_hoi_la_tu_choi(self):
        reply = self.ask()
        self.app.quit()
        deadline = time.monotonic() + 5
        while not reply and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(reply["decision"], "deny")


class AgentCliTest(unittest.TestCase):
    def test_cli_agent_co_trong_help(self):
        with self.assertRaises(SystemExit) as caught, mock.patch("sys.stdout"):
            cli.main(["agent", "--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
