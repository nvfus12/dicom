"""Kiem thu demo xin tai nguyen, chay duoc ma khong can nguoi bam nut.

    python -m unittest discover -s demo -v

Cac test dieu khien giao dien bang cach goi .invoke() tren nut, va bom vong
lap su kien cua Tk bang root.update(). Cua so se nhap nhay vai giay, do la
binh thuong. Can mot man hinh (khong chay duoc tren may chi co dong lenh).
"""
from __future__ import annotations

import gc
import json
import socket
import sys
import threading
import time
import tkinter as tk
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import controller  # noqa: E402

TOKEN = "test-token"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_widgets(parent, cls) -> list:
    """Moi widget con thuoc lop cls, tim de quy."""
    found = []
    for child in parent.winfo_children():
        if isinstance(child, cls):
            found.append(child)
        found.extend(find_widgets(child, cls))
    return found


def button_titled(window, text: str):
    for button in find_widgets(window, tk.Widget):
        try:
            if button.cget("text") == text and button.winfo_class() == "TButton":
                return button
        except tk.TclError:
            continue
    raise AssertionError(f"không tìm thấy nút '{text}'")


def close(app) -> None:
    """Dong agent va thu gom rac ngay trong luong chinh.

    Moi test tao mot tk.Tk() rieng. Neu de bo thu gom rac don chung o luong
    khac, Tcl bao 'async handler deleted by the wrong thread' va tien trinh
    chet giua chung. Goi gc.collect() o day giu viec do lai trong luong chinh.
    """
    app.quit()
    gc.collect()


def pump(root: tk.Tk, seconds: float = 0.5) -> None:
    """Chay vong lap su kien Tk trong mot khoang thoi gian."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)


def pump_until(root: tk.Tk, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return False


class AgentProtocolTest(unittest.TestCase):
    """Kiem tra tang HTTP: token, duong dan, JSON hong."""

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.app = agent.AgentApp("may-test", "127.0.0.1", cls.port, TOKEN)
        cls.app.serve()
        cls.url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        close(cls.app)

    def test_ping_tra_ve_ten_may(self):
        reply = controller.get_json(f"{self.url}/ping", TOKEN, 5)
        self.assertEqual(reply["machine"], "may-test")
        self.assertEqual(reply["status"], "ok")

    def test_token_sai_bi_tu_choi(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            controller.get_json(f"{self.url}/ping", "token-sai", 5)
        self.assertEqual(caught.exception.code, 401)

    def test_duong_dan_la_tra_ve_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            controller.get_json(f"{self.url}/khong-co", TOKEN, 5)
        self.assertEqual(caught.exception.code, 404)

    def test_than_yeu_cau_khong_phai_json(self):
        import urllib.request
        req = urllib.request.Request(f"{self.url}/request", data=b"khong-phai-json",
                                     method="POST", headers={"X-Demo-Token": TOKEN})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 400)


class DialogTest(unittest.TestCase):
    """Kiem tra hop thoai: noi dung hien ra va hai nut tra ve dung quyet dinh."""

    def setUp(self):
        self.app = agent.AgentApp("may-test", "127.0.0.1", free_port(), TOKEN)

    def tearDown(self):
        close(self.app)

    def _ask(self) -> agent.PendingRequest:
        request = agent.PendingRequest(
            payload={"controller": "Máy điều phối", "job": "Huấn luyện ResNet-20",
                     "note": "ghi chú thử", "resources": {"gpu_vram_gb": 4, "minutes": 30}},
            peer="127.0.0.1")
        self.app.ask_owner(request)
        pump(self.app.root, 0.3)
        return request

    def test_nut_cho_phep(self):
        request = self._ask()
        dialog = find_widgets(self.app.root, tk.Toplevel)[0]
        button_titled(dialog, "Cho phép").invoke()
        self.assertTrue(request.answered.is_set())
        self.assertEqual(request.decision, "allow")

    def test_nut_tu_choi(self):
        request = self._ask()
        dialog = find_widgets(self.app.root, tk.Toplevel)[0]
        button_titled(dialog, "Từ chối").invoke()
        self.assertEqual(request.decision, "deny")

    def test_dong_cua_so_la_tu_choi(self):
        request = self._ask()
        dialog = find_widgets(self.app.root, tk.Toplevel)[0]
        # protocol() tra ve ten lenh Tcl, khong phai ham Python; goi qua Tk.
        dialog.tk.call(dialog.protocol("WM_DELETE_WINDOW"))
        self.assertEqual(request.decision, "deny")

    def test_hop_thoai_hien_thong_tin_yeu_cau(self):
        self._ask()
        dialog = find_widgets(self.app.root, tk.Toplevel)[0]
        texts = []
        for widget in find_widgets(dialog, tk.Widget):
            try:
                texts.append(str(widget.cget("text")))
            except tk.TclError:
                continue
        blob = " | ".join(texts)
        self.assertIn("Máy điều phối", blob)
        self.assertIn("Huấn luyện ResNet-20", blob)
        self.assertIn("4 GB", blob)          # VRAM duoc dinh dang
        self.assertIn("30 phút", blob)


class EndToEndTest(unittest.TestCase):
    """Tu luc controller gui yeu cau den luc nhan duoc cau tra loi."""

    def setUp(self):
        self.port = free_port()
        self.app = agent.AgentApp("may-test", "127.0.0.1", self.port, TOKEN)
        self.app.serve()
        self.app._poll_timer = self.app.root.after(100, self.app.poll)

    def tearDown(self):
        close(self.app)

    def _run_exchange(self, button: str) -> dict:
        reply: dict = {}

        def client():
            reply.update(controller.post_json(
                f"http://127.0.0.1:{self.port}/request", TOKEN,
                {"request_id": "r1", "controller": "Máy điều phối", "job": "thử",
                 "resources": {"gpu_vram_gb": 4, "minutes": 30}}, 30))

        threading.Thread(target=client, daemon=True).start()
        appeared = pump_until(self.app.root,
                              lambda: bool(find_widgets(self.app.root, tk.Toplevel)))
        self.assertTrue(appeared, "hộp thoại không hiện ra")
        dialog = find_widgets(self.app.root, tk.Toplevel)[0]
        button_titled(dialog, button).invoke()
        self.assertTrue(pump_until(self.app.root, lambda: bool(reply)),
                        "controller không nhận được câu trả lời")
        return reply

    def test_cho_phep_di_het_vong(self):
        reply = self._run_exchange("Cho phép")
        self.assertEqual(reply["decision"], "allow")
        self.assertEqual(reply["machine"], "may-test")
        self.assertEqual(reply["request_id"], "r1")

    def test_tu_choi_di_het_vong(self):
        reply = self._run_exchange("Từ chối")
        self.assertEqual(reply["decision"], "deny")

    def test_quyet_dinh_duoc_ghi_vao_nhat_ky(self):
        before = agent.LOG_PATH.read_text(encoding="utf-8").count("\n") \
            if agent.LOG_PATH.exists() else 0
        self._run_exchange("Cho phép")
        lines = agent.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), before + 1)
        self.assertEqual(json.loads(lines[-1])["decision"], "allow")


class ConfigTest(unittest.TestCase):
    def test_machines_json_doc_duoc(self):
        config = controller.load_config()
        self.assertIn("token", config)
        self.assertTrue(config["machines"])
        for machine in config["machines"]:
            self.assertIn("name", machine)
            self.assertIn("address", machine)

    def test_thong_bao_loi_de_hieu(self):
        self.assertIn("Token", controller.friendly_error(
            urllib.error.HTTPError("u", 401, "x", {}, None)))
        self.assertIn("Không kết nối được", controller.friendly_error(
            urllib.error.URLError("máy không trả lời")))


if __name__ == "__main__":
    unittest.main()
