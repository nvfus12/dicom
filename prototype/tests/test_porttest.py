"""`gpubench porttest`: can this machine's Python accept a TCP connection from the other one."""
from __future__ import annotations

import io
import socket
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpubench.app import porttest  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def captured(fn, *args, **kwargs) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = fn(*args, **kwargs)
    return code, out.getvalue()


class PortTest(unittest.TestCase):
    def test_nghe_va_thu_thanh_cong(self):
        port = free_port()
        listener_out = io.StringIO()

        def listen():
            with redirect_stdout(listener_out):
                porttest.listen(port, seconds=5, host="127.0.0.1")
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        time.sleep(0.5)
        code, out = captured(porttest.probe, "127.0.0.1", port, timeout=5)
        thread.join(10)
        self.assertEqual(code, 0)
        self.assertIn("ĐẠT", out)
        # redirect_stdout is process-wide, so per-connection lines may land in the probe's buffer;
        # the listener's own verdict is printed after the probe finished.
        self.assertIn("ĐẠT: đã nhận 1 kết nối", listener_out.getvalue())

    def test_cong_khong_ai_nghe_la_tu_choi(self):
        code, out = captured(porttest.probe, "127.0.0.1", free_port(), timeout=5)
        self.assertEqual(code, 1)
        self.assertIn("LỖI", out)
        self.assertIn("từ chối", out)

    def test_khong_phan_hoi_la_bi_chan(self):
        code, out = captured(porttest.probe, "10.255.255.1", 45000, timeout=1)
        self.assertEqual(code, 1)
        self.assertIn("LỖI", out)
        self.assertIn("bị chặn", out)

    def test_nghe_het_gio_khong_ai_goi(self):
        code, out = captured(porttest.listen, free_port(), seconds=1, host="127.0.0.1")
        self.assertEqual(code, 1)
        self.assertIn("Không nhận được kết nối nào", out)


if __name__ == "__main__":
    unittest.main()
