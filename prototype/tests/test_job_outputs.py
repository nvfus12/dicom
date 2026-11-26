"""Machine-readable results of the check commands, which the controller collects and reads."""
from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROTOTYPE = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def free_port() -> int:
    for _ in range(200):
        port = random.randrange(30000, 40000)
        try:
            for candidate in (port, port + 1):
                with socket.socket() as sock:
                    sock.bind(("0.0.0.0", candidate))
            return port
        except OSError:
            continue
    raise RuntimeError("không có cổng trống")


def gpubench(*args, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "gpubench", *args], cwd=PROTOTYPE, env=ENV,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


class JsonOutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_glootest_tu_thu_ghi_json(self):
        out = self.tmp / "self.json"
        done = gpubench("glootest", "-m", "sim-0", "--self", "--toward", "127.0.0.1",
                        "--port-base", str(free_port()), "--wait", "60", "--json-out", str(out))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        data = json.loads(out.read_text("utf-8"))
        self.assertEqual((data["machine"], data["self_test"], data["master"]), ("sim-0", True, "127.0.0.1"))
        self.assertEqual(data["variants"]["theo-ip"]["verdict"], "ĐẠT")
        self.assertEqual(data["variants"]["theo-ten-card"]["verdict"], "BỎ QUA")

    def test_porttest_pair_hai_chieu(self):
        port = free_port()
        outs = [self.tmp / "r0.json", self.tmp / "r1.json"]
        procs = [subprocess.Popen([sys.executable, "-m", "gpubench", "porttest", "pair", "-m", name,
                                   "--rank", str(rank), "--peer", "127.0.0.1", "--port", str(port),
                                   "--window", "8", "--json-out", str(outs[rank])],
                                  cwd=PROTOTYPE, env=ENV, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                 for rank, name in enumerate(("sim-0", "sim-1"))]
        texts = [p.communicate(timeout=120)[0] for p in procs]
        for rank, path in enumerate(outs):
            data = json.loads(path.read_text("utf-8"))
            self.assertEqual(data["rank"], rank)
            self.assertTrue(data["listen"]["ok"], texts[rank])
            self.assertTrue(data["probe"]["ok"], texts[rank])
            self.assertIn("ĐẠT", data["probe"]["message"])

    def test_porttest_pair_may_kia_khong_chay(self):
        out = self.tmp / "alone.json"
        done = gpubench("porttest", "pair", "-m", "sim-1", "--rank", "1", "--peer", "127.0.0.1",
                        "--port", str(free_port()), "--window", "3", "--json-out", str(out))
        data = json.loads(out.read_text("utf-8"))
        self.assertFalse(data["probe"]["ok"])
        self.assertIn("từ chối", data["probe"]["message"])
        self.assertFalse(data["listen"]["ok"])
        self.assertNotEqual(done.returncode, 0)

    def test_profile_ghi_json(self):
        out = self.tmp / "profile.json"
        done = gpubench("profile", "-m", "sim-0", "--models", "tiny_cnn", "--batches", "16", "32",
                        "--steps", "2", "--json-out", str(out))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        data = json.loads(out.read_text("utf-8"))
        self.assertEqual(data["machine"], "sim-0")
        rows = data["models"]["tiny_cnn"]
        self.assertEqual([r["batch"] for r in rows], [16, 32])
        self.assertGreater(rows[0]["samples_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
