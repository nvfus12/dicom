"""Tests for the lab agent's HTTP layer, over real sockets on localhost."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # shared fakes in test_control_agent

from gpubench.control.agent_core import AgentCore  # noqa: E402
from gpubench.control.agent_server import make_server  # noqa: E402
from gpubench.control.protocol import TOKEN_HEADER  # noqa: E402
from test_control_agent import PARAMS, PROTOTYPE, RUN, WORKERS, FakeLauncher, make_configs, write_events  # noqa: E402

TOKEN = "test-token"


class AgentServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.launcher = FakeLauncher()
        self.core = AgentCore("sim-1", lambda payload: ("allow", ""), self.launcher,
                              configs_dir=make_configs(self.tmp), results_dir=self.tmp / "results",
                              project_dir=PROTOTYPE)
        self.server = make_server(self.core, "127.0.0.1", 0, TOKEN)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method: str, path: str, body=None, token: str = TOKEN, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={TOKEN_HEADER: token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read()
            if resp.headers.get("Content-Type") == "application/zip":
                return resp.headers.get("Content-Type"), payload
            return json.loads(payload)

    def error(self, *args, **kwargs) -> tuple[int, dict]:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call(*args, **kwargs)
        return caught.exception.code, json.loads(caught.exception.read())

    def request_body(self):
        return {"run_id": RUN, "controller": "C", "params": PARAMS, "workers": WORKERS}

    def start_body(self):
        return {**self.request_body(), "master_addr": "127.0.0.1"}

    def test_sai_token_401(self):
        code, body = self.error("GET", "/info", token="sai")
        self.assertEqual(code, 401)
        self.assertIn("token", body["error"].lower())

    def test_agent_thu_hai_cung_cong_bi_tu_choi(self):
        # On Windows SO_REUSEADDR lets a second server bind the same port silently, so requests
        # would be split between two agents. The second agent must fail to start instead.
        port = self.server.server_address[1]
        with self.assertRaises(OSError):
            second = make_server(self.core, "127.0.0.1", port, TOKEN)
            second.server_close()

    def test_post_sai_token_van_nhan_401_khong_bi_cat_ket_noi(self):
        # Answering before reading the body makes Windows reset the connection; the controller
        # would then report "cannot connect" instead of "wrong token".
        big = {"run_id": RUN, "padding": "x" * 2_000_000}
        for _ in range(3):
            self.assertEqual(self.error("POST", "/request", big, token="sai")[0], 401)
            self.assertEqual(self.error("POST", "/khong-co", big)[0], 404)

    def test_info(self):
        self.assertEqual(self.call("GET", "/info")["name"], "sim-1")

    def test_duong_dan_la_404(self):
        self.assertEqual(self.error("GET", "/khong-co")[0], 404)
        self.assertEqual(self.error("POST", "/khong-co", {})[0], 404)

    def test_json_hong_400(self):
        self.assertEqual(self.error("POST", "/request", raw=b"khong-phai-json")[0], 400)

    def test_thieu_truong_400(self):
        self.assertEqual(self.error("POST", "/request", {"run_id": RUN})[0], 400)

    def test_vong_request_start_status_stop(self):
        self.assertEqual(self.call("POST", "/request", self.request_body())["decision"], "allow")
        self.assertEqual(self.call("POST", "/start", self.start_body())["state"], "RUNNING")
        self.assertEqual(self.call("GET", f"/status?run_id={RUN}")["state"], "RUNNING")
        stopped = self.call("POST", "/stop", {"run_id": RUN, "reason": "thử"})
        self.assertEqual((stopped["state"], stopped["stopped_by"]), ("STOPPED", "controller"))

    def test_start_khong_xin_phep_409(self):
        code, body = self.error("POST", "/start", self.start_body())
        self.assertEqual(code, 409)
        self.assertIn("chưa được chủ máy cho phép", body["error"])

    def test_status_thieu_run_id_400(self):
        self.assertEqual(self.error("GET", "/status")[0], 400)

    def test_tai_results_la_zip(self):
        self.call("POST", "/request", self.request_body())
        self.call("POST", "/start", self.start_body())
        write_events(self.tmp / "results" / RUN / "rank1-sim-1.jsonl", {"event": "train_end"})
        self.launcher.process.exit_code = 0
        content_type, blob = self.call("GET", f"/results?run_id={RUN}")
        self.assertEqual(content_type, "application/zip")
        self.assertIn("rank1-sim-1.jsonl", zipfile.ZipFile(io.BytesIO(blob)).namelist())


if __name__ == "__main__":
    unittest.main()
