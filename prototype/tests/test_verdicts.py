"""Turning the agents' check results into a verdict and the next step of the diagnosis ladder."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpubench.control.verdicts import summarize  # noqa: E402

W = ["doan-tuan", "trinh-duc"]


def gloo(ten_card: str, ip: str, detail: str = "") -> dict:
    return {"variants": {"theo-ten-card": {"verdict": ten_card, "detail": detail},
                         "theo-ip": {"verdict": ip, "detail": detail}}}


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def put(self, machine: str, data: dict) -> None:
        (self.dir / f"diag-{machine}.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")

    def text(self, verdict: dict) -> str:
        return "\n".join(verdict["lines"]) + "\n" + verdict["hint"]

    def test_tu_thu_ca_hai_dat(self):
        self.put("doan-tuan", gloo("LỖI", "ĐẠT"))
        self.put("trinh-duc", gloo("ĐẠT", "ĐẠT"))
        v = summarize("selftest", self.dir, W)
        self.assertEqual(v["status"], "ĐẠT")
        self.assertIn("doan-tuan: theo-ten-card LỖI, theo-ip ĐẠT", self.text(v))
        self.assertIn("Thông đường TCP", v["hint"])

    def test_tu_thu_mot_may_loi(self):
        self.put("doan-tuan", gloo("LỖI", "LỖI", "RuntimeError: x"))
        self.put("trinh-duc", gloo("ĐẠT", "ĐẠT"))
        v = summarize("selftest", self.dir, W)
        self.assertEqual(v["status"], "LỖI")
        self.assertIn("doan-tuan", v["hint"])
        self.assertIn("không do mạng", v["hint"])
        self.assertIn("RuntimeError: x", self.text(v))

    def test_thieu_ket_qua_mot_may(self):
        self.put("doan-tuan", gloo("ĐẠT", "ĐẠT"))
        v = summarize("selftest", self.dir, W)
        self.assertEqual(v["status"], "LỖI")
        self.assertIn("trinh-duc: không có kết quả", self.text(v))

    def test_tcp_hai_chieu_thong(self):
        for i, m in enumerate(W):
            self.put(m, {"rank": i, "listen": {"ok": True, "message": "ĐẠT"},
                         "probe": {"ok": True, "message": "ĐẠT: kết nối được"}})
        v = summarize("porttest", self.dir, W)
        self.assertEqual(v["status"], "ĐẠT")
        self.assertIn("doan-tuan → trinh-duc: ĐẠT", self.text(v))
        self.assertIn("trinh-duc → doan-tuan: ĐẠT", self.text(v))
        self.assertIn("glootest", v["hint"])

    def test_tcp_bi_chan_toi_mot_may(self):
        self.put("doan-tuan", {"rank": 0, "listen": {"ok": False, "message": "LỖI"},
                               "probe": {"ok": True, "message": "ĐẠT"}})
        self.put("trinh-duc", {"rank": 1, "listen": {"ok": True, "message": "ĐẠT"},
                               "probe": {"ok": False, "message": "LỖI: ... bị chặn (tường lửa ...)"}})
        v = summarize("porttest", self.dir, W)
        self.assertEqual(v["status"], "LỖI")
        self.assertIn("trinh-duc → doan-tuan: LỖI", self.text(v))
        self.assertIn("tường lửa máy doan-tuan", v["hint"])

    def test_glootest_tim_ra_chon_sai_dia_chi(self):
        self.put("doan-tuan", gloo("LỖI", "ĐẠT"))
        self.put("trinh-duc", gloo("LỖI", "ĐẠT"))
        v = summarize("glootest", self.dir, W)
        self.assertEqual(v["status"], "ĐẠT")
        self.assertIn("theo-ten-card: LỖI", self.text(v))
        self.assertIn("theo-ip: ĐẠT", self.text(v))
        self.assertIn("theo IP", v["hint"])
        self.assertIn("chạy thử", v["hint"])

    def test_glootest_ten_card_bo_qua_khong_ket_luan_tim_ra(self):
        self.put("doan-tuan", gloo("BỎ QUA", "ĐẠT"))
        self.put("trinh-duc", gloo("BỎ QUA", "ĐẠT"))
        v = summarize("glootest", self.dir, W)
        self.assertEqual(v["status"], "ĐẠT")
        self.assertNotIn("Tìm ra", v["hint"])
        self.assertIn("không thử được", v["hint"])
        self.assertIn("chạy thử", v["hint"])

    def test_glootest_chi_ten_card_dat(self):
        self.put("doan-tuan", gloo("ĐẠT", "LỖI"))
        self.put("trinh-duc", gloo("ĐẠT", "LỖI"))
        v = summarize("glootest", self.dir, W)
        self.assertIn("theo tên card", v["hint"])

    def test_glootest_ca_hai_loi(self):
        self.put("doan-tuan", gloo("LỖI", "LỖI"))
        self.put("trinh-duc", gloo("LỖI", "LỖI"))
        v = summarize("glootest", self.dir, W)
        self.assertEqual(v["status"], "LỖI")
        self.assertIn("điểm gặp", v["hint"])
        self.assertIn("IPv6", v["hint"])

    def test_do_gpu(self):
        self.put("trinh-duc", {"models": {"resnet20": [
            {"batch": 64, "samples_per_s": 1000.0, "vram_peak_mb": 300},
            {"batch": 128, "samples_per_s": 1500.0, "vram_peak_mb": 500},
            {"batch": 256, "out_of_memory": True}]}, "suggestion": {"resnet20": 128}})
        v = summarize("profile", self.dir, ["trinh-duc"])
        self.assertEqual(v["status"], "ĐẠT")
        self.assertIn("trinh-duc resnet20: batch 128 = 1500 mẫu/s, VRAM 500 MB", self.text(v))
        self.assertIn("lớn nhất vừa VRAM: 128", self.text(v))

    def test_loai_khong_phai_chan_doan(self):
        self.assertIsNone(summarize("ddp", self.dir, W))


if __name__ == "__main__":
    unittest.main()
