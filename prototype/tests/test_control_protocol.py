"""Tests for the lab control protocol: run parameters and the experiment they become."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpubench.control import protocol  # noqa: E402
from gpubench.domain.config import ExperimentSpec  # noqa: E402

QUICK = {"quick": True, "dataset": "synthetic", "model": "tiny_cnn"}


class ValidateParamsTest(unittest.TestCase):
    def test_tham_so_hop_le_va_mac_dinh(self):
        p = protocol.validate_params({"mode": "localsgd", "local_steps": 64})
        self.assertEqual((p.model, p.dataset, p.local_steps, p.quick), ("resnet20", "cifar10", 64, False))
        self.assertEqual((p.max_train_minutes, p.max_epochs, p.batch_size, p.lr, p.comm_dtype),
                         (30, 30, 128, 0.1, "fp32"))

    def test_khoa_la_bi_tu_choi(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.validate_params({"mode": "ddp", "command": "del C:\\"})
        self.assertIn("command", str(caught.exception))

    def test_gioi_han(self):
        for bad in ({"mode": "khong-co"}, {},{"mode": "ddp", "max_train_minutes": 0},
                    {"mode": "ddp", "max_train_minutes": 241}, {"mode": "localsgd", "local_steps": 5000},
                    {"mode": "localsgd", "local_steps": 0}, {"mode": "ddp", "max_epochs": 301},
                    {"mode": "ddp", "batch_size": 8}, {"mode": "ddp", "lr": 2},
                    {"mode": "ddp", "comm_dtype": "bf16"}, {"mode": "ddp", "model": "tiny_cnn"},
                    {"mode": "ddp", "dataset": "synthetic"},
                    {"mode": "ddp", "quick": True, "dataset": "cifar10", "model": "tiny_cnn"},
                    {"mode": "ddp", "quick": True, "dataset": "synthetic", "model": "resnet20"},
                    {"mode": "ddp", "local_steps": "nhieu"}, {"mode": "ddp", "quick": "co"}):
            with self.subTest(bad=bad), self.assertRaises(protocol.ProtocolError):
                protocol.validate_params(bad)

    def test_khong_phai_dict(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_params(["ddp"])

    def test_to_dict_doc_lai_duoc(self):
        p = protocol.validate_params({"mode": "sync", **QUICK})
        self.assertEqual(protocol.validate_params(p.to_dict()), p)


class BuildExperimentTest(unittest.TestCase):
    def _spec(self, data: dict) -> ExperimentSpec:
        spec = ExperimentSpec(**{**data, "workers": tuple(data["workers"])})
        spec.validate()
        return spec

    def test_build_experiment_qua_validate_va_giong_nhau(self):
        p = protocol.validate_params({"mode": "ddp"})
        run_id = "lab-ddp-resnet20-20261116-100000"
        a = protocol.build_experiment(p, ["doan-tuan", "trinh-duc"], run_id)
        b = protocol.build_experiment(p, ["doan-tuan", "trinh-duc"], run_id)
        self.assertEqual(a, b)
        spec = self._spec(a)
        self.assertEqual((spec.name, spec.workers), (run_id, ("doan-tuan", "trinh-duc")))
        self.assertEqual((spec.warmup_epochs, spec.lr_scaling, spec.timeout_s, spec.master_port),
                         (2, "linear", 300, 29500))
        self.assertEqual((spec.target_acc, spec.amp, spec.seed), (None, False, 0))

    def test_build_experiment_localsgd(self):
        p = protocol.validate_params({"mode": "localsgd", "local_steps": 128})
        d = protocol.build_experiment(p, ["a", "b"], "x")
        self.assertEqual((d["local_steps"], d["lr_scaling"], d["warmup_epochs"]), (128, "none", 1))
        self._spec(d)

    def test_build_experiment_sync_va_chay_thu(self):
        p = protocol.validate_params({"mode": "sync", "max_epochs": 2, **QUICK})
        d = protocol.build_experiment(p, ["sim-0", "sim-1"], "thu")
        spec = self._spec(d)
        self.assertEqual((spec.dataset, spec.model, spec.synthetic_samples, spec.max_epochs),
                         ("synthetic", "tiny_cnn", 2048, 2))
        self.assertNotIn("local_steps", d)


class RunIdAndRankTest(unittest.TestCase):
    def test_run_id(self):
        now = datetime(2026, 11, 16, 9, 5, 7)
        self.assertEqual(protocol.new_run_id(protocol.validate_params({"mode": "ddp"}), now),
                         "lab-ddp-resnet20-20261116-090507")
        q = protocol.validate_params({"mode": "localsgd", **QUICK})
        self.assertEqual(protocol.new_run_id(q, now), "thu-localsgd-tiny_cnn-20261116-090507")

    def test_run_id_hop_le(self):
        self.assertTrue(protocol.valid_run_id("lab-ddp-resnet20-20261116-090507"))
        self.assertTrue(protocol.valid_run_id("thu-localsgd-tiny_cnn-20261116-090507"))
        for bad in ("..", "../a", r"a\b", "a/b", "C:x", "", ".a", "-a", "x" * 81, None, 5):
            self.assertFalse(protocol.valid_run_id(bad), bad)

    def test_rank_order(self):
        infos = [{"name": "trinh-duc", "gpu": {"vram_mb": 4096}},
                 {"name": "doan-tuan", "gpu": {"vram_mb": 6141}},
                 {"name": "sim-1", "gpu": {"vram_mb": None}},
                 {"name": "sim-0", "gpu": {"vram_mb": None}}]
        self.assertEqual(protocol.rank_order(infos), ["doan-tuan", "trinh-duc", "sim-0", "sim-1"])

    def test_describe_params(self):
        rows = dict(protocol.describe_params(protocol.validate_params({"mode": "localsgd"})))
        self.assertEqual(rows["Chế độ"], "Local SGD")
        self.assertEqual(rows["Đồng bộ mỗi (H)"], "256 bước")
        self.assertEqual(rows["Thời gian tối đa"], "30 phút")
        ddp = dict(protocol.describe_params(protocol.validate_params({"mode": "ddp"})))
        self.assertNotIn("Đồng bộ mỗi (H)", ddp)


NATIVE_CRASH = """[doan-tuan] rank 0/2, chờ các máy khác tại 192.168.0.102:29500 ...
[W1117 16:03:59.000000000 socket.cpp:759] [c10d] The client socket has failed to connect to [LAPTOP-X]:29500 (system error: 10049 - The requested address is not valid in its context.).
Unhandled exception caught in c10/util/AbortHandler.h
00007FF8B6F194E400007FF8B6EF6170 torch_python.dll!THPGenerator_initDefaultGenerator [<unknown file> @ <unknown line number>]
00007FF9723B218700007FF9723B2170 ucrtbase.dll!terminate [<unknown file> @ <unknown line number>]
00007FF94FED1AB1 <unknown symbol address> VCRUNTIME140_1.dll!<unknown symbol> [<unknown file> @ <unknown line number>]
00007FFF99CA7CCF00007FFF99CA7910 torch_cpu.dll!c10d::ProcessGroupGloo::ProcessGroupGloo [<unknown file> @ <unknown line number>]
"""

PY_TRACEBACK = """[sim-0] rank 0/2, chờ các máy khác tại 127.0.0.1:29500 ...
Traceback (most recent call last):
  File "x.py", line 3, in <module>
    main()
RuntimeError: Gloo connectFullMesh failed
"""


class LogExcerptTest(unittest.TestCase):
    def test_bo_dong_stack_trace_giu_dong_co_nghia(self):
        text = protocol.log_excerpt(NATIVE_CRASH)
        self.assertIn("failed to connect", text)
        self.assertIn("Unhandled exception caught", text)
        self.assertIn("ProcessGroupGloo::ProcessGroupGloo", text)   # the frame that names the failing step
        self.assertNotIn("THPGenerator_initDefaultGenerator", text)
        self.assertNotIn("<unknown symbol>", text)

    def test_giu_nguyen_traceback_python(self):
        text = protocol.log_excerpt(PY_TRACEBACK)
        self.assertIn("Traceback (most recent call last)", text)
        self.assertIn("RuntimeError: Gloo connectFullMesh failed", text)

    def test_gioi_han_so_dong(self):
        text = protocol.log_excerpt("\n".join(f"dòng {i}" for i in range(500)), max_lines=10)
        self.assertEqual(len(text.splitlines()), 10)
        self.assertIn("dòng 499", text)

    def test_log_rong(self):
        self.assertEqual(protocol.log_excerpt(""), "")


class NetcheckParamsTest(unittest.TestCase):
    def test_netcheck_hop_le_va_khong_phai_chay_thu(self):
        p = protocol.validate_params({"mode": "netcheck"})
        self.assertEqual((p.mode, p.quick), ("netcheck", False))
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_params({"mode": "netcheck", "quick": True, "dataset": "synthetic",
                                      "model": "tiny_cnn"})

    def test_netcheck_run_id_va_mo_ta(self):
        p = protocol.validate_params({"mode": "netcheck"})
        self.assertEqual(protocol.new_run_id(p, datetime(2026, 11, 18, 9, 0, 0)),
                         "lab-netcheck-20261118-090000")
        rows = dict(protocol.describe_params(p))
        self.assertIn("Kiểm tra mạng", rows["Chế độ"])
        self.assertNotIn("Model", rows)

    def test_netcheck_experiment_hop_le(self):
        p = protocol.validate_params({"mode": "netcheck"})
        d = protocol.build_experiment(p, ["doan-tuan", "trinh-duc"], "lab-netcheck-x")
        spec = ExperimentSpec(**{**d, "workers": tuple(d["workers"])})
        spec.validate()
        self.assertEqual((spec.world_size, spec.master_port, spec.timeout_s), (2, 29500, 300))


class JobKindsTest(unittest.TestCase):
    NOW = datetime(2026, 11, 27, 9, 15, 0)

    def test_moi_loai_luot_hop_le(self):
        for mode in ("selftest", "glootest", "porttest", "profile", "single"):
            p = protocol.validate_params({"mode": mode})
            self.assertEqual((p.mode, p.gloo_device), (mode, "ipv4"))

    def test_cach_ket_noi_gloo(self):
        self.assertEqual(protocol.validate_params({"mode": "ddp", "gloo_device": "interface"}).gloo_device,
                         "interface")
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_params({"mode": "ddp", "gloo_device": "wifi"})

    def test_chan_doan_khong_co_chay_thu(self):
        for mode in ("selftest", "glootest", "porttest"):
            with self.subTest(mode=mode), self.assertRaises(protocol.ProtocolError):
                protocol.validate_params({"mode": mode, **QUICK})

    def test_so_may_va_gpu_theo_loai(self):
        k = protocol.job_kind
        self.assertEqual((k("selftest").min_machines, k("selftest").max_machines), (1, None))
        self.assertEqual((k("porttest").min_machines, k("porttest").max_machines), (2, 2))
        self.assertEqual((k("single").min_machines, k("single").max_machines), (1, 1))
        self.assertEqual(k("ddp").min_machines, 2)
        self.assertTrue(k("profile").needs_cuda and k("single").needs_cuda and k("ddp").needs_cuda)
        self.assertFalse(k("glootest").needs_cuda or k("porttest").needs_cuda or k("netcheck").needs_cuda)
        self.assertTrue(k("glootest").diagnostic and not k("ddp").diagnostic)
        self.assertTrue(k("ddp").analyse and k("single").analyse and k("netcheck").analyse)
        self.assertFalse(k("glootest").analyse or k("profile").analyse)

    def test_ten_thu_muc_theo_loai(self):
        rid = protocol.new_run_id
        p = protocol.validate_params
        self.assertEqual(rid(p({"mode": "selftest"}), self.NOW, ["doan-tuan", "trinh-duc"]),
                         "lab-selftest-20261127-091500")
        self.assertEqual(rid(p({"mode": "glootest"}), self.NOW, ["trinh-duc", "doan-tuan"]),
                         "lab-glootest-trinh-duc-doan-tuan-20261127-091500")
        self.assertEqual(rid(p({"mode": "porttest"}), self.NOW, ["doan-tuan", "trinh-duc"]),
                         "lab-porttest-doan-tuan-trinh-duc-20261127-091500")
        self.assertEqual(rid(p({"mode": "profile"}), self.NOW, ["doan-tuan"]),
                         "lab-profile-resnet20-20261127-091500")
        self.assertEqual(rid(p({"mode": "single"}), self.NOW, ["trinh-duc"]),
                         "lab-single-resnet20-trinh-duc-20261127-091500")
        self.assertEqual(rid(p({"mode": "ddp"}), self.NOW, ["a", "b"]), "lab-ddp-resnet20-20261127-091500")
        self.assertEqual(rid(p({"mode": "single", **QUICK}), self.NOW, ["sim-0"]),
                         "thu-single-tiny_cnn-sim-0-20261127-091500")

    def test_experiment_single_va_glootest(self):
        d = protocol.build_experiment(protocol.validate_params({"mode": "single"}), ["trinh-duc"], "x")
        spec = ExperimentSpec(**{**d, "workers": tuple(d["workers"])})
        spec.validate()
        self.assertEqual((spec.mode, spec.lr_scaling), ("single", "none"))
        d = protocol.build_experiment(protocol.validate_params({"mode": "glootest"}), ["a", "b"], "y")
        ExperimentSpec(**{**d, "workers": tuple(d["workers"])}).validate()

    def test_mo_ta_hop_thoai(self):
        rows = dict(protocol.describe_params(protocol.validate_params({"mode": "porttest"})))
        self.assertIn("TCP", rows["Chế độ"])
        rows = dict(protocol.describe_params(protocol.validate_params({"mode": "ddp",
                                                                       "gloo_device": "interface"})))
        self.assertEqual(rows["Cách kết nối gloo"], "theo tên card")


if __name__ == "__main__":
    unittest.main()
