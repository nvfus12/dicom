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
        for bad in ({"mode": "single"}, {}, {"mode": "ddp", "max_train_minutes": 0},
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


if __name__ == "__main__":
    unittest.main()
