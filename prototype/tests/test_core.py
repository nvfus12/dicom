"""Correctness checks that run on CPU, no GPU or network needed.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpubench.adapters.comm import GlooComm  # noqa: E402
from gpubench.adapters.device import CpuDevice  # noqa: E402
from gpubench.adapters.models import tiny_cnn  # noqa: E402
from gpubench.app.config_loader import load_experiment, load_machine, list_machines  # noqa: E402
from gpubench.domain.config import ConfigError, ExperimentSpec  # noqa: E402
from gpubench.app.netcheck import parse_tailscale_ping  # noqa: E402
from gpubench.domain.sync import build_strategy  # noqa: E402

WORLD = 3


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _same_on_all_ranks(t: torch.Tensor) -> bool:
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return all(torch.allclose(gathered[0], g) for g in gathered[1:])


def _worker(rank: int, port: int, mode: str, local_steps: int, results) -> None:
    comm = GlooComm("127.0.0.1", port, rank, WORLD, timeout_s=60)
    try:
        device = CpuDevice()
        torch.manual_seed(0)
        model = tiny_cnn()
        torch.manual_seed(100 + rank)           # different data per rank
        strategy = build_strategy(mode, comm, device, local_steps)
        model = strategy.prepare(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        same_after = []
        for step in range(1, 5):
            x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
            torch.nn.functional.cross_entropy(model(x), y).backward()
            strategy.after_backward(model)
            opt.step()
            opt.zero_grad()
            strategy.after_step(model, step)
            same_after.append(_same_on_all_ranks(_flat_params(strategy.unwrap(model))))
        if rank == 0:
            results[mode] = same_after
    finally:
        comm.close()


def _run(mode: str, local_steps: int = 1) -> list[bool]:
    port = _free_port()
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(port, mode, local_steps, results), nprocs=WORLD, join=True)
        return list(results[mode])


class SyncStrategyTest(unittest.TestCase):
    def test_sync_keeps_models_identical(self):
        self.assertEqual(_run("sync"), [True, True, True, True])

    def test_ddp_keeps_models_identical(self):
        self.assertEqual(_run("ddp"), [True, True, True, True])

    def test_localsgd_identical_only_after_averaging(self):
        # H=2: models diverge after odd steps, agree right after each averaging.
        self.assertEqual(_run("localsgd", local_steps=2), [False, True, False, True])


class ConfigTest(unittest.TestCase):
    def test_all_shipped_configs_load(self):
        for name in list_machines():
            load_machine(name)
        for ref in ("e2_ddp_t1000x2", "e2_ddp_x3", "e3_localsgd_x3", "smoke_dist"):
            load_experiment(ref)
        load_experiment("e1_single", runner="doan-tuan")

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_experiment("smoke_dist", {"batchsize": 64})

    def test_fingerprint_changes_with_overrides(self):
        a = load_experiment("e3_localsgd_x3")
        b = load_experiment("e3_localsgd_x3", {"local_steps": 64})
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_single_mode_rejects_many_workers(self):
        with self.assertRaises(ConfigError):
            ExperimentSpec(name="x", mode="single", workers=("a", "b")).validate()

    def test_linear_lr_scaling_only_for_gradient_sync(self):
        ddp = ExperimentSpec(name="x", mode="ddp", workers=("a", "b", "c"), lr=0.1)
        lsgd = ExperimentSpec(name="x", mode="localsgd", workers=("a", "b", "c"), lr=0.1)
        self.assertAlmostEqual(ddp.effective_lr(), 0.3)
        self.assertAlmostEqual(lsgd.effective_lr(), 0.1)


class TailscalePingParseTest(unittest.TestCase):
    def test_relayed_then_direct_counts_as_direct(self):
        text = ("pong from a (100.64.0.1) via DERP(sin) in 180ms\n"
                "pong from a (100.64.0.1) via 203.0.113.5:41641 in 30ms")
        r = parse_tailscale_ping(text)
        self.assertFalse(r["relayed"])
        self.assertEqual(r["via"], "203.0.113.5:41641")

    def test_relayed_only(self):
        r = parse_tailscale_ping("pong from a (100.64.0.1) via DERP(sin) in 180ms")
        self.assertTrue(r["relayed"])

    def test_unparseable_output_is_unknown(self):
        self.assertIsNone(parse_tailscale_ping("no matching peer")["relayed"])


if __name__ == "__main__":
    unittest.main()
