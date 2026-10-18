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
from gpubench.domain.sync import build_strategy, outer_update_  # noqa: E402

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


def _worker(rank: int, port: int, mode: str, options: dict, results) -> None:
    comm = GlooComm("127.0.0.1", port, rank, WORLD, timeout_s=60)
    try:
        device = CpuDevice()
        torch.manual_seed(0)
        model = tiny_cnn()
        torch.manual_seed(100 + rank)           # different data per rank
        strategy = build_strategy(mode, comm, device, **options)
        model = strategy.prepare(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        same_after, mean_err = [], []
        h = options.get("local_steps", 1)
        for step in range(1, 5):
            x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
            torch.nn.functional.cross_entropy(model(x), y).backward()
            strategy.after_backward(model)
            opt.step()
            opt.zero_grad()
            syncing = mode == "localsgd" and step % h == 0
            if syncing:
                # Plain mean of the workers' models just before synchronisation.
                before = _flat_params(model)
                gathered = [torch.empty_like(before) for _ in range(WORLD)]
                dist.all_gather(gathered, before)
                expected_mean = torch.stack(gathered).mean(0)
            strategy.after_step(model, step)
            after = _flat_params(strategy.unwrap(model))
            same_after.append(_same_on_all_ranks(after))
            if syncing:
                mean_err.append((after - expected_mean).abs().max().item())
        if rank == 0:
            results["out"] = {"same": same_after, "mean_err": mean_err}
    finally:
        comm.close()


def _run(mode: str, **options) -> dict:
    port = _free_port()
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(port, mode, options, results), nprocs=WORLD, join=True)
        return dict(results["out"])


class SyncStrategyTest(unittest.TestCase):
    def test_sync_keeps_models_identical(self):
        for dtype in ("fp32", "fp16", "bf16"):
            with self.subTest(comm_dtype=dtype):
                self.assertEqual(_run("sync", comm_dtype=dtype)["same"], [True] * 4)

    def test_ddp_keeps_models_identical(self):
        for dtype in ("fp32", "fp16"):
            with self.subTest(comm_dtype=dtype):
                self.assertEqual(_run("ddp", comm_dtype=dtype)["same"], [True] * 4)

    def test_localsgd_avg_is_exact_mean_of_workers(self):
        # H=2: models diverge after odd steps, agree right after each averaging,
        # and the agreed model is the plain mean of the workers' models.
        out = _run("localsgd", local_steps=2)
        self.assertEqual(out["same"], [False, True, False, True])
        self.assertLess(max(out["mean_err"]), 1e-6)

    def test_localsgd_nesterov_without_momentum_equals_avg(self):
        out = _run("localsgd", local_steps=2, outer_optimizer="nesterov")
        self.assertLess(max(out["mean_err"]), 1e-6)

    def test_localsgd_diloco_moves_beyond_the_mean(self):
        out = _run("localsgd", local_steps=2, outer_optimizer="nesterov",
                   outer_lr=0.7, outer_momentum=0.9)
        self.assertEqual(out["same"], [False, True, False, True])
        self.assertGreater(max(out["mean_err"]), 1e-4)

    def test_localsgd_fp16_close_to_exact_mean(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(comm_dtype=dtype):
                out = _run("localsgd", local_steps=2, comm_dtype=dtype)
                self.assertEqual(out["same"], [False, True, False, True])
                # Only the pseudo-gradient is rounded, never the weights themselves.
                self.assertLess(max(out["mean_err"]), 1e-3)


class LocalSGDEvalModelTest(unittest.TestCase):
    def test_eval_uses_global_model_and_restores_local_weights(self):
        from gpubench.adapters.comm import LocalComm

        torch.manual_seed(0)
        model = tiny_cnn()
        strategy = build_strategy("localsgd", LocalComm(), CpuDevice(), local_steps=100)
        strategy.prepare(model)
        initial = _flat_params(model).clone()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)                     # local training, no sync yet
        local = _flat_params(model).clone()
        with strategy.eval_model(model) as m:
            self.assertTrue(torch.equal(_flat_params(m), initial))
        self.assertTrue(torch.equal(_flat_params(model), local))


class OuterUpdateTest(unittest.TestCase):
    def test_matches_torch_sgd_nesterov(self):
        # The outer step is SGD-Nesterov with the pseudo-gradient as gradient.
        ref = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        opt = torch.optim.SGD([ref], lr=0.7, momentum=0.9, nesterov=True)
        mine, buf = torch.tensor([1.0, -2.0]), torch.zeros(2)
        for g in (torch.tensor([0.1, 0.3]), torch.tensor([-0.2, 0.05]), torch.tensor([0.4, 0.0])):
            ref.grad = g.clone()
            opt.step()
            outer_update_(mine, g, buf, lr=0.7, momentum=0.9)
            self.assertTrue(torch.allclose(mine, ref.detach(), atol=1e-6))

    def test_lr_one_no_momentum_is_plain_subtraction(self):
        g = torch.tensor([1.0, 2.0])
        outer_update_(g, torch.tensor([0.25, -0.5]), None, lr=1.0, momentum=0.0)
        self.assertTrue(torch.equal(g, torch.tensor([0.75, 2.5])))


class ConfigTest(unittest.TestCase):
    def test_all_shipped_configs_load(self):
        for name in list_machines():
            load_machine(name)
        for ref in ("e2_ddp_t1000x2", "e2_ddp_x3", "e3_localsgd_x3", "smoke_dist",
                    "run2_ddp_x3", "run3_localsgd_x3"):
            load_experiment(ref)
        load_experiment("e1_single", runner="doan-tuan")
        load_experiment("run1_single", runner="doan-tuan")

    def test_amp_is_part_of_the_experiment_fingerprint(self):
        a = load_experiment("run2_ddp_x3")
        b = load_experiment("run2_ddp_x3", {"amp": True})
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_machine_files_no_longer_accept_amp(self):
        from gpubench.domain.config import MachineProfile
        with self.assertRaises(TypeError):
            MachineProfile(name="x", amp=True)

    def test_master_addr_override_skips_rank0_file(self):
        from gpubench.app.bootstrap import rendezvous_address
        spec = ExperimentSpec(name="x", mode="ddp", workers=("no-such-machine", "b"))
        self.assertEqual(rendezvous_address(spec, Path("."), "100.64.0.7"), "100.64.0.7")

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_experiment("smoke_dist", {"batchsize": 64})

    def test_fingerprint_changes_with_overrides(self):
        a = load_experiment("e3_localsgd_x3")
        b = load_experiment("e3_localsgd_x3", {"local_steps": 64})
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_outer_options_rejected_outside_localsgd(self):
        with self.assertRaises(ConfigError):
            ExperimentSpec(name="x", mode="ddp", workers=("a", "b"),
                           outer_optimizer="nesterov").validate()

    def test_avg_rejects_outer_hyperparameters(self):
        with self.assertRaises(ConfigError):
            ExperimentSpec(name="x", mode="localsgd", workers=("a", "b"), outer_lr=0.7).validate()

    def test_comm_dtype_validation(self):
        with self.assertRaises(ConfigError):
            ExperimentSpec(name="x", mode="sync", workers=("a", "b"), comm_dtype="int8").validate()
        with self.assertRaises(ConfigError):
            ExperimentSpec(name="x", mode="single", workers=("a",), comm_dtype="fp16").validate()
        ExperimentSpec(name="x", mode="localsgd", workers=("a", "b"), outer_optimizer="nesterov",
                       outer_lr=0.7, outer_momentum=0.9, comm_dtype="fp16").validate()

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
