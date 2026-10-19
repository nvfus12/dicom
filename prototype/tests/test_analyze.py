"""Checks of the analysis script on small hand-made logs (no training, no network).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyze"))

import analyze  # noqa: E402

BATCH = 100


def _spec(mode: str, workers: list[str], **extra) -> dict:
    spec = {"mode": mode, "workers": workers, "model": "resnet20", "batch_size": BATCH,
            "local_steps": 1, "comm_dtype": "fp32", "amp": False}
    spec.update(extra)
    return spec


def _training_log(spec: dict, rank: int, *, steps: int, train_time: float, accs: list[float],
                  comm_s: float = 0.0, comm_bytes: int = 0, temps: bool = False) -> list[dict]:
    """One rank's log: steps spread evenly over len(accs) epochs; data time 10% of busy."""
    epochs = len(accs)
    busy = train_time
    events = [{"event": "run_start", "run_id": "x", "rank": rank, "experiment": spec},
              {"event": "train_start", "base_lr": 0.1, "model_params": 1000, "batch_size": BATCH}]
    gpu = {"temp_c": 70, "sm_clock_mhz": 1500, "util_pct": 90, "mem_used_mb": 1} if temps else {}
    events.append({"event": "step", "step": steps, "epoch": epochs - 1, "train_time_s": train_time,
                   "steps_in_window": steps, "loss": 1.0, "data_s": 0.1 * busy,
                   "compute_s": 0.9 * busy - comm_s, "comm_s": comm_s, "comm_bytes": comm_bytes,
                   "samples_per_s": None, "gpu": gpu})
    if rank == 0:
        for i, acc in enumerate(accs):
            events.append({"event": "eval", "step": (i + 1) * steps // epochs, "epoch": i,
                           "train_time_s": train_time * (i + 1) / epochs, "test_acc": acc,
                           "test_loss": 1.0, "eval_s": 1.0})
    events.append({"event": "train_end", "stop_reason": "max_epochs", "steps": steps,
                   "epochs": epochs, "train_time_s": train_time, "sync_rounds": None,
                   "peak_memory_bytes": 2**30 if temps else None})
    return events


def _write(folder: Path, name: str, events: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / name).open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class AnalyzeEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        res = Path(cls.tmp.name)
        # Fast machine a: 1000 steps in 100 s -> 1000 samples/s, step 100 ms.
        _write(res / "single-a", "rank0-a.jsonl",
               _training_log(_spec("single", ["a"]), 0, steps=1000, train_time=100,
                             accs=[0.5, 0.7, 0.8, 0.9], temps=True))
        # Slow machine b: 1000 steps in 200 s -> 500 samples/s, step 200 ms.
        _write(res / "single-b", "rank0-b.jsonl",
               _training_log(_spec("single", ["b"]), 0, steps=1000, train_time=200,
                             accs=[0.5, 0.7, 0.8, 0.9]))
        # DDP a+b: 500 steps in 125 s -> group 2*500*100/125 = 800 samples/s, step 250 ms.
        ddp = _spec("ddp", ["a", "b"])
        _write(res / "ddp-ab", "rank0-a.jsonl",
               _training_log(ddp, 0, steps=500, train_time=125, accs=[0.4, 0.6, 0.75, 0.85]))
        _write(res / "ddp-ab", "rank1-b.jsonl",
               _training_log(ddp, 1, steps=500, train_time=125, accs=[]))
        # Netcheck folder: no run_start; used to crash the old script.
        _write(res / "netcheck-1", "rank0-a.jsonl", [
            {"event": "tailscale", "machine": "a",
             "peers": {"b": {"available": True, "relayed": False, "via": "1.2.3.4:41641"}}},
            {"event": "pairs", "machine": "a", "latency_ms": {"a<->b": 12.5},
             "mb_per_s": {"a->b": 5.0}},
            {"event": "allreduce", "seconds_median": {"1.1MB": 0.4}}])
        _write(res / "netcheck-1", "rank1-b.jsonl", [
            {"event": "tailscale", "machine": "b", "peers": {"a": {"available": False}}},
            {"event": "pairs", "machine": "b", "latency_ms": {}, "mb_per_s": {"b->a": 4.0}}])
        # A folder with nothing usable must be skipped, not crash.
        _write(res / "broken", "rank0-a.jsonl", [{"event": "error", "traceback": "boom"}])
        cls.out = res / "report"
        code = analyze.main(["--results", str(res), "--acc", "0.8", "--at-minutes", "1"])
        assert code == 0
        with (cls.out / "runs.csv").open(encoding="utf-8") as f:
            cls.runs = {r["run_id"]: r for r in csv.DictReader(f)}
        with (cls.out / "network.csv").open(encoding="utf-8") as f:
            cls.net = list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_netcheck_and_broken_folders_do_not_crash_and_are_not_runs(self):
        self.assertEqual(set(self.runs), {"single-a", "single-b", "ddp-ab"})

    def test_time_to_accuracy_and_speedup_against_fastest_single(self):
        # single-a reaches 0.8 at epoch 3 of 4 -> 75 s; ddp reaches 0.85 at epoch 4 -> 125 s.
        self.assertAlmostEqual(float(self.runs["single-a"]["time_to_0.8_s"]), 75.0)
        self.assertAlmostEqual(float(self.runs["ddp-ab"]["time_to_0.8_s"]), 125.0)
        self.assertEqual(self.runs["ddp-ab"]["baseline_run"], "single-a")
        self.assertAlmostEqual(float(self.runs["ddp-ab"]["speedup_0.8"]), 75 / 125)

    def test_scaling_efficiency_and_sync_overhead(self):
        r = self.runs["ddp-ab"]
        self.assertAlmostEqual(float(r["throughput_samples_per_s"]), 800.0)
        self.assertAlmostEqual(float(r["scaling_efficiency"]), 800 / (1000 + 500))
        # DDP step 250 ms minus the slowest machine alone (200 ms).
        self.assertAlmostEqual(float(r["sync_overhead_ms_per_step"]), 50.0)

    def test_ddp_traffic_is_estimated_from_parameter_count(self):
        r = self.runs["ddp-ab"]
        self.assertEqual(r["comm_mb_is_estimate"], "True")
        self.assertAlmostEqual(float(r["comm_mb_rank0"]), 500 * 1000 * 4 / 2**20)

    def test_accuracy_at_minutes(self):
        # Within 60 s single-a has evaluated at 25 s and 50 s -> best 0.7.
        self.assertAlmostEqual(float(self.runs["single-a"]["acc_at_1min"]), 0.7)

    def test_network_table(self):
        pairs = {r["item"]: r for r in self.net if r["kind"] == "pair"}
        self.assertAlmostEqual(float(pairs["a->b"]["rtt_ms"]), 12.5)
        self.assertAlmostEqual(float(pairs["b->a"]["rtt_ms"]), 12.5)   # same pair, either order
        self.assertEqual(pairs["a->b"]["path"], "direct")
        self.assertEqual(pairs["b->a"]["path"], "unknown")
        allreduce = [r for r in self.net if r["kind"] == "allreduce"]
        self.assertAlmostEqual(float(allreduce[0]["allreduce_s"]), 0.4)

    def test_all_outputs_written(self):
        for name in ("runs.csv", "ranks.csv", "network.csv", "table_runs.tex",
                     "table_ranks.tex", "table_network.tex", "accuracy_vs_time.png",
                     "accuracy_vs_epoch.png", "time_breakdown.png", "gpu_thermal.png"):
            self.assertTrue((self.out / name).is_file(), name)

    def test_latex_escaping(self):
        tex = (self.out / "table_ranks.tex").read_text(encoding="utf-8")
        self.assertIn(r"\begin{tabular}", tex)
        self.assertEqual(analyze.latex_escape("a_b 50% & #1"), r"a\_b 50\% \& \#1")


class MetricHelpersTest(unittest.TestCase):
    def test_time_to_accuracy_never_reached(self):
        self.assertIsNone(analyze.time_to_accuracy([{"test_acc": 0.5, "train_time_s": 10}], 0.9))

    def test_ratio_handles_missing_values(self):
        self.assertIsNone(analyze.ratio(None, 2))
        self.assertIsNone(analyze.ratio(1, 0))
        self.assertEqual(analyze.ratio(3, 2), 1.5)


if __name__ == "__main__":
    unittest.main()
