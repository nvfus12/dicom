"""Summarises runs from results/<run_id>/rank0-*.jsonl.

    python analyze/analyze.py                 # every run
    python analyze/analyze.py e3_localsgd     # runs whose id contains this text

Writes results/summary.csv and, if matplotlib is installed, results/accuracy_vs_time.png.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise(events: list[dict]) -> dict:
    start = next(e for e in events if e["event"] == "run_start")
    end = next((e for e in events if e["event"] == "train_end"), {})
    exp = start["experiment"]
    steps = [e for e in events if e["event"] == "step"]
    data_s = sum(e["data_s"] for e in steps)
    compute_s = sum(e["compute_s"] for e in steps)
    comm_s = sum(e["comm_s"] for e in steps)
    busy = data_s + compute_s + comm_s
    samples = sum(e["steps_in_window"] for e in steps) * exp["batch_size"]
    return {
        "run_id": start["run_id"],
        "mode": exp["mode"],
        "workers": "+".join(exp["workers"]),
        "model": exp["model"],
        "local_steps": exp["local_steps"] if exp["mode"] == "localsgd" else "",
        "outer_optimizer": exp.get("outer_optimizer", "avg") if exp["mode"] == "localsgd" else "",
        "outer_lr": exp.get("outer_lr", 1.0) if exp["mode"] == "localsgd" else "",
        "outer_momentum": exp.get("outer_momentum", 0.0) if exp["mode"] == "localsgd" else "",
        "comm_dtype": exp.get("comm_dtype", "fp32"),
        "sync_rounds": end.get("sync_rounds"),
        "batch_per_worker": exp["batch_size"],
        "stop_reason": end.get("stop_reason", "INCOMPLETE"),
        "best_acc": end.get("best_acc"),
        "target_acc": exp["target_acc"],
        "time_to_target_s": end.get("time_to_target_s"),
        "train_time_s": end.get("train_time_s"),
        "comm_share": round(comm_s / busy, 3) if busy else None,
        "data_share": round(data_s / busy, 3) if busy else None,
        "rank0_samples_per_s": round(samples / busy, 1) if busy else None,
        "comm_mb_rank0": round(sum(e["comm_bytes"] for e in steps) / 2**20, 1),
        "peak_mem_mb": (end.get("peak_memory_bytes") or 0) // 2**20 or None,
    }


def main() -> int:
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    logs = sorted(p for p in RESULTS.glob("*/rank0-*.jsonl") if needle in p.parent.name)
    if not logs:
        print(f"Không có run nào trong {RESULTS} khớp '{needle}'")
        return 1
    runs = [(p, load(p)) for p in logs]
    rows = [summarise(ev) for _, ev in runs]

    out_csv = RESULTS / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['run_id']:<45} {r['mode']:<8} H={r['local_steps'] or '-':<3} "
              f"outer={r['outer_optimizer'] or '-'} {r['comm_dtype']} "
              f"acc={r['best_acc']} ttt={r['time_to_target_s']} comm={r['comm_share']}")
    print(f"Đã ghi {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib, bỏ qua đồ thị")
        return 0
    fig, ax = plt.subplots(figsize=(8, 5))
    for (_, events), row in zip(runs, rows):
        evals = [e for e in events if e["event"] == "eval"]
        ax.plot([e["train_time_s"] / 60 for e in evals], [e["test_acc"] for e in evals],
                marker="o", markersize=3, label=row["run_id"])
    targets = {r["target_acc"] for r in rows if r["target_acc"]}
    for t in targets:
        ax.axhline(t, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Thời gian huấn luyện (phút, không tính đánh giá)")
    ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    out_png = RESULTS / "accuracy_vs_time.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Đã ghi {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
