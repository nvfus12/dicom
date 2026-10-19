"""Turns results/ into the tables and figures of the benchmark report.

    python analyze/analyze.py                          # every run in results/
    python analyze/analyze.py run                      # only run ids containing "run"
    python analyze/analyze.py --acc 0.6 0.7 0.8 --at-minutes 10 30
    python analyze/analyze.py --baseline run1_single-2026...   # baseline run (default: fastest single run)

Reads the logs of every rank, and netcheck folders. Writes to results/report/ (or --out):
  runs.csv, ranks.csv, network.csv                  tables as CSV
  table_runs.tex, table_ranks.tex, table_network.tex  same tables, LaTeX (booktabs)
  accuracy_vs_time.png, accuracy_vs_epoch.png, time_breakdown.png, gpu_thermal.png
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------- loading

@dataclass
class RankLog:
    rank: int
    machine: str
    events: list[dict]

    def of(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]

    def first(self, name: str) -> dict:
        return next((e for e in self.events if e["event"] == name), {})


@dataclass
class TrainingRun:
    run_id: str
    ranks: dict[int, RankLog]

    @property
    def root(self) -> RankLog:
        return self.ranks[min(self.ranks)]

    @property
    def spec(self) -> dict:
        return self.root.first("run_start")["experiment"]


@dataclass
class NetcheckRun:
    run_id: str
    ranks: dict[int, RankLog] = field(default_factory=dict)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rank_and_machine(path: Path) -> tuple[int, str]:
    # File names look like rank0-doan-tuan.jsonl
    head, _, machine = path.stem.partition("-")
    return int(head.removeprefix("rank")), machine


def load_results(root: Path, needle: str = "") -> tuple[list[TrainingRun], list[NetcheckRun]]:
    """Training runs need a run_start event; netcheck folders have 'pairs' events.
    Anything else (an empty or broken folder) is skipped with a message."""
    runs, netchecks = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and needle in p.name):
        ranks = {}
        for path in sorted(folder.glob("rank*-*.jsonl")):
            rank, machine = _rank_and_machine(path)
            if rank in ranks:
                print(f"CẢNH BÁO {folder.name}: hai file cùng rank {rank} ({ranks[rank].machine}, "
                      f"{machine}); có thể là hai lần chạy khác nhau trùng run_id. "
                      f"Chỉ dùng file đầu tiên, hãy tách thư mục.", file=sys.stderr)
                continue
            ranks[rank] = RankLog(rank, machine, load_jsonl(path))
        if not ranks:
            continue
        if any(r.first("run_start") for r in ranks.values()):
            runs.append(TrainingRun(folder.name, {k: v for k, v in ranks.items()
                                                  if v.first("run_start")}))
        elif any(r.of("pairs") for r in ranks.values()):
            netchecks.append(NetcheckRun(folder.name, ranks))
        else:
            print(f"Bỏ qua {folder.name}: không có run_start hay kết quả netcheck", file=sys.stderr)
    return runs, netchecks


# ---------------------------------------------------------------- metrics

def time_to_accuracy(evals: list[dict], level: float) -> float | None:
    """Training time (s) at the first evaluation reaching `level`; None if never reached."""
    for e in evals:
        if e["test_acc"] >= level:
            return e["train_time_s"]
    return None


def accuracy_at(evals: list[dict], seconds: float) -> float | None:
    """Best accuracy among evaluations done within `seconds` of training."""
    seen = [e["test_acc"] for e in evals if e["train_time_s"] <= seconds]
    return max(seen) if seen else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def rank_summary(log: RankLog) -> dict:
    steps = log.of("step")
    n_steps = sum(e["steps_in_window"] for e in steps)
    data_s = sum(e["data_s"] for e in steps)
    compute_s = sum(e["compute_s"] for e in steps)
    comm_s = sum(e["comm_s"] for e in steps)
    busy = data_s + compute_s + comm_s
    batch = log.first("run_start")["experiment"]["batch_size"]
    gpu = [e["gpu"] for e in steps if e.get("gpu") and "temp_c" in e["gpu"]]
    end = log.first("train_end")

    def per_step_ms(seconds: float) -> float | None:
        return 1000 * seconds / n_steps if n_steps else None

    return {
        "machine": log.machine,
        "rank": log.rank,
        "steps": n_steps,
        "samples_per_s": n_steps * batch / busy if busy else None,
        "step_ms": per_step_ms(busy),
        "data_ms": per_step_ms(data_s),
        "compute_ms": per_step_ms(compute_s),
        "comm_ms": per_step_ms(comm_s),
        "comm_share": comm_s / busy if busy else None,
        "comm_mb": sum(e["comm_bytes"] for e in steps) / 2**20,
        "peak_mem_mb": (end.get("peak_memory_bytes") or 0) / 2**20 or None,
        "gpu_temp_max_c": max((g["temp_c"] for g in gpu), default=None),
        "gpu_sm_clock_min_mhz": min((g["sm_clock_mhz"] for g in gpu), default=None),
        "gpu_sm_clock_median_mhz": statistics.median(g["sm_clock_mhz"] for g in gpu) if gpu else None,
        "gpu_util_mean_pct": statistics.mean(g["util_pct"] for g in gpu) if gpu else None,
    }


def run_summary(run: TrainingRun, levels: list[float], minutes: list[float]) -> dict:
    root = run.root
    spec = run.spec
    end = root.first("train_end")
    start = root.first("train_start")
    evals = root.of("eval")
    steps = end.get("steps") or 0
    train_time = end.get("train_time_s")
    world = len(spec["workers"])
    params = start.get("model_params")
    is_local = spec["mode"] == "localsgd"
    row = {
        "run_id": run.run_id,
        "mode": spec["mode"],
        "workers": "+".join(spec["workers"]),
        "world_size": world,
        "model": spec["model"],
        "local_steps": spec["local_steps"] if is_local else None,
        "outer_optimizer": spec.get("outer_optimizer", "avg") if is_local else None,
        "outer_momentum": spec.get("outer_momentum", 0.0) if is_local else None,
        "comm_dtype": spec.get("comm_dtype", "fp32"),
        "amp": spec.get("amp", False),
        "batch_per_worker": spec["batch_size"],
        "global_batch": spec["batch_size"] * world,
        "effective_lr": start.get("base_lr"),
        "stop_reason": end.get("stop_reason", "INCOMPLETE"),
        "error": bool(root.of("error")),
        "epochs": end.get("epochs"),
        "steps": steps,
        "sync_rounds": end.get("sync_rounds"),
        "train_time_s": train_time,
        "best_acc": max((e["test_acc"] for e in evals), default=None),
        "final_acc": evals[-1]["test_acc"] if evals else None,
        # Throughput of the whole group: every worker runs the same number of steps.
        "throughput_samples_per_s": ratio(world * steps * spec["batch_size"], train_time),
        "step_ms": ratio(1000 * train_time if train_time else None, steps),
        "model_params": params,
    }
    comm_mb = rank_summary(root)["comm_mb"]
    if spec["mode"] == "ddp" and params:
        # DDP does not report its traffic; estimate the gradient volume fed to all-reduce.
        row["comm_mb_rank0"] = steps * params * 4 / 2**20
        row["comm_mb_is_estimate"] = True
    else:
        row["comm_mb_rank0"] = comm_mb
        row["comm_mb_is_estimate"] = False
    for level in levels:
        row[f"time_to_{level:g}_s"] = time_to_accuracy(evals, level)
    for m in minutes:
        row[f"acc_at_{m:g}min"] = accuracy_at(evals, m * 60)
    return row


def add_comparisons(rows: list[dict], levels: list[float], baseline_id: str | None) -> dict | None:
    """Adds speed-up against a baseline single run, scaling efficiency against the single
    runs of the same machines, and the estimated synchronisation overhead per step."""
    singles = [r for r in rows if r["mode"] == "single" and r["throughput_samples_per_s"]]
    if baseline_id:
        matches = [r for r in rows if baseline_id in r["run_id"]]
        baseline = matches[0] if matches else None
    else:
        baseline = max(singles, key=lambda r: r["throughput_samples_per_s"], default=None)
    # Latest single run per machine, as the reference for that machine alone.
    single_by_machine = {}
    for r in sorted(singles, key=lambda r: r["run_id"]):
        single_by_machine[r["workers"]] = r

    for r in rows:
        r["baseline_run"] = baseline["run_id"] if baseline else None
        for level in levels:
            key = f"time_to_{level:g}_s"
            r[f"speedup_{level:g}"] = ratio(baseline[key], r[key]) if baseline else None
        r["scaling_efficiency"] = None
        r["sync_overhead_ms_per_step"] = None
        if r["world_size"] > 1:
            refs = [single_by_machine.get(m) for m in r["workers"].split("+")]
            if all(refs) and r["throughput_samples_per_s"]:
                r["scaling_efficiency"] = (r["throughput_samples_per_s"]
                                           / sum(x["throughput_samples_per_s"] for x in refs))
                # Lock-step training cannot beat the slowest machine; the rest is
                # communication, waiting and overhead. Indirect estimate.
                same_batch = [x for x in refs if x["batch_per_worker"] == r["batch_per_worker"]]
                if len(same_batch) == len(refs) and r["step_ms"]:
                    r["sync_overhead_ms_per_step"] = r["step_ms"] - max(x["step_ms"] for x in refs)
    return baseline


def network_rows(check: NetcheckRun) -> list[dict]:
    rtt, rate, path = {}, {}, {}
    allreduce = {}
    for log in check.ranks.values():
        for e in log.of("pairs"):
            rtt.update(e.get("latency_ms", {}))
            rate.update(e.get("mb_per_s", {}))
        for e in log.of("tailscale"):
            for peer, info in e.get("peers", {}).items():
                if info.get("available"):
                    relayed = info.get("relayed")
                    path[f"{log.machine}->{peer}"] = {True: "relay", False: "direct"}.get(relayed, "unknown")
        for e in log.of("allreduce"):
            allreduce.update(e.get("seconds_median", {}))
    rows = []
    for pair, mbps in sorted(rate.items()):
        a, _, b = pair.partition("->")
        rows.append({"netcheck": check.run_id, "kind": "pair", "item": pair,
                     "rtt_ms": rtt.get(f"{a}<->{b}", rtt.get(f"{b}<->{a}")),
                     "mb_per_s": mbps, "mbit_per_s": mbps * 8,
                     "path": path.get(pair, "unknown"), "allreduce_s": None})
    for size, seconds in allreduce.items():
        rows.append({"netcheck": check.run_id, "kind": "allreduce", "item": size, "rtt_ms": None,
                     "mb_per_s": None, "mbit_per_s": None, "path": None, "allreduce_s": seconds})
    return rows


# ---------------------------------------------------------------- output

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                 "^": r"\textasciicircum{}"}


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in text)


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "có" if value else "không"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def latex_table(rows: list[dict], columns: list[tuple[str, str]], caption: str, label: str) -> str:
    """columns: (key, header). Headers are LaTeX already; cell values are escaped."""
    spec = "l" + "r" * (len(columns) - 1)
    lines = [r"\begin{table}[H]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(h for _, h in columns) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(latex_escape(fmt(r.get(k))) for k, _ in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def run_label(r: dict) -> str:
    if r["mode"] == "single":
        return f"1 GPU ({r['workers']})"
    extra = f", H={r['local_steps']}" if r["mode"] == "localsgd" else ""
    return f"{r['mode']} x{r['world_size']}{extra}"


def make_figures(runs: list[TrainingRun], rows: list[dict], rank_rows: list[dict], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["Chưa cài matplotlib, bỏ qua đồ thị"]
    notes = []
    by_id = {r["run_id"]: r for r in rows}

    for xkey, xlabel, name in (("train_time_s", "Thời gian huấn luyện (phút, không tính đánh giá)",
                                "accuracy_vs_time.png"),
                               ("epoch", "Epoch", "accuracy_vs_epoch.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for run in runs:
            evals = run.root.of("eval")
            if not evals:
                continue
            r = by_id[run.run_id]
            xs = [e[xkey] / 60 if xkey == "train_time_s" else e["epoch"] + 1 for e in evals]
            single = r["mode"] == "single"
            ax.plot(xs, [e["test_acc"] for e in evals], marker="o", markersize=3,
                    linestyle="--" if single else "-", alpha=0.6 if single else 1.0,
                    label=run_label(r))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)

    if rank_rows:
        labels = [f"{run_label(by_id[r['run_id']])}\n{r['machine']}" for r in rank_rows]
        parts = [("data_ms", "Tải dữ liệu"), ("compute_ms", "Tính toán"),
                 ("comm_ms", "Giao tiếp + chờ (đo trực tiếp)")]
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(rank_rows)), 5))
        bottom = [0.0] * len(rank_rows)
        for key, label in parts:
            vals = [r[key] or 0.0 for r in rank_rows]
            ax.bar(range(len(rank_rows)), vals, bottom=bottom, label=label)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(rank_rows)), labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("ms mỗi bước")
        ax.margins(y=0.12)
        ax.set_title("DDP gộp giao tiếp vào phần tính toán; xem sync_overhead_ms_per_step", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "time_breakdown.png", dpi=150)
        plt.close(fig)

    series = []
    for run in runs:
        for log in run.ranks.values():
            pts = [(e["train_time_s"] / 60, e["gpu"]) for e in log.of("step")
                   if e.get("gpu") and "temp_c" in e["gpu"]]
            if pts:
                series.append((f"{run_label(by_id[run.run_id])} / {log.machine}", pts))
    if series:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for label, pts in series:
            ax1.plot([t for t, _ in pts], [g["temp_c"] for _, g in pts], label=label)
            ax2.plot([t for t, _ in pts], [g["sm_clock_mhz"] for _, g in pts], label=label)
        ax1.set_ylabel("Nhiệt độ GPU (°C)")
        ax2.set_ylabel("Xung nhịp SM (MHz)")
        ax2.set_xlabel("Thời gian huấn luyện (phút)")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        ax1.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "gpu_thermal.png", dpi=150)
        plt.close(fig)
    else:
        notes.append("Không có số liệu NVML (chạy trên CPU hoặc thiếu nvidia-ml-py): bỏ qua gpu_thermal.png")
    return notes


RUN_COLUMNS = [("label", "Cấu hình"), ("stop_reason", "Kết thúc"), ("epochs", "Epoch"),
               ("train_time_s", "Thời gian (s)"), ("best_acc", "Acc tốt nhất"),
               ("throughput_samples_per_s", "Mẫu/s (cả nhóm)"),
               ("scaling_efficiency", "Hiệu suất mở rộng"),
               ("sync_overhead_ms_per_step", r"Chi phí đồng bộ (ms/bước)"),
               ("sync_rounds", "Vòng đồng bộ"), ("comm_mb_rank0", "MB truyền (1 máy; DDP: ước lượng)")]
RANK_COLUMNS = [("label", "Cấu hình"), ("machine", "Máy"), ("samples_per_s", "Mẫu/s"),
                ("data_ms", "Dữ liệu (ms)"), ("compute_ms", "Tính toán (ms)"),
                ("comm_ms", "Giao tiếp + chờ (ms)"), ("peak_mem_mb", "VRAM đỉnh (MB)"),
                ("gpu_temp_max_c", r"Nhiệt độ max (\textdegree C)"),
                ("gpu_sm_clock_min_mhz", "Xung nhịp min (MHz)")]
NETWORK_COLUMNS = [("item", "Cặp máy / kích thước"), ("rtt_ms", "RTT (ms)"),
                   ("mbit_per_s", "Mbit/s"), ("path", "Đường đi"), ("allreduce_s", "All-reduce (s)")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("needle", nargs="?", default="", help="chỉ lấy thư mục có tên chứa chuỗi này")
    parser.add_argument("--acc", nargs="*", type=float, default=[],
                        help="các mức accuracy cần tính thời gian đạt, ví dụ 0.6 0.7 0.8")
    parser.add_argument("--at-minutes", nargs="*", type=float, default=[],
                        help="các mốc thời gian (phút) cần lấy accuracy, ví dụ 10 30")
    parser.add_argument("--baseline", default=None,
                        help="một phần run_id của run gốc; mặc định là run 1 GPU nhanh nhất")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None, help="mặc định: <results>/report")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    runs, netchecks = load_results(args.results, args.needle)
    if not runs and not netchecks:
        print(f"Không có kết quả nào trong {args.results} khớp '{args.needle}'")
        return 1
    out = args.out or args.results / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows = [run_summary(run, args.acc, args.at_minutes) for run in runs]
    baseline = add_comparisons(rows, args.acc, args.baseline)
    for r in rows:
        r["label"] = run_label(r)
    rank_rows = []
    for run, r in zip(runs, rows):
        for log in sorted(run.ranks.values(), key=lambda x: x.rank):
            rank_rows.append({"run_id": run.run_id, "label": r["label"], **rank_summary(log)})
    net_rows = [row for check in netchecks for row in network_rows(check)]

    write_csv(out / "runs.csv", rows)
    write_csv(out / "ranks.csv", rank_rows)
    write_csv(out / "network.csv", net_rows)
    level_cols = ([(f"time_to_{lv:g}_s", rf"T@{lv:g} (s)") for lv in args.acc]
                  + [(f"speedup_{lv:g}", rf"Tăng tốc @{lv:g}") for lv in args.acc]
                  + [(f"acc_at_{m:g}min", rf"Acc @{m:g} phút") for m in args.at_minutes])
    (out / "table_runs.tex").write_text(latex_table(
        rows, RUN_COLUMNS[:5] + level_cols + RUN_COLUMNS[5:],
        "So sánh các cấu hình huấn luyện", "tab:runs"), encoding="utf-8")
    (out / "table_ranks.tex").write_text(latex_table(
        rank_rows, RANK_COLUMNS, "Phân rã thời gian mỗi bước theo máy", "tab:ranks"), encoding="utf-8")
    if net_rows:
        (out / "table_network.tex").write_text(latex_table(
            net_rows, NETWORK_COLUMNS, "Kết quả đo mạng", "tab:network"), encoding="utf-8")
    notes = make_figures(runs, rows, rank_rows, out)

    print(f"Run gốc để tính tăng tốc: {baseline['run_id'] if baseline else 'không có run 1 GPU'}")
    for r in rows:
        extra = "".join(f" T@{lv:g}={fmt(r[f'time_to_{lv:g}_s'])}s x{fmt(r[f'speedup_{lv:g}'])}"
                        for lv in args.acc)
        print(f"{r['label']:<28} {r['stop_reason']:<15} acc={fmt(r['best_acc'])} "
              f"mẫu/s={fmt(r['throughput_samples_per_s'])} "
              f"hiệu suất={fmt(r['scaling_efficiency'])}{extra}")
    for row in net_rows:
        if row["kind"] == "pair":
            print(f"mạng {row['item']:<28} RTT={fmt(row['rtt_ms'])}ms {fmt(row['mbit_per_s'])}Mbit/s "
                  f"{row['path']}")
    for note in notes:
        print(note)
    print(f"Đã ghi bảng và đồ thị vào {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
