"""Runs every worker of an experiment as a local process, using the real CLI.

Intended for the sim-* CPU profiles: it exercises the same code path as the
real machines (rendezvous over TCP, gloo collectives, JSONL logs).

    python scripts/simulate.py -e smoke_dist --set mode=sync
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from gpubench.app.config_loader import load_experiment, parse_overrides  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--command", choices=("run", "netcheck"), default="run")
    args = parser.parse_args()

    spec = load_experiment(args.experiment, parse_overrides(args.set))
    run_id = f"{spec.name}-{spec.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    sets = [x for s in args.set for x in ("--set", s)]
    extra = ["--run-id", run_id] if args.command == "run" else []
    procs = []
    for name in spec.workers:
        cmd = [sys.executable, "-m", "gpubench", args.command, "--machine", name,
               "--experiment", args.experiment, *extra, *sets]
        procs.append((name, subprocess.Popen(cmd, cwd=PROJECT_DIR)))
    codes = {name: p.wait() for name, p in procs}
    print(f"run_id={run_id} exit codes: {codes}")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
