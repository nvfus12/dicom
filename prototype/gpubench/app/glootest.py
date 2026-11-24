"""`gpubench glootest`: which way of opening gloo connections works between these machines.

Each way runs in its own process, because a failing gloo connection on Windows aborts the
whole process with a native stack trace instead of raising. Run it on every worker with the
same --master-addr (rank 0's IPv4) and the same --port-base, a few seconds apart.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from ..adapters.comm import GlooComm, local_ipv4_toward
from ..control.protocol import log_excerpt
from ..domain.config import ConfigError, ExperimentSpec
from .config_loader import CONFIGS_DIR, PROJECT_DIR, load_machine

# (label, how the gloo device is chosen)
VARIANTS = (("theo-ten-card", "interface"), ("theo-ip", "ipv4"))
OK_MARK = "GLOO-OK"


def _hostname_addresses() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
    except OSError as e:
        return [f"(không tra được: {e})"]
    return sorted({info[4][0] for info in infos})


def run_glootest(machine_name: str, spec: ExperimentSpec, master_addr: str, port_base: int,
                 wait_s: int, configs_dir: Path = CONFIGS_DIR) -> int:
    if spec.world_size < 2:
        raise ConfigError("glootest cần thí nghiệm có ít nhất 2 workers")
    machine = load_machine(machine_name, configs_dir)
    if machine.name not in spec.workers:
        raise ConfigError(f"Máy '{machine.name}' không có trong workers {list(spec.workers)}")
    rank = spec.workers.index(machine.name)
    print(f"Máy {machine.name}: rank {rank}/{spec.world_size}, rank 0 tại {master_addr}")
    print(f"IPv4 dùng để đi tới rank 0: {local_ipv4_toward(master_addr, port_base)}")
    print(f"Tên card trong file máy (network_interface): {machine.network_interface}")
    print(f"Địa chỉ của tên máy {socket.gethostname()}: {', '.join(_hostname_addresses())}")
    print("Chạy lần lượt từng cách; mỗi cách chờ máy kia tối đa "
          f"{wait_s} giây. Hai máy phải chạy lệnh này gần như cùng lúc.\n", flush=True)

    results = []
    for index, (label, kind) in enumerate(VARIANTS):
        if kind == "interface" and not machine.network_interface:
            results.append((label, "BỎ QUA", "file máy không có network_interface"))
            continue
        port = port_base + index
        print(f"--- {label} (cổng {port}) ...", flush=True)
        argv = [sys.executable, "-m", "gpubench.app.glootest", "--child", kind,
                "--master-addr", master_addr, "--port", str(port), "--rank", str(rank),
                "--world", str(spec.world_size), "--wait", str(wait_s),
                "--interface", machine.network_interface or ""]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "USE_LIBUV": "0"}
        env.pop("GLOO_SOCKET_IFNAME", None)
        try:
            done = subprocess.run(argv, cwd=PROJECT_DIR, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=wait_s + 60)
            output, code = (done.stdout or "") + (done.stderr or ""), done.returncode
        except subprocess.TimeoutExpired as e:
            output, code = f"quá {wait_s + 60} giây không xong\n{e.stdout or ''}", -1
        ok = code == 0 and OK_MARK in output
        detail = next((line for line in output.splitlines() if OK_MARK in line), "") if ok \
            else log_excerpt(output, max_lines=8) or f"thoát với mã {code}"
        results.append((label, "ĐẠT" if ok else "LỖI", detail))
        print(f"    {'ĐẠT' if ok else 'LỖI'}", flush=True)

    print("\n=== Kết quả glootest ===")
    for label, verdict, detail in results:
        print(f"{label:<16}{verdict}")
        for line in detail.splitlines():
            print(f"    {line}")
    return 0 if any(verdict == "ĐẠT" for _, verdict, _ in results) else 1


def _child(args) -> int:
    """One connection attempt: build the gloo group the given way, all-reduce, report."""
    bind = None
    if args.child == "interface":
        os.environ["GLOO_SOCKET_IFNAME"] = args.interface
    else:
        bind = local_ipv4_toward(args.master_addr, args.port)
    print(f"cách {args.child}: gloo dùng {bind or 'card ' + args.interface}", flush=True)
    start = time.perf_counter()
    comm = GlooComm(args.master_addr, args.port, args.rank, args.world, args.wait, bind_addr=bind)
    try:
        import torch
        value = torch.tensor([float(args.rank + 1)])
        comm.all_reduce_mean_(value)
        expected = (args.world + 1) / 2
        greeting = comm.broadcast_object("xin chào từ rank 0")
        comm.barrier()
    finally:
        comm.close()
    if abs(value.item() - expected) > 1e-6 or greeting != "xin chào từ rank 0":
        print(f"Sai kết quả: all-reduce {value.item()} (cần {expected}), broadcast {greeting!r}")
        return 1
    print(f"{OK_MARK}: kết nối và all-reduce đúng sau {time.perf_counter() - start:.1f} giây", flush=True)
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--child", choices=[kind for _, kind in VARIANTS], required=True)
    p.add_argument("--master-addr", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--wait", type=int, required=True)
    p.add_argument("--interface", default="")
    return _child(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
