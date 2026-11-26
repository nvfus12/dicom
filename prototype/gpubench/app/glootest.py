"""`gpubench glootest`: which way of opening gloo connections works between these machines.

Each way runs in its own process, because a failing gloo connection on Windows aborts the
whole process with a native stack trace instead of raising. Run it on every worker with the
same --master-addr (rank 0's IPv4) and the same --port-base, a few seconds apart.
"""
from __future__ import annotations

import argparse
import json
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


def child_env() -> dict:
    """Environment of one attempt: PyTorch's detailed distributed logging on, so a crash leaves
    its last messages in the output; each way sets GLOO_SOCKET_IFNAME itself."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "USE_LIBUV": "0",
           "TORCH_CPP_LOG_LEVEL": "INFO", "TORCH_DISTRIBUTED_DEBUG": "DETAIL"}
    env.pop("GLOO_SOCKET_IFNAME", None)
    return env


def _attempt(kind: str, master: str, port: int, ranks: list[int], world: int, wait_s: int,
             interface: str) -> tuple[bool, str]:
    """Runs one way for the given ranks of this machine, each in its own process."""
    procs = []
    for rank in ranks:
        argv = [sys.executable, "-m", "gpubench.app.glootest", "--child", kind,
                "--master-addr", master, "--port", str(port), "--rank", str(rank),
                "--world", str(world), "--wait", str(wait_s), "--interface", interface]
        procs.append((rank, subprocess.Popen(argv, cwd=PROJECT_DIR, env=child_env(),
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace")))
    ok, outputs = True, []
    for rank, proc in procs:
        try:
            output = proc.communicate(timeout=wait_s + 60)[0] or ""
        except subprocess.TimeoutExpired:
            proc.kill()
            output = f"quá {wait_s + 60} giây không xong\n" + (proc.communicate()[0] or "")
        ok = ok and proc.returncode == 0 and OK_MARK in output
        prefix = f"[rank {rank}] " if len(ranks) > 1 else ""
        outputs.append("\n".join(prefix + line for line in output.splitlines()))
    return ok, "\n".join(outputs)


def run_glootest(machine_name: str, spec: ExperimentSpec, master_addr: str | None, port_base: int,
                 wait_s: int, configs_dir: Path = CONFIGS_DIR, self_test: bool = False,
                 toward: str = "192.168.0.1", json_out: Path | None = None) -> int:
    machine = load_machine(machine_name, configs_dir)
    print(f"Tên card trong file máy (network_interface): {machine.network_interface}")
    print(f"Địa chỉ của tên máy {socket.gethostname()}: {', '.join(_hostname_addresses())}")
    if self_test:
        master = local_ipv4_toward(toward, port_base)
        ranks, world = [0, 1], 2
        print(f"Tự thử trên máy {machine.name}: hai tiến trình trên chính máy này nói chuyện qua "
              f"IPv4 {master} (không cần máy kia).\n", flush=True)
    else:
        if not master_addr:
            raise ConfigError("Thiếu --master-addr (IPv4 của máy rank 0). Hoặc dùng --self để "
                              "tự thử trên một máy.")
        if spec.world_size < 2:
            raise ConfigError("glootest cần thí nghiệm có ít nhất 2 workers")
        if machine.name not in spec.workers:
            raise ConfigError(f"Máy '{machine.name}' không có trong workers {list(spec.workers)}")
        master, ranks, world = master_addr, [spec.workers.index(machine.name)], spec.world_size
        print(f"Máy {machine.name}: rank {ranks[0]}/{world}, rank 0 tại {master}")
        print(f"IPv4 dùng để đi tới rank 0: {local_ipv4_toward(master, port_base)}")
        print(f"Chạy lần lượt từng cách; mỗi cách chờ máy kia tối đa {wait_s} giây. "
              f"Hai máy phải chạy lệnh này gần như cùng lúc.\n", flush=True)

    results = []
    for index, (label, kind) in enumerate(VARIANTS):
        if kind == "interface" and not machine.network_interface:
            results.append((label, "BỎ QUA", "file máy không có network_interface"))
            continue
        port = port_base + index
        print(f"--- {label} (cổng {port}) ...", flush=True)
        ok, output = _attempt(kind, master, port, ranks, world, wait_s,
                              machine.network_interface or "")
        if ok:
            detail = "\n".join(line for line in output.splitlines() if OK_MARK in line)
        else:
            detail = log_excerpt(output, max_lines=14) or "không có thông báo"
        results.append((label, "ĐẠT" if ok else "LỖI", detail))
        print(f"    {'ĐẠT' if ok else 'LỖI'}", flush=True)

    print("\n=== Kết quả glootest" + (" (tự thử trên một máy)" if self_test else "") + " ===")
    for label, verdict, detail in results:
        print(f"{label:<16}{verdict}")
        for line in detail.splitlines():
            print(f"    {line}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "machine": machine.name, "self_test": self_test, "master": master,
            "network_interface": machine.network_interface,
            "hostname_addresses": _hostname_addresses(),
            "variants": {label: {"verdict": verdict, "detail": detail}
                         for label, verdict, detail in results},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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
