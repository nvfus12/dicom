"""Network check between the workers of an experiment, measured with the same
transport (gloo) that training will use.

Measures, per machine pair: round-trip latency and one-way throughput in both
directions; for the whole group: all-reduce time at the sizes of the models.
Optionally asks the Tailscale CLI whether each peer is reached directly or
through a relay (a relay is much slower and changes the benchmark).
"""
from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import torch

from ..adapters.comm import GlooComm
from ..adapters.metrics import JsonlSink
from ..domain.config import ConfigError, ExperimentSpec
from .bootstrap import apply_environment, rendezvous_address
from .config_loader import CONFIGS_DIR, RESULTS_DIR, load_machine

MB = 2**20
PING_ROUNDS = 20
# Float32 sizes of the CIFAR ResNet-20 (0.27M params) and ResNet-18 (11.2M params).
ALLREDUCE_SIZES_MB = (1.1, 8.0, 44.7)


def _tensor(size_mb: float) -> torch.Tensor:
    return torch.zeros(max(1, int(size_mb * MB / 4)), dtype=torch.float32)


def measure_pair_latency(comm, a: int, b: int, rounds: int = PING_ROUNDS) -> float | None:
    """Median round-trip time in ms; result is only meaningful on rank `a`."""
    ping = torch.zeros(1)
    samples = []
    for i in range(rounds + 2):                     # first 2 rounds are warm-up
        if comm.rank == a:
            t0 = time.perf_counter()
            comm.send(ping, b)
            comm.recv(ping, b)
            if i >= 2:
                samples.append((time.perf_counter() - t0) * 1000)
        elif comm.rank == b:
            comm.recv(ping, a)
            comm.send(ping, a)
    return statistics.median(samples) if samples else None


def measure_pair_throughput(comm, a: int, b: int, size_mb: float = 8.0,
                            repeats: int = 3) -> float | None:
    """MB/s from a to b, until b has acknowledged; meaningful on rank `a`."""
    payload, ack = _tensor(size_mb), torch.zeros(1)
    rates = []
    for i in range(repeats + 1):                    # first transfer is warm-up
        if comm.rank == a:
            t0 = time.perf_counter()
            comm.send(payload, b)
            comm.recv(ack, b)
            if i >= 1:
                rates.append(size_mb / (time.perf_counter() - t0))
        elif comm.rank == b:
            comm.recv(payload, a)
            comm.send(ack, a)
    return statistics.median(rates) if rates else None


def measure_allreduce(comm, size_mb: float, repeats: int = 3) -> float:
    """Median seconds for one all-reduce of this size across all workers."""
    t = _tensor(size_mb)
    times = []
    for i in range(repeats + 1):                    # first one is warm-up
        comm.barrier()
        t0 = time.perf_counter()
        comm.all_reduce_mean_(t)
        if i >= 1:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def tailscale_path(address: str) -> dict:
    """Asks the Tailscale CLI how this machine reaches `address`. Never raises."""
    exe = shutil.which("tailscale")
    if exe is None:
        return {"available": False}
    try:
        out = subprocess.run([exe, "ping", "--c", "3", address], capture_output=True, text=True,
                             timeout=30)
        text = (out.stdout + out.stderr).strip()
    except Exception as e:  # CLI present but failed or timed out
        return {"available": True, "error": str(e)}
    return {"available": True, **parse_tailscale_ping(text)}


def parse_tailscale_ping(text: str) -> dict:
    """Reads the output of `tailscale ping`. Lines look like:
        pong from host (100.x.y.z) via DERP(sin) in 120ms          (relayed)
        pong from host (100.x.y.z) via 203.0.113.5:41641 in 30ms   (direct)
    The last pong wins: Tailscale often starts relayed and then upgrades to direct.
    """
    via = re.findall(r"\bvia ([^\s]+)", text)
    return {"relayed": via[-1].startswith("DERP") if via else None,
            "via": via[-1] if via else None, "raw": text[-300:]}


def run_netcheck(machine_name: str, spec: ExperimentSpec, master_addr: str | None = None,
                 configs_dir: Path = CONFIGS_DIR,
                 results_dir: Path = RESULTS_DIR) -> dict:
    if spec.world_size < 2:
        raise ConfigError("netcheck cần thí nghiệm có ít nhất 2 workers")
    machine = load_machine(machine_name, configs_dir)
    if machine.name not in spec.workers:
        raise ConfigError(f"Máy '{machine.name}' không có trong workers {list(spec.workers)}")
    rank = spec.workers.index(machine.name)
    apply_environment(machine)
    master = rendezvous_address(spec, configs_dir, master_addr)
    print(f"[{machine.name}] rank {rank}/{spec.world_size}, chờ các máy khác tại "
          f"{master}:{spec.master_port} ...", flush=True)
    comm = GlooComm(master, spec.master_port, rank, spec.world_size, spec.timeout_s)
    sink = None
    try:
        run_id = comm.broadcast_object(f"netcheck-{time.strftime('%Y%m%d-%H%M%S')}")
        sink = JsonlSink(results_dir / run_id / f"rank{rank}-{machine.name}.jsonl")

        peers = {}
        for other_rank, name in enumerate(spec.workers):
            if other_rank != rank:
                try:
                    address = load_machine(name, configs_dir).address
                except ConfigError:     # peer's file not copied to this machine
                    address = None
                peers[name] = tailscale_path(address) if address else {"available": False}
        sink.emit("tailscale", machine=machine.name, peers=peers)

        pairs = [(a, b) for a in range(spec.world_size) for b in range(spec.world_size) if a != b]
        mine = {"latency_ms": {}, "mb_per_s": {}}
        for a in range(spec.world_size):
            for b in range(a + 1, spec.world_size):
                rtt = measure_pair_latency(comm, a, b)
                if rtt is not None:
                    mine["latency_ms"][f"{spec.workers[a]}<->{spec.workers[b]}"] = round(rtt, 2)
                comm.barrier()   # one pair at a time, so measurements do not share the link
        for a, b in pairs:
            rate = measure_pair_throughput(comm, a, b)
            if rate is not None:
                mine["mb_per_s"][f"{spec.workers[a]}->{spec.workers[b]}"] = round(rate, 2)
            comm.barrier()
        sink.emit("pairs", machine=machine.name, **mine)

        allreduce = {f"{s}MB": round(measure_allreduce(comm, s), 4) for s in ALLREDUCE_SIZES_MB}
        sink.emit("allreduce", seconds_median=allreduce)

        everyone = comm.all_gather_object({"machine": machine.name, "pairs": mine, "tailscale": peers})
        report = {"run_id": run_id, "workers": list(spec.workers), "allreduce_s": allreduce,
                  "nodes": everyone}
        if rank == 0:
            _print_report(report)
        return report
    finally:
        if sink:
            sink.close()
        comm.close()


def _print_report(report: dict) -> None:
    print("\n=== Kết quả kiểm tra mạng ===")
    print("Độ trễ khứ hồi (ms):")
    rtts, rates = {}, {}
    for node in report["nodes"]:
        rtts.update(node["pairs"]["latency_ms"])
        rates.update(node["pairs"]["mb_per_s"])
    for k, v in sorted(rtts.items()):
        print(f"  {k:<32} {v:>9.2f}")
    print("Thông lượng một chiều (MB/s, đo đến khi bên nhận xác nhận):")
    for k, v in sorted(rates.items()):
        print(f"  {k:<32} {v:>9.2f}  (~{v * 8:.0f} Mbit/s)")
    print("Thời gian một lần all-reduce trên cả nhóm (giây):")
    for size, sec in report["allreduce_s"].items():
        print(f"  {size:<10} {sec:>9.4f}")
    print("  (1.1MB ~ ResNet-20, 44.7MB ~ ResNet-18; DDP phải trả mức này MỖI BƯỚC, "
          "local SGD chỉ trả mỗi H bước)")
    print("Đường đi theo Tailscale (nếu có):")
    any_info = False
    for node in report["nodes"]:
        for peer, info in node["tailscale"].items():
            if info.get("available"):
                any_info = True
                state = {True: "QUA RELAY (chậm, làm lệch benchmark)", False: "trực tiếp",
                         None: "không xác định"}[info.get("relayed")]
                print(f"  {node['machine']} -> {peer}: {state} via {info.get('via')}")
    if not any_info:
        print("  (không có lệnh tailscale trên máy rank 0; kiểm tra tay bằng `tailscale status`)")
