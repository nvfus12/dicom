"""Composition root: turns a machine profile + an experiment into a running trainer."""
from __future__ import annotations

import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from ..adapters.comm import GlooComm, LocalComm
from ..adapters.data import build_dataset
from ..adapters.device import build_device
from ..adapters.metrics import JsonlSink
from ..adapters.models import build_model
from ..adapters.monitor import build_monitor
from ..domain.config import ConfigError, ExperimentSpec, MachineProfile
from ..domain.sync import build_strategy
from ..domain.trainer import Trainer, TrainResult
from .config_loader import CONFIGS_DIR, RESULTS_DIR, load_machine


def rendezvous_address(spec: ExperimentSpec, configs_dir: Path) -> str:
    root = load_machine(spec.workers[0], configs_dir)
    if not root.address:
        raise ConfigError(f"Máy rank 0 '{root.name}' chưa có address trong "
                          f"configs/machines/{root.name}.yaml")
    return root.address


def apply_environment(machine: MachineProfile) -> None:
    if machine.network_interface:
        os.environ["GLOO_SOCKET_IFNAME"] = machine.network_interface
    for key, value in machine.env.items():
        os.environ[key] = str(value)


def run(machine_name: str, spec: ExperimentSpec, run_id: str | None = None,
        configs_dir: Path = CONFIGS_DIR, results_dir: Path = RESULTS_DIR,
        echo: bool = True) -> TrainResult:
    machine = load_machine(machine_name, configs_dir)
    if machine.name not in spec.workers:
        raise ConfigError(f"Máy '{machine.name}' không có trong workers {list(spec.workers)} "
                          f"của thí nghiệm '{spec.name}'")
    rank = spec.workers.index(machine.name)
    apply_environment(machine)

    if spec.world_size == 1:
        comm = LocalComm()
    else:
        master = rendezvous_address(spec, configs_dir)
        print(f"[{machine.name}] rank {rank}/{spec.world_size}, chờ các máy khác tại "
              f"{master}:{spec.master_port} ...", flush=True)
        comm = GlooComm(master, spec.master_port, rank, spec.world_size, spec.timeout_s)

    try:
        # Every worker must run the same experiment.
        root_fp = comm.broadcast_object(spec.fingerprint())
        if comm.any_true(root_fp != spec.fingerprint()):
            raise ConfigError("Cấu hình thí nghiệm khác nhau giữa các máy (fingerprint lệch). "
                              "Kiểm tra file experiment và các --set trên mọi máy.")

        # Local setup can fail on one machine; decide together so nobody hangs.
        error = None
        try:
            device = build_device(machine)
            limit = machine.max_batch.get(spec.model)
            if limit is not None and spec.batch_size > limit:
                raise ConfigError(f"batch_size={spec.batch_size} vượt max_batch[{spec.model}]={limit} "
                                  f"của máy '{machine.name}'")
            data = build_dataset(spec, machine)
        except Exception as e:  # reported below, after the collective check
            error = e
        if comm.any_true(error is not None):
            if error is not None:
                raise error
            raise ConfigError("Một máy khác báo lỗi khi khởi tạo; xem thông báo trên máy đó.")

        run_id = comm.broadcast_object(run_id or f"{spec.name}-{time.strftime('%Y%m%d-%H%M%S')}")
        sink = JsonlSink(results_dir / run_id / f"rank{rank}-{machine.name}.jsonl",
                         echo=echo and rank == 0)
        try:
            sink.emit("run_start", run_id=run_id, rank=rank, world_size=spec.world_size,
                      experiment=asdict(spec), fingerprint=spec.fingerprint(),
                      machine=asdict(machine), device=device.describe(),
                      host=socket.gethostname(), platform=platform.platform(),
                      python=sys.version.split()[0], torch=torch.__version__)

            torch.manual_seed(spec.seed)          # identical initial weights everywhere
            model = build_model(spec.model, data.num_classes).to(device.torch_device)
            torch.manual_seed(spec.seed + rank)   # different augmentation per worker

            strategy = build_strategy(spec.mode, comm, device, spec.local_steps)
            trainer = Trainer(spec, model, data, device, comm, strategy, sink,
                              build_monitor(machine.device, machine.cuda_index))
            result = trainer.run()
            if rank == 0:
                print(f"[{machine.name}] xong: {result}", flush=True)
            return result
        except Exception:
            sink.emit("error", traceback=traceback.format_exc())
            raise
        finally:
            sink.close()
    finally:
        comm.close()
