"""Training loop. Depends only on ports and the sync strategy."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .config import ExperimentSpec
from .ports import Communicator, DatasetProvider, Device, GpuMonitor, MetricsSink
from .sync import SyncStrategy


@dataclass
class TrainResult:
    stop_reason: str
    steps: int
    epochs: int
    train_time_s: float          # training only, excludes evaluation pauses
    best_acc: float | None       # known on rank 0 only
    time_to_target_s: float | None


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: Device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device.torch_device, non_blocking=True)
        y = y.to(device.torch_device, non_blocking=True)
        with torch.autocast(device.torch_device.type, enabled=device.amp):
            out = model(x)
        loss_sum += criterion(out.float(), y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total, loss_sum / total


class Trainer:
    def __init__(self, spec: ExperimentSpec, model: nn.Module, data: DatasetProvider,
                 device: Device, comm: Communicator, strategy: SyncStrategy,
                 sink: MetricsSink, monitor: GpuMonitor,
                 clock: Callable[[], float] = time.perf_counter):
        self.spec = spec
        self.model = model
        self.data = data
        self.device = device
        self.comm = comm
        self.strategy = strategy
        self.sink = sink
        self.monitor = monitor
        self.clock = clock

    def run(self) -> TrainResult:
        spec, device, comm = self.spec, self.device, self.comm
        is_root = comm.rank == 0

        model = self.strategy.prepare(self.model)
        train_loader, sampler = self.data.train_loader(comm.rank, comm.world_size,
                                                       spec.batch_size, spec.seed)
        test_loader = self.data.test_loader(256) if is_root else None
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise RuntimeError("Shard dữ liệu nhỏ hơn 1 batch; giảm batch_size hoặc tăng dữ liệu")
        total_steps = steps_per_epoch * spec.max_epochs
        warmup_steps = int(spec.warmup_epochs * steps_per_epoch)
        base_lr = spec.effective_lr()

        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=spec.momentum,
                                    weight_decay=spec.weight_decay, nesterov=True)
        scaler = torch.amp.GradScaler(device.torch_device.type, enabled=device.amp)
        criterion = nn.CrossEntropyLoss()

        self.sink.emit("train_start", steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                       base_lr=base_lr, warmup_steps=warmup_steps,
                       model_params=sum(p.numel() for p in self.model.parameters()),
                       batch_size=spec.batch_size)
        device.reset_peak_memory()
        model.train()

        step = 0
        train_time = 0.0
        best_acc: float | None = None
        time_to_target: float | None = None
        stop_reason = "max_epochs"
        window = _Window()

        for epoch in range(spec.max_epochs):
            sampler.set_epoch(epoch)
            batches = iter(train_loader)
            for _ in range(steps_per_epoch):
                t0 = self.clock()
                x, y = next(batches)
                x = x.to(device.torch_device, non_blocking=True)
                y = y.to(device.torch_device, non_blocking=True)
                t1 = self.clock()

                lr = lr_at(step, total_steps, warmup_steps, base_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.autocast(device.torch_device.type, enabled=device.amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                self.strategy.after_backward(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                self.strategy.after_step(model, step)
                device.synchronize()
                t2 = self.clock()

                comm_s, comm_bytes = self.strategy.pop_stats()
                train_time += t2 - t0
                window.add(loss.item(), y.numel(), t1 - t0, t2 - t1 - comm_s, comm_s, comm_bytes)

                if step % spec.log_every_steps == 0:
                    self.sink.emit("step", step=step, epoch=epoch, lr=lr, train_time_s=train_time,
                                   **window.flush(), gpu=self.monitor.sample())

            epochs_done = epoch + 1
            reached = False
            if epochs_done % spec.eval_every_epochs == 0 or epochs_done == spec.max_epochs:
                if is_root:
                    t_eval = self.clock()
                    with self.strategy.eval_model(model) as eval_model:
                        acc, test_loss = evaluate(eval_model, test_loader, device)
                    best_acc = acc if best_acc is None else max(best_acc, acc)
                    reached = spec.target_acc is not None and acc >= spec.target_acc
                    if reached and time_to_target is None:
                        time_to_target = train_time
                    self.sink.emit("eval", step=step, epoch=epoch, train_time_s=train_time,
                                   test_acc=acc, test_loss=test_loss,
                                   eval_s=self.clock() - t_eval)
            over_time = (spec.max_train_minutes is not None
                         and train_time >= spec.max_train_minutes * 60)
            # Collective decision: keeps every worker in lock-step and doubles as a barrier.
            if comm.any_true(reached):
                stop_reason = "target_reached"
                break
            if comm.any_true(over_time):
                stop_reason = "time_limit"
                break

        if window.steps:
            self.sink.emit("step", step=step, epoch=epoch, train_time_s=train_time,
                           **window.flush(), gpu=self.monitor.sample())
        result = TrainResult(stop_reason, step, epochs_done, train_time, best_acc, time_to_target)
        self.sink.emit("train_end", stop_reason=stop_reason, steps=step, epochs=epochs_done,
                       sync_rounds=getattr(self.strategy, "rounds", None),
                       train_time_s=train_time, best_acc=best_acc,
                       time_to_target_s=time_to_target,
                       peak_memory_bytes=device.peak_memory_bytes())
        return result


class _Window:
    """Accumulates per-step timings between two log lines."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.steps = 0
        self.samples = 0
        self.loss_sum = 0.0
        self.data_s = self.compute_s = self.comm_s = 0.0
        self.comm_bytes = 0

    def add(self, loss, samples, data_s, compute_s, comm_s, comm_bytes):
        self.steps += 1
        self.samples += samples
        self.loss_sum += loss
        self.data_s += data_s
        self.compute_s += compute_s
        self.add_comm(comm_s, comm_bytes)

    def add_comm(self, comm_s, comm_bytes):
        self.comm_s += comm_s
        self.comm_bytes += comm_bytes

    def flush(self) -> dict:
        busy = self.data_s + self.compute_s + self.comm_s
        out = dict(steps_in_window=self.steps,
                   loss=self.loss_sum / max(1, self.steps),
                   data_s=self.data_s, compute_s=self.compute_s, comm_s=self.comm_s,
                   comm_bytes=self.comm_bytes,
                   samples_per_s=self.samples / busy if busy > 0 else None)
        self._reset()
        return out
