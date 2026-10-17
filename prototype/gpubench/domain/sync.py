"""Synchronisation strategies: how workers keep their models consistent.

single   : no communication.
ddp      : torch DistributedDataParallel (overlaps all-reduce with backward,
           so communication time is included in compute time).
sync     : all-reduce of flattened gradients after backward, timed separately.
localsgd : each worker trains alone, parameters are averaged every H steps.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .ports import Communicator, Device


class SyncStrategy:
    name = "single"

    def __init__(self, comm: Communicator, device: Device):
        self.comm = comm
        self.device = device
        self.comm_s = 0.0
        self.comm_bytes = 0

    def prepare(self, model: nn.Module) -> nn.Module:
        return model

    def unwrap(self, model: nn.Module) -> nn.Module:
        return model

    def after_backward(self, model: nn.Module) -> None:
        pass

    def after_step(self, model: nn.Module, step: int) -> None:
        pass

    def end_of_epoch(self, model: nn.Module) -> None:
        pass

    def pop_stats(self) -> tuple[float, int]:
        stats = (self.comm_s, self.comm_bytes)
        self.comm_s, self.comm_bytes = 0.0, 0
        return stats

    def _reduce_mean(self, tensors: list[torch.Tensor]) -> None:
        """Average a list of tensors in one collective and write results back."""
        if not tensors:
            return
        self.device.synchronize()
        t0 = time.perf_counter()
        flat = torch.cat([t.detach().reshape(-1).float() for t in tensors])
        self.comm.all_reduce_mean_(flat)
        offset = 0
        for t in tensors:
            n = t.numel()
            t.data.copy_(flat[offset:offset + n].view_as(t))
            offset += n
        self.device.synchronize()
        self.comm_s += time.perf_counter() - t0
        self.comm_bytes += flat.numel() * flat.element_size()

    def _broadcast_model(self, model: nn.Module) -> None:
        for t in list(model.parameters()) + list(model.buffers()):
            self.comm.broadcast_(t.data, src=0)


class DDPStrategy(SyncStrategy):
    name = "ddp"

    def prepare(self, model: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.device.torch_device] if self.device.torch_device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids,
                                       process_group=self.comm.process_group())

    def unwrap(self, model: nn.Module) -> nn.Module:
        return model.module


class GradAllReduceStrategy(SyncStrategy):
    name = "sync"

    def prepare(self, model: nn.Module) -> nn.Module:
        self._broadcast_model(model)
        return model

    def after_backward(self, model: nn.Module) -> None:
        self._reduce_mean([p.grad for p in model.parameters() if p.grad is not None])


class LocalSGDStrategy(SyncStrategy):
    """Averages parameters and floating-point buffers (BatchNorm statistics).

    Optimizer state (momentum) stays local to each worker.
    """

    name = "localsgd"

    def __init__(self, comm: Communicator, device: Device, local_steps: int):
        super().__init__(comm, device)
        self.local_steps = local_steps
        self._step = 0
        self._last_avg_step = 0

    def prepare(self, model: nn.Module) -> nn.Module:
        self._broadcast_model(model)
        return model

    def after_step(self, model: nn.Module, step: int) -> None:
        self._step = step
        if step % self.local_steps == 0:
            self._average(model)
            self._last_avg_step = step

    def end_of_epoch(self, model: nn.Module) -> None:
        # Evaluation needs one consistent model; average unless the last step just did.
        if self._step != self._last_avg_step:
            self._average(model)
            self._last_avg_step = self._step

    def _average(self, model: nn.Module) -> None:
        tensors = list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()]
        self._reduce_mean(tensors)


def build_strategy(mode: str, comm: Communicator, device: Device, local_steps: int = 1) -> SyncStrategy:
    if mode == "single":
        return SyncStrategy(comm, device)
    if mode == "ddp":
        return DDPStrategy(comm, device)
    if mode == "sync":
        return GradAllReduceStrategy(comm, device)
    if mode == "localsgd":
        return LocalSGDStrategy(comm, device, local_steps)
    raise ValueError(f"unknown mode {mode}")
