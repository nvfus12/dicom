"""Synchronisation strategies: how workers keep their models consistent.

single   : no communication.
ddp      : torch DistributedDataParallel (overlaps all-reduce with backward,
           so communication time is included in compute time).
sync     : all-reduce of flattened gradients after backward, timed separately.
localsgd : each worker trains alone; every H steps the workers average their
           pseudo-gradients (global - local) and apply an outer step:
             avg      -> plain parameter averaging (FedAvg)
             nesterov -> SGD with Nesterov momentum on the pseudo-gradient (DiLoCo)

comm_dtype sets the precision of what goes over the network (fp32/fp16/bf16);
all arithmetic on the model stays in fp32.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .ports import Communicator, Device

COMM_TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def outer_update_(global_params: torch.Tensor, pseudo_grad: torch.Tensor,
                  momentum_buf: torch.Tensor | None, lr: float, momentum: float) -> None:
    """In-place outer step: SGD with Nesterov momentum, as in DiLoCo.

    With lr=1 and momentum=0 it reduces to global -= pseudo_grad, i.e. the new
    global model is exactly the average of the workers' models.
    """
    if momentum > 0:
        momentum_buf.mul_(momentum).add_(pseudo_grad)
        step = pseudo_grad.add(momentum_buf, alpha=momentum)
    else:
        step = pseudo_grad
    global_params.sub_(step, alpha=lr)


class SyncStrategy:
    name = "single"

    def __init__(self, comm: Communicator, device: Device, comm_dtype: str = "fp32"):
        self.comm = comm
        self.device = device
        self.comm_dtype = COMM_TORCH_DTYPES[comm_dtype]
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

    def _all_reduce_mean(self, flat: torch.Tensor) -> torch.Tensor:
        """Averages a flat fp32 tensor across workers, sent in comm_dtype; returns fp32."""
        self.device.synchronize()
        t0 = time.perf_counter()
        wire = flat if flat.dtype == self.comm_dtype else flat.to(self.comm_dtype)
        self.comm.all_reduce_mean_(wire)
        out = wire.float()
        self.device.synchronize()
        self.comm_s += time.perf_counter() - t0
        self.comm_bytes += wire.numel() * wire.element_size()
        return out

    def _reduce_mean(self, tensors: list[torch.Tensor]) -> None:
        """Average a list of tensors in one collective and write results back."""
        if not tensors:
            return
        flat = self._all_reduce_mean(_flatten(tensors))
        _unflatten_into(flat, tensors)

    def _broadcast_model(self, model: nn.Module) -> None:
        for t in list(model.parameters()) + list(model.buffers()):
            self.comm.broadcast_(t.data, src=0)


class DDPStrategy(SyncStrategy):
    name = "ddp"

    def prepare(self, model: nn.Module) -> nn.Module:
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.device.torch_device] if self.device.torch_device.type == "cuda" else None
        ddp = DistributedDataParallel(model, device_ids=device_ids,
                                      process_group=self.comm.process_group())
        if self.comm_dtype != torch.float32:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

            hook = (default_hooks.fp16_compress_hook if self.comm_dtype == torch.float16
                    else default_hooks.bf16_compress_hook)
            ddp.register_comm_hook(self.comm.process_group(), hook)
        return ddp

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
    """Local SGD with an outer optimizer (avg = FedAvg, nesterov = DiLoCo).

    Keeps an fp32 copy of the global model. At each synchronisation every worker
    sends its pseudo-gradient (global - local); the averaged pseudo-gradient
    drives the outer step on parameters. Floating-point buffers (BatchNorm
    statistics) are always plainly averaged, without momentum. Inner optimizer
    state (momentum) stays local to each worker.
    """

    name = "localsgd"

    def __init__(self, comm: Communicator, device: Device, local_steps: int,
                 outer_optimizer: str = "avg", outer_lr: float = 1.0,
                 outer_momentum: float = 0.0, comm_dtype: str = "fp32"):
        super().__init__(comm, device, comm_dtype)
        self.local_steps = local_steps
        self.outer_lr = outer_lr if outer_optimizer == "nesterov" else 1.0
        self.outer_momentum = outer_momentum if outer_optimizer == "nesterov" else 0.0
        self.rounds = 0
        self._step = 0
        self._last_sync_step = 0

    def prepare(self, model: nn.Module) -> nn.Module:
        self._broadcast_model(model)
        self._params = list(model.parameters())
        self._buffers = [b for b in model.buffers() if b.is_floating_point()]
        self._n_params = sum(p.numel() for p in self._params)
        self._global = _flatten(self._params + self._buffers).clone()
        self._momentum = (torch.zeros(self._n_params, device=self._global.device)
                          if self.outer_momentum > 0 else None)
        return model

    def after_step(self, model: nn.Module, step: int) -> None:
        self._step = step
        if step % self.local_steps == 0:
            self._sync()
            self._last_sync_step = step

    def end_of_epoch(self, model: nn.Module) -> None:
        # Evaluation needs one consistent model; sync unless the last step just did.
        if self._step != self._last_sync_step:
            self._sync()
            self._last_sync_step = self._step

    @torch.no_grad()
    def _sync(self) -> None:
        tensors = self._params + self._buffers
        pseudo_grad = self._all_reduce_mean(self._global - _flatten(tensors))
        n = self._n_params
        outer_update_(self._global[:n], pseudo_grad[:n], self._momentum,
                      self.outer_lr, self.outer_momentum)
        self._global[n:].sub_(pseudo_grad[n:])
        _unflatten_into(self._global, tensors)
        self.rounds += 1


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float() for t in tensors])


def _unflatten_into(flat: torch.Tensor, tensors: list[torch.Tensor]) -> None:
    offset = 0
    for t in tensors:
        n = t.numel()
        t.data.copy_(flat[offset:offset + n].view_as(t))
        offset += n


def build_strategy(mode: str, comm: Communicator, device: Device, local_steps: int = 1,
                   outer_optimizer: str = "avg", outer_lr: float = 1.0,
                   outer_momentum: float = 0.0, comm_dtype: str = "fp32") -> SyncStrategy:
    if mode == "single":
        return SyncStrategy(comm, device)
    if mode == "ddp":
        return DDPStrategy(comm, device, comm_dtype)
    if mode == "sync":
        return GradAllReduceStrategy(comm, device, comm_dtype)
    if mode == "localsgd":
        return LocalSGDStrategy(comm, device, local_steps, outer_optimizer, outer_lr,
                                outer_momentum, comm_dtype)
    raise ValueError(f"unknown mode {mode}")
