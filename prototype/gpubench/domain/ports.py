"""Interfaces the training core depends on. Adapters implement them.

The core imports torch (it is the computation itself) but never imports
torch.distributed, torchvision, YAML, NVML or the filesystem directly.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol

import torch


class Device(Protocol):
    torch_device: torch.device
    amp: bool

    def synchronize(self) -> None: ...
    def reset_peak_memory(self) -> None: ...
    def peak_memory_bytes(self) -> int | None: ...
    def describe(self) -> dict[str, Any]: ...


class Communicator(Protocol):
    rank: int
    world_size: int

    def all_reduce_mean_(self, tensor: torch.Tensor) -> None:
        """In-place average across workers."""
    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> None: ...
    def broadcast_object(self, obj: Any, src: int = 0) -> Any: ...
    def any_true(self, flag: bool) -> bool:
        """True on every worker if the flag is true on at least one. Acts as a barrier."""
    def barrier(self) -> None: ...
    def process_group(self) -> Any:
        """Native handle, only needed by torch DDP."""
    def close(self) -> None: ...


class Sampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class DatasetProvider(Protocol):
    num_classes: int

    def train_loader(self, rank: int, world_size: int, batch_size: int,
                     seed: int) -> tuple[Iterable, Sampler]: ...
    def test_loader(self, batch_size: int) -> Iterable: ...


class MetricsSink(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...
    def close(self) -> None: ...


class GpuMonitor(Protocol):
    def sample(self) -> dict[str, Any]: ...
