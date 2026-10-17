"""Communication adapters: torch.distributed (gloo) and a single-process stand-in."""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


class LocalComm:
    """World of one: every collective is a no-op."""

    rank = 0
    world_size = 1

    def all_reduce_mean_(self, tensor: torch.Tensor) -> None:
        pass

    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> None:
        pass

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj

    def any_true(self, flag: bool) -> bool:
        return flag

    def barrier(self) -> None:
        pass

    def process_group(self) -> Any:
        return None

    def close(self) -> None:
        pass


class GlooComm:
    """torch.distributed with the gloo backend (the only multi-node backend on Windows).

    Collectives run on CPU tensors: CUDA tensors are copied to host explicitly so
    the code does not depend on gloo being built with CUDA support.
    """

    def __init__(self, master_addr: str, master_port: int, rank: int, world_size: int,
                 timeout_s: int):
        if sys.platform == "win32":
            # Windows wheels of PyTorch may lack libuv for TCPStore.
            os.environ.setdefault("USE_LIBUV", "0")
        dist.init_process_group(backend="gloo",
                                init_method=f"tcp://{master_addr}:{master_port}",
                                rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=timeout_s))
        self.rank = rank
        self.world_size = world_size

    def all_reduce_mean_(self, tensor: torch.Tensor) -> None:
        host = tensor.cpu() if tensor.is_cuda else tensor
        dist.all_reduce(host, op=dist.ReduceOp.SUM)
        host.div_(self.world_size)
        if host is not tensor:
            tensor.copy_(host)

    def broadcast_(self, tensor: torch.Tensor, src: int = 0) -> None:
        host = tensor.cpu() if tensor.is_cuda else tensor
        dist.broadcast(host, src=src)
        if host is not tensor:
            tensor.copy_(host)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]

    # Point-to-point and gather helpers: used by the network check, not by training.
    def send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor, dst=dst)

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        dist.recv(tensor, src=src)

    def all_gather_object(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def any_true(self, flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def barrier(self) -> None:
        dist.barrier()

    def process_group(self) -> Any:
        return dist.group.WORLD

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
