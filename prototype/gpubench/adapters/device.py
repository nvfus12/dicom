from __future__ import annotations

from typing import Any

import torch

from ..domain.config import MachineProfile


class CudaDevice:
    def __init__(self, index: int, amp: bool):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Profile yêu cầu device=cuda nhưng torch.cuda.is_available() = False. "
                f"Bản PyTorch hiện tại: {torch.__version__}. Cài bản PyTorch có CUDA phù hợp driver, "
                "hoặc đặt device: cpu trong file máy.")
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda_index={index} nhưng máy chỉ có {torch.cuda.device_count()} GPU CUDA")
        self.torch_device = torch.device(f"cuda:{index}")
        self.amp = amp
        torch.cuda.set_device(self.torch_device)
        torch.backends.cudnn.benchmark = True

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.torch_device)

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.torch_device)

    def peak_memory_bytes(self) -> int | None:
        return torch.cuda.max_memory_allocated(self.torch_device)

    def describe(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(self.torch_device)
        return {"type": "cuda", "name": props.name, "total_memory_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}", "cuda_runtime": torch.version.cuda,
                "amp": self.amp}


class CpuDevice:
    def __init__(self):
        self.torch_device = torch.device("cpu")
        self.amp = False

    def synchronize(self) -> None:
        pass

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_bytes(self) -> int | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"type": "cpu", "threads": torch.get_num_threads(), "amp": False}


def build_device(machine: MachineProfile, amp: bool = False):
    if machine.device == "cuda":
        return CudaDevice(machine.cuda_index, amp)
    if amp:
        raise ValueError(f"Máy '{machine.name}' chạy CPU nên không dùng được amp=true")
    return CpuDevice()
