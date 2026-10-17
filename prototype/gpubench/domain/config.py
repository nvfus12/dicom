"""Configuration objects. Pure data: no I/O, no torch."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

MODES = ("single", "ddp", "sync", "localsgd")
DEVICES = ("cuda", "cpu")
LR_SCALING = ("linear", "none")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MachineProfile:
    """Everything that differs between machines. One YAML file per machine."""

    name: str
    owner: str = ""
    address: str | None = None          # IPv4 reachable by the other workers
    device: str = "cuda"
    cuda_index: int = 0
    amp: bool = False                   # mixed precision, only used on cuda
    num_workers: int = 2                # DataLoader worker processes
    data_dir: str = "./data"
    download: bool = True               # allow dataset download into data_dir
    max_batch: dict[str, int] = field(default_factory=dict)  # per model, from profiling
    network_interface: str | None = None  # GLOO_SOCKET_IFNAME if gloo picks the wrong NIC
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> None:
        if self.device not in DEVICES:
            raise ConfigError(f"[{self.name}] device phải là một trong {DEVICES}, nhận '{self.device}'")
        if self.num_workers < 0:
            raise ConfigError(f"[{self.name}] num_workers phải >= 0")


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything that must be identical on every worker of a run."""

    name: str
    mode: str
    workers: tuple[str, ...] = ()       # rank order; rank 0 hosts the rendezvous
    model: str = "resnet20"
    dataset: str = "cifar10"
    synthetic_samples: int = 2048       # only for dataset=synthetic
    batch_size: int = 128               # per worker
    lr: float = 0.1                     # base lr for one worker
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scaling: str = "linear"          # applied to ddp/sync only
    warmup_epochs: float = 1.0
    max_epochs: int = 30
    max_train_minutes: float | None = None
    target_acc: float | None = None     # stop when test accuracy reaches this
    eval_every_epochs: int = 1
    log_every_steps: int = 50
    local_steps: int = 1                # H, localsgd only
    seed: int = 0
    master_port: int = 29500
    timeout_s: int = 600

    @property
    def world_size(self) -> int:
        return len(self.workers)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ConfigError(f"mode phải là một trong {MODES}, nhận '{self.mode}'")
        if self.lr_scaling not in LR_SCALING:
            raise ConfigError(f"lr_scaling phải là một trong {LR_SCALING}")
        if not self.workers:
            raise ConfigError("workers rỗng")
        if len(set(self.workers)) != len(self.workers):
            raise ConfigError(f"workers bị trùng: {self.workers}")
        if self.mode == "single" and self.world_size != 1:
            raise ConfigError("mode=single chỉ chạy 1 worker")
        if self.mode != "single" and self.world_size < 2:
            raise ConfigError(f"mode={self.mode} cần ít nhất 2 workers")
        if self.mode == "localsgd" and self.local_steps < 1:
            raise ConfigError("local_steps phải >= 1")
        if self.batch_size < 1 or self.max_epochs < 1 or self.eval_every_epochs < 1:
            raise ConfigError("batch_size, max_epochs, eval_every_epochs phải >= 1")
        if self.target_acc is not None and not 0 < self.target_acc <= 1:
            raise ConfigError("target_acc là tỉ lệ trong (0, 1], ví dụ 0.85")

    def fingerprint(self) -> str:
        """Hash used to check that every worker loaded the same experiment."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def effective_lr(self) -> float:
        if self.mode in ("ddp", "sync") and self.lr_scaling == "linear":
            return self.lr * self.world_size
        return self.lr
