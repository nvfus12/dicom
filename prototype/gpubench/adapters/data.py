"""Dataset adapters. Each worker reads a disjoint, equal-sized shard per epoch."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from ..domain.config import ExperimentSpec, MachineProfile

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class _LoaderFactory:
    num_classes = 10

    def __init__(self, train: Dataset, test: Dataset, num_workers: int, pin_memory: bool):
        self._train = train
        self._test = test
        self._num_workers = num_workers
        self._pin = pin_memory

    def _kwargs(self) -> dict:
        kw = dict(num_workers=self._num_workers, pin_memory=self._pin)
        if self._num_workers > 0:
            # Windows spawns worker processes; keep them alive across epochs.
            kw["persistent_workers"] = True
        return kw

    def train_loader(self, rank: int, world_size: int, batch_size: int, seed: int):
        # drop_last keeps the number of steps identical on every worker,
        # which the collectives require.
        sampler = DistributedSampler(self._train, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
        loader = DataLoader(self._train, batch_size=batch_size, sampler=sampler,
                            drop_last=True, **self._kwargs())
        return loader, sampler

    def test_loader(self, batch_size: int):
        return DataLoader(self._test, batch_size=batch_size, shuffle=False, **self._kwargs())


def cifar10(machine: MachineProfile) -> _LoaderFactory:
    from torchvision import datasets, transforms

    root = Path(machine.data_dir).expanduser()
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    try:
        train = datasets.CIFAR10(root, train=True, transform=train_tf, download=machine.download)
        test = datasets.CIFAR10(root, train=False, transform=test_tf, download=machine.download)
    except RuntimeError as e:
        raise RuntimeError(f"Không tìm thấy CIFAR-10 trong '{root}'. Bật download: true trong file máy "
                           f"hoặc chép dữ liệu vào đó. Lỗi gốc: {e}") from e
    return _LoaderFactory(train, test, machine.num_workers, machine.device == "cuda")


def synthetic(samples: int, seed: int = 1234) -> _LoaderFactory:
    """Learnable fake CIFAR-shaped data: each class has its own mean image plus noise.

    Generated from a fixed seed, so every worker builds exactly the same dataset.
    """
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(10, 3, 32, 32, generator=g)

    def make(n: int):
        y = torch.randint(0, 10, (n,), generator=g)
        x = prototypes[y] + 1.5 * torch.randn(n, 3, 32, 32, generator=g)
        return TensorDataset(x, y)

    return _LoaderFactory(make(samples), make(max(256, samples // 4)), 0, False)


def build_dataset(spec: ExperimentSpec, machine: MachineProfile) -> _LoaderFactory:
    if spec.dataset == "cifar10":
        return cifar10(machine)
    if spec.dataset == "synthetic":
        return synthetic(spec.synthetic_samples)
    raise ValueError(f"dataset '{spec.dataset}' không có; chọn cifar10 hoặc synthetic")
