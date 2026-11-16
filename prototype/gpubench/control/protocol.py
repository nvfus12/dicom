"""What the lab controller and its agents agree on: constants, run parameters, experiment building.

The controller never sends commands or YAML. It sends a small set of named parameters,
and every agent turns the same parameters into the same experiment, so fingerprints match.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any

DEFAULT_PORT = 8765
TOKEN_HEADER = "X-Lab-Token"
DEFAULT_TOKEN = "lab-dicom"
APPROVAL_TIMEOUT_S = 120            # the owner's dialog; no answer = refused
APPROVAL_VALID_S = 600              # an approval covers one run_id for this long
POLL_INTERVAL_S = 3
AGENT_VERSION = "1.0"

MODES = ("ddp", "localsgd", "sync")
MODE_LABELS = {"ddp": "DDP", "localsgd": "Local SGD", "sync": "Sync (all-reduce thủ công)"}
REAL_DATASETS = ("cifar10",)
REAL_MODELS = ("resnet20", "resnet18")
QUICK_DATASETS = ("synthetic",)
QUICK_MODELS = ("tiny_cnn",)
COMM_DTYPES = ("fp32", "fp16")
QUICK_SAMPLES = 2048


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RunParams:
    mode: str
    dataset: str = "cifar10"
    model: str = "resnet20"
    local_steps: int = 256
    max_train_minutes: float = 30
    max_epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    comm_dtype: str = "fp32"
    quick: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(data: dict, key: str, kind: type, low: float, high: float) -> Any:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"'{key}' phải là số, nhận {value!r}")
    if kind is int and value != int(value):
        raise ProtocolError(f"'{key}' phải là số nguyên, nhận {value!r}")
    if not low <= value <= high:
        raise ProtocolError(f"'{key}' phải trong khoảng {low}–{high}, nhận {value!r}")
    return kind(value)


def _choice(data: dict, key: str, allowed: tuple[str, ...]) -> str:
    value = data[key]
    if value not in allowed:
        raise ProtocolError(f"'{key}' phải là một trong {list(allowed)}, nhận {value!r}")
    return value


def validate_params(data: Any) -> RunParams:
    """Checks a parameter mapping received from the network against the allow-lists."""
    if not isinstance(data, dict):
        raise ProtocolError("Tham số lượt chạy phải là một mapping")
    allowed = {f.name for f in fields(RunParams)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ProtocolError(f"Tham số không hợp lệ: {unknown}")
    if "mode" not in data:
        raise ProtocolError("Thiếu tham số 'mode'")
    merged = {**RunParams(mode="ddp").to_dict(), **data}

    quick = merged["quick"]
    if not isinstance(quick, bool):
        raise ProtocolError(f"'quick' phải là true/false, nhận {quick!r}")
    return RunParams(
        mode=_choice(merged, "mode", MODES),
        dataset=_choice(merged, "dataset", QUICK_DATASETS if quick else REAL_DATASETS),
        model=_choice(merged, "model", QUICK_MODELS if quick else REAL_MODELS),
        local_steps=_number(merged, "local_steps", int, 1, 4096),
        max_train_minutes=_number(merged, "max_train_minutes", float, 1, 240),
        max_epochs=_number(merged, "max_epochs", int, 1, 300),
        batch_size=_number(merged, "batch_size", int, 16, 1024),
        lr=_number(merged, "lr", float, 0.001, 1),
        comm_dtype=_choice(merged, "comm_dtype", COMM_DTYPES),
        quick=quick,
    )


def build_experiment(params: RunParams, workers: list[str], run_id: str) -> dict[str, Any]:
    """The experiment mapping every agent writes to disk; identical input gives identical output."""
    local = params.mode == "localsgd"
    data: dict[str, Any] = {
        "name": run_id,
        "mode": params.mode,
        "workers": list(workers),
        "model": params.model,
        "dataset": params.dataset,
        "batch_size": params.batch_size,
        "lr": params.lr,
        "lr_scaling": "none" if local else "linear",
        "warmup_epochs": 1 if local else 2,
        "max_epochs": params.max_epochs,
        "max_train_minutes": params.max_train_minutes,
        "target_acc": None,
        "eval_every_epochs": 1,
        "log_every_steps": 50,
        "amp": False,
        "comm_dtype": params.comm_dtype,
        "seed": 0,
        "master_port": 29500,
        "timeout_s": 300,
    }
    if local:
        data["local_steps"] = params.local_steps
    if params.quick:
        data["synthetic_samples"] = QUICK_SAMPLES
    return data


def new_run_id(params: RunParams, now: datetime) -> str:
    prefix = "thu" if params.quick else "lab"
    return f"{prefix}-{params.mode}-{params.model}-{now:%Y%m%d-%H%M%S}"


def rank_order(infos: list[dict]) -> list[str]:
    """Largest VRAM first (rank 0 evaluates and hosts the rendezvous); unknown VRAM last."""
    def key(info: dict):
        vram = (info.get("gpu") or {}).get("vram_mb")
        return (vram is None, -(vram or 0), info["name"])
    return [info["name"] for info in sorted(infos, key=key)]


def describe_params(params: RunParams) -> list[tuple[str, str]]:
    """Label/value rows for the owner's dialog."""
    rows = [("Chế độ", MODE_LABELS[params.mode]),
            ("Model", "Tiny CNN (chạy thử)" if params.quick else params.model),
            ("Dataset", "Dữ liệu tổng hợp (chạy thử)" if params.quick else "CIFAR-10")]
    if params.mode == "localsgd":
        rows.append(("Đồng bộ mỗi (H)", f"{params.local_steps} bước"))
    rows.append(("Thời gian tối đa", f"{params.max_train_minutes:g} phút"))
    rows.append(("Epoch tối đa", str(params.max_epochs)))
    return rows
