"""What the lab controller and its agents agree on: constants, run parameters, experiment building.

The controller never sends commands or YAML. It sends a small set of named parameters,
and every agent turns the same parameters into the same experiment, so fingerprints match.
"""
from __future__ import annotations

import re
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

MODES = ("ddp", "localsgd", "sync", "netcheck", "selftest", "glootest", "porttest", "profile", "single")
MODE_LABELS = {"ddp": "DDP", "localsgd": "Local SGD", "sync": "Sync (all-reduce thủ công)",
               "netcheck": "Kiểm tra mạng (khoảng 1–2 phút, không dùng GPU)",
               "selftest": "Tự thử gloo trên máy này (khoảng 1 phút, không dùng GPU)",
               "glootest": "Chẩn đoán kết nối gloo giữa các máy (tối đa 4 phút, không dùng GPU)",
               "porttest": "Thử thông đường TCP hai chiều giữa hai máy (khoảng 1 phút, không dùng GPU)",
               "profile": "Đo tốc độ GPU (khoảng 3 phút)",
               "single": "Single GPU (huấn luyện trên một máy)"}
GLOO_DEVICES = ("ipv4", "interface")
GLOO_DEVICE_LABELS = {"ipv4": "theo IP", "interface": "theo tên card"}


@dataclass(frozen=True)
class JobKind:
    min_machines: int
    max_machines: int | None
    needs_cuda: bool        # unless it is a quick run
    diagnostic: bool        # writes a verdict file instead of a training log
    analyse: bool           # goes into results/report-lab


JOB_KINDS = {
    "ddp": JobKind(2, None, True, False, True),
    "localsgd": JobKind(2, None, True, False, True),
    "sync": JobKind(2, None, True, False, True),
    "single": JobKind(1, 1, True, False, True),
    "netcheck": JobKind(2, None, False, False, True),
    "selftest": JobKind(1, None, False, True, False),
    "glootest": JobKind(2, None, False, True, False),
    "porttest": JobKind(2, 2, False, True, False),
    "profile": JobKind(1, None, True, True, False),
}
NO_QUICK = ("netcheck", "selftest", "glootest", "porttest")


def job_kind(mode: str) -> JobKind:
    return JOB_KINDS[mode]
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
    gloo_device: str = "ipv4"       # ipv4: the address that reaches rank 0; interface: by adapter name

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
    if merged["mode"] in NO_QUICK and quick:
        raise ProtocolError(f"{MODE_LABELS.get(merged['mode'], merged['mode'])}: không có chế độ chạy thử")
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
        gloo_device=_choice(merged, "gloo_device", GLOO_DEVICES),
    )


def build_experiment(params: RunParams, workers: list[str], run_id: str) -> dict[str, Any]:
    """The experiment mapping every agent writes to disk; identical input gives identical output."""
    local = params.mode == "localsgd"
    if params.mode == "single":
        mode = "single"
    elif params.mode in JOB_KINDS and JOB_KINDS[params.mode].diagnostic or params.mode == "netcheck":
        mode = "sync"   # checks only need a valid distributed experiment: workers, port, timeout
    else:
        mode = params.mode
    data: dict[str, Any] = {
        "name": run_id,
        "mode": mode,
        "workers": list(workers),
        "model": params.model,
        "dataset": params.dataset,
        "batch_size": params.batch_size,
        "lr": params.lr,
        "lr_scaling": "none" if local or mode == "single" else "linear",
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


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


def valid_run_id(value: Any) -> bool:
    """A run_id becomes a folder and file name on every machine; nothing that can leave it."""
    return isinstance(value, str) and bool(_RUN_ID.fullmatch(value)) and ".." not in value


def new_run_id(params: RunParams, now: datetime, workers: list[str] = ()) -> str:
    """Folder name of a run: kind first, then what it ran on, then when.
    `workers` are in rank order (rank 0 first)."""
    stamp = f"{now:%Y%m%d-%H%M%S}"
    prefix = "thu" if params.quick else "lab"
    if params.mode in ("netcheck", "selftest"):
        return f"lab-{params.mode}-{stamp}"
    if params.mode in ("glootest", "porttest"):
        return f"lab-{params.mode}-{'-'.join(workers)}-{stamp}"
    if params.mode == "profile":
        return f"{prefix}-profile-{params.model}-{stamp}"
    if params.mode == "single":
        return f"{prefix}-single-{params.model}-{workers[0]}-{stamp}"
    return f"{prefix}-{params.mode}-{params.model}-{stamp}"


def rank_order(infos: list[dict]) -> list[str]:
    """Largest VRAM first (rank 0 evaluates and hosts the rendezvous); unknown VRAM last."""
    def key(info: dict):
        vram = (info.get("gpu") or {}).get("vram_mb")
        return (vram is None, -(vram or 0), info["name"])
    return [info["name"] for info in sorted(infos, key=key)]


def describe_params(params: RunParams) -> list[tuple[str, str]]:
    """Label/value rows for the owner's dialog."""
    gloo = ("Cách kết nối gloo", GLOO_DEVICE_LABELS[params.gloo_device])
    if params.mode in ("netcheck", "selftest", "glootest", "porttest"):
        rows = [("Chế độ", MODE_LABELS[params.mode])]
        return rows + [gloo] if params.mode != "porttest" else rows
    if params.mode == "profile":
        return [("Chế độ", MODE_LABELS["profile"]),
                ("Model", "Tiny CNN (chạy thử)" if params.quick else params.model)]
    rows = [("Chế độ", MODE_LABELS[params.mode]),
            ("Model", "Tiny CNN (chạy thử)" if params.quick else params.model),
            ("Dataset", "Dữ liệu tổng hợp (chạy thử)" if params.quick else "CIFAR-10")]
    if params.mode == "localsgd":
        rows.append(("Đồng bộ mỗi (H)", f"{params.local_steps} bước"))
    rows.append(("Thời gian tối đa", f"{params.max_train_minutes:g} phút"))
    rows.append(("Epoch tối đa", str(params.max_epochs)))
    if params.mode != "single":
        rows.append(gloo)
    return rows


_FRAME = re.compile(r"^\s*[0-9A-Fa-f]{12,}")
_USEFUL_FRAME = re.compile(r"c10d::|gloo", re.IGNORECASE)


def log_excerpt(text: str, max_lines: int = 30) -> str:
    """The lines of a training log worth reading. On Windows a crash inside PyTorch prints a
    native stack trace of hundreds of address lines and no message; keep the messages, the
    warnings, and only the frames that name the distributed step that failed."""
    kept: list[str] = []
    for line in text.splitlines():
        if _FRAME.match(line):
            if not _USEFUL_FRAME.search(line):
                continue
            line = "  [frame] " + line.split(" ", 1)[-1].split(" [<unknown file>", 1)[0].strip()
        if line.strip() and (not kept or kept[-1] != line):
            kept.append(line)
    return "\n".join(kept[-max_lines:])
