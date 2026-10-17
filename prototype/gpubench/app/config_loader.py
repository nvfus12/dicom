"""Reads YAML files into config objects. Unknown keys are errors, so typos fail fast."""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from ..domain.config import ConfigError, ExperimentSpec, MachineProfile

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_DIR / "configs"
RESULTS_DIR = PROJECT_DIR / "results"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} phải là một mapping YAML")
    return data


def _check_keys(cls, data: dict, source: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{source}: khóa không hợp lệ {sorted(unknown)}. Khóa hợp lệ: {sorted(allowed)}")


def load_machine(name: str, configs_dir: Path = CONFIGS_DIR) -> MachineProfile:
    path = configs_dir / "machines" / f"{name}.yaml"
    data = _read(path)
    _check_keys(MachineProfile, data, str(path))
    data.setdefault("name", name)
    if data["name"] != name:
        raise ConfigError(f"{path}: name '{data['name']}' phải trùng tên file '{name}'")
    # Relative data_dir is relative to the project, not to wherever the command runs.
    data_dir = Path(data.get("data_dir", "./data")).expanduser()
    data["data_dir"] = str(data_dir if data_dir.is_absolute() else PROJECT_DIR / data_dir)
    machine = MachineProfile(**data)
    machine.validate()
    return machine


def list_machines(configs_dir: Path = CONFIGS_DIR) -> list[str]:
    return sorted(p.stem for p in (configs_dir / "machines").glob("*.yaml"))


def resolve_experiment_path(ref: str, configs_dir: Path = CONFIGS_DIR) -> Path:
    path = Path(ref)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        return path
    return configs_dir / "experiments" / f"{ref}.yaml"


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """`key=value` pairs; values are parsed as YAML (numbers, lists, null...)."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set cần dạng key=value, nhận '{item}'")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_experiment(ref: str, overrides: dict[str, Any] | None = None,
                    runner: str | None = None, configs_dir: Path = CONFIGS_DIR) -> ExperimentSpec:
    """`runner` fills `workers` for single-GPU experiments that leave it empty."""
    path = resolve_experiment_path(ref, configs_dir)
    data = _read(path)
    data.update(overrides or {})
    _check_keys(ExperimentSpec, data, str(path))
    data.setdefault("name", path.stem)
    workers = tuple(data.get("workers") or ())
    if not workers and data.get("mode") == "single" and runner:
        workers = (runner,)
    data["workers"] = workers
    spec = ExperimentSpec(**data)
    spec.validate()
    return spec


def with_overrides(spec: ExperimentSpec, **changes: Any) -> ExperimentSpec:
    new = replace(spec, **changes)
    new.validate()
    return new
