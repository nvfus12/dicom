"""What the agent reports about its machine: software versions, driver, power, firewall.

Read-only. Every probe is allowed to fail (other OS, missing tool, no permission) and then
reports None, so a report never stops the agent.
"""
from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import Any

POWER_MODES = {"ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
               "00000000-0000-0000-0000-000000000000": "Balanced",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency"}
FIREWALL_RULE = "gpubench-lab"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _run(argv: list[str], timeout: float = 10) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _nvidia() -> tuple[str | None, str | None]:
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    full = _run(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", full or "")
    return (driver.strip().splitlines()[0] if driver and driver.strip() else None,
            cuda.group(1) if cuda else None)


def _plugged_in() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {1: True, 0: False}.get(status.ACLineStatus)
    except Exception:
        return None


def _power_mode() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as key:
            value, _ = winreg.QueryValueEx(key, "ActiveOverlayAcPowerScheme")
        return POWER_MODES.get(str(value).lower(), f"khác ({value})")
    except OSError:
        return None


def _sleep_ac_minutes() -> int | None:
    """Minutes before sleep on AC power; 0 means never. Reads values by position, not by the
    localised labels powercfg prints."""
    out = _run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"])
    values = re.findall(r":\s*(0x[0-9a-fA-F]+)\s*$", out or "", re.MULTILINE)
    if len(values) < 2:
        return None
    return int(values[-2], 16) // 60


def _firewall_rule() -> bool | None:
    out = _run(["powershell", "-NoProfile", "-Command",
                f"(Get-NetFirewallRule -DisplayName {FIREWALL_RULE} -ErrorAction SilentlyContinue | "
                f"Where-Object {{ $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' -and "
                f"$_.Action -eq 'Allow' }} | Measure-Object).Count"], timeout=20)
    if out is None or not out.strip().isdigit():
        return None
    return int(out.strip()) > 0


def machine_report(torch_version: str | None) -> dict[str, Any]:
    driver, cuda = _nvidia()
    return {"python": platform.python_version(), "python_path": sys.executable,
            "torch": torch_version, "nvidia_driver": driver, "cuda_driver": cuda,
            "plugged_in": _plugged_in(), "power_mode": _power_mode(),
            "sleep_ac_minutes": _sleep_ac_minutes(), "firewall_rule": _firewall_rule(),
            "os": platform.platform()}
