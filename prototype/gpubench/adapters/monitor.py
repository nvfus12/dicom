"""GPU telemetry through NVML (package nvidia-ml-py). Optional."""
from __future__ import annotations

from typing import Any


class NullMonitor:
    def sample(self) -> dict[str, Any]:
        return {}


class NvmlMonitor:
    def __init__(self, index: int):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def sample(self) -> dict[str, Any]:
        n, h = self._nvml, self._handle
        try:
            util = n.nvmlDeviceGetUtilizationRates(h)
            mem = n.nvmlDeviceGetMemoryInfo(h)
            return {"util_pct": util.gpu,
                    "mem_used_mb": mem.used // 2**20,
                    "temp_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                    "sm_clock_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                    "power_w": n.nvmlDeviceGetPowerUsage(h) / 1000}
        except n.NVMLError as e:
            # A failed query is recorded as such, never as an idle GPU.
            return {"nvml_error": str(e)}


def build_monitor(device_type: str, index: int):
    if device_type != "cuda":
        return NullMonitor()
    try:
        return NvmlMonitor(index)
    except Exception:  # pynvml missing or NVML unavailable
        return NullMonitor()
