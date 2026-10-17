from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlSink:
    """One JSON object per line; flushed after every event so a crash keeps the log."""

    def __init__(self, path: Path, echo: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = path.open("a", encoding="utf-8")
        self._echo = echo

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()
        if self._echo:
            print(_short(record), flush=True)

    def close(self) -> None:
        self._f.close()


def _short(record: dict) -> str:
    keys = ("step", "epoch", "loss", "test_acc", "train_time_s", "comm_s", "samples_per_s",
            "stop_reason", "time_to_target_s")
    parts = [record["event"]]
    for k in keys:
        v = record.get(k)
        if v is not None:
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)
