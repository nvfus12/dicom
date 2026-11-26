"""Verdicts of the check runs, read from the agents' diag-<machine>.json files.

Each verdict has a status (ĐẠT / LỖI), readable lines, and a hint: the next step of the
diagnosis ladder in the session guide, so the person at the controller can go on without help.
"""
from __future__ import annotations

import json
from pathlib import Path

VARIANTS = ("theo-ten-card", "theo-ip")


def _load(run_dir: Path, machine: str) -> dict | None:
    path = run_dir / f"diag-{machine}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _missing(machine: str) -> str:
    return f"{machine}: không có kết quả (xem agent-{machine}.log trong thư mục lượt này)"


def _detail_lines(detail: str, limit: int = 4) -> list[str]:
    lines = [line.strip() for line in (detail or "").splitlines() if line.strip()]
    return [f"    {line}" for line in lines[-limit:]]


def _selftest(run_dir: Path, workers: list[str]) -> dict:
    lines, broken = [], []
    for machine in workers:
        data = _load(run_dir, machine)
        if data is None:
            lines.append(_missing(machine))
            broken.append(machine)
            continue
        v = data["variants"]
        lines.append(f"{machine}: " + ", ".join(f"{name} {v[name]['verdict']}" for name in VARIANTS if name in v))
        if v.get("theo-ip", {}).get("verdict") != "ĐẠT":
            broken.append(machine)
            lines += _detail_lines(v.get("theo-ip", {}).get("detail", ""))
    if broken:
        return {"status": "LỖI", "lines": lines,
                "hint": f"Lỗi nằm trong máy {', '.join(broken)} (không do mạng): máy đó không tự mở được "
                        f"kết nối gloo với chính nó. Dừng thang chẩn đoán; về mạng trường gửi AI thư mục "
                        f"lượt này."}
    return {"status": "ĐẠT", "lines": lines,
            "hint": "Mỗi máy tự chạy được gloo. Bước tiếp: Thông đường TCP."}


def _porttest(run_dir: Path, workers: list[str]) -> dict:
    lines, hints = [], []
    for i, machine in enumerate(workers):
        data = _load(run_dir, machine)
        other = workers[1 - i] if len(workers) == 2 else "?"
        if data is None:
            lines.append(_missing(machine))
            hints.append(f"{machine} không chạy được phép thử; xem log của máy đó.")
            continue
        probe = data.get("probe", {})
        lines.append(f"{machine} → {other}: {'ĐẠT' if probe.get('ok') else 'LỖI'}")
        if not probe.get("ok"):
            message = probe.get("message", "")
            lines.append(f"    {message}")
            if "bị chặn" in message:
                hints.append(f"Kết nối tới {other} bị chặn: kiểm tra tường lửa máy {other} (quy tắc "
                             f"gpubench-lab, quy tắc chặn Python), rồi chạy lại.")
            elif "từ chối" in message:
                hints.append(f"{other} chưa nghe kịp hoặc sai IP của {other}: kiểm tra IP rồi chạy lại.")
            else:
                hints.append(f"Lỗi lạ khi gọi {other}; xem dòng lỗi ở trên.")
    if hints:
        return {"status": "LỖI", "lines": lines, "hint": " ".join(hints)}
    return {"status": "ĐẠT", "lines": lines,
            "hint": "Đường TCP hai chiều thông. Bước tiếp: glootest."}


def _glootest(run_dir: Path, workers: list[str]) -> dict:
    per_variant = {name: [] for name in VARIANTS}
    lines, details = [], []
    for machine in workers:
        data = _load(run_dir, machine)
        if data is None:
            lines.append(_missing(machine))
            for name in VARIANTS:
                per_variant[name].append("LỖI")
            continue
        for name in VARIANTS:
            entry = data["variants"].get(name, {"verdict": "BỎ QUA", "detail": ""})
            per_variant[name].append(entry["verdict"])
            if entry["verdict"] == "LỖI":
                details.append(f"{machine}, {name}:")
                details += _detail_lines(entry.get("detail", ""))

    def combined(name: str) -> str:
        verdicts = per_variant[name]
        if "LỖI" in verdicts:
            return "LỖI"
        return "ĐẠT" if "ĐẠT" in verdicts else "BỎ QUA"
    card, ip = combined("theo-ten-card"), combined("theo-ip")
    lines = [f"theo-ten-card: {card}", f"theo-ip: {ip}"] + lines + details
    if ip == "ĐẠT" and card == "BỎ QUA":
        hint = ("theo-ip ĐẠT; cách theo tên card không thử được (không tìm thấy card Wi-Fi theo tên). "
                "Giữ Cách kết nối gloo = theo IP. Bước tiếp: chạy thử.")
    elif ip == "ĐẠT" and card == "LỖI":
        hint = ("Tìm ra: gloo chọn sai địa chỉ khi chọn theo tên card. Giữ Cách kết nối gloo = theo IP "
                "(mặc định). Bước tiếp: chạy thử.")
    elif ip == "ĐẠT":
        hint = "Cả hai cách đều chạy. Giữ Cách kết nối gloo = theo IP. Bước tiếp: chạy thử."
    elif card == "ĐẠT":
        hint = "Chỉ cách theo tên card chạy: chọn Cách kết nối gloo = theo tên card, rồi chạy thử."
    else:
        hint = ("Cả hai cách LỖI. Nếu chưa đổi vai: chọn Máy làm điểm gặp = máy còn lại rồi chạy lại "
                "glootest. Nếu đã đổi vai: tắt IPv6 trên card Wi-Fi của hai máy (xem hướng dẫn), lấy "
                "lại IP, Sửa IP, rồi chạy lại glootest.")
    return {"status": "ĐẠT" if "ĐẠT" in (ip, card) else "LỖI", "lines": lines, "hint": hint}


def _profile(run_dir: Path, workers: list[str]) -> dict:
    lines, ok = [], True
    for machine in workers:
        data = _load(run_dir, machine)
        if data is None:
            lines.append(_missing(machine))
            ok = False
            continue
        for model, rows in data.get("models", {}).items():
            fit = [r for r in rows if not r.get("out_of_memory")]
            at_128 = next((r for r in fit if r["batch"] == 128), fit[-1] if fit else None)
            if at_128 is None:
                lines.append(f"{machine} {model}: không batch nào vừa VRAM")
                ok = False
                continue
            vram = f", VRAM {at_128['vram_peak_mb']} MB" if at_128.get("vram_peak_mb") is not None else ""
            lines.append(f"{machine} {model}: batch {at_128['batch']} = {at_128['samples_per_s']:.0f} "
                         f"mẫu/s{vram} (batch lớn nhất vừa VRAM: {fit[-1]['batch']})")
    return {"status": "ĐẠT" if ok else "LỖI", "lines": lines,
            "hint": "Số liệu cho mục Tốc độ từng GPU của báo cáo." if ok else "Xem log của máy bị lỗi."}


_SUMMARIES = {"selftest": _selftest, "porttest": _porttest, "glootest": _glootest, "profile": _profile}


def summarize(mode: str, run_dir: Path, workers: list[str]) -> dict | None:
    """Verdict of a check run, or None for training runs."""
    summary = _SUMMARIES.get(mode)
    return summary(run_dir, workers) if summary else None
