#!/usr/bin/env python3
"""Giao dien may dieu phoi: chon mot may worker va xin dung tai nguyen cua no.

    python controller.py

Doc danh sach may tu machines.json, cho chon mot may, nhap luong tai nguyen
muon xin, roi gui yeu cau den agent dang chay tren may do. Agent se hien hop
thoai cho chu may bam Cho phep hoac Tu choi, va cau tra loi hien lai o day.

Demo nay KHONG chiem tai nguyen that va KHONG chay huan luyen.

Chi dung thu vien chuan cua Python. Khong can cai them goi nao.
"""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

CONFIG_PATH = Path(__file__).with_name("machines.json")
REQUEST_TIMEOUT_S = 180         # dai hon thoi gian cho cua agent
PING_TIMEOUT_S = 4


def load_config() -> dict:
    """Doc machines.json; neu chua co thi dung 3 may gia lap tren cung may nay."""
    if not CONFIG_PATH.is_file():
        return {"token": "dicom-demo", "controller_name": "Máy điều phối",
                "machines": [{"name": f"may-{c}", "address": "127.0.0.1", "port": port}
                             for c, port in zip("abc", (8765, 8766, 8767))]}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def post_json(url: str, token: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json; charset=utf-8", "X-Demo-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, token: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"X-Demo-Token": token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return "Token không khớp giữa controller và agent."
        return f"Agent trả về lỗi HTTP {exc.code}."
    if isinstance(exc, urllib.error.URLError):
        return (f"Không kết nối được ({exc.reason}). Kiểm tra agent đã chạy chưa, "
                "địa chỉ có đúng không, và tường lửa có chặn không.")
    if isinstance(exc, TimeoutError):
        return "Quá thời gian chờ: chủ máy không trả lời."
    return f"{type(exc).__name__}: {exc}"


class ControllerApp:
    def __init__(self) -> None:
        self.config = load_config()
        self.token = self.config.get("token", "dicom-demo")
        self.controller_name = self.config.get("controller_name", "Máy điều phối")
        self.busy = False

        self.root = tk.Tk()
        self.root.title("Máy điều phối — demo xin tài nguyên")
        self.root.geometry("760x600")
        self._build_ui()
        self._fill_machines()

    # ---------------------------------------------------------------- giao dien

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        box = ttk.LabelFrame(outer, text="Chọn máy worker", padding=8)
        box.pack(fill="both", expand=False)
        self.tree = ttk.Treeview(box, columns=("address", "status"), height=5,
                                 selectmode="browse")
        self.tree.heading("#0", text="Tên máy")
        self.tree.heading("address", text="Địa chỉ")
        self.tree.heading("status", text="Trạng thái")
        self.tree.column("#0", width=180)
        self.tree.column("address", width=180)
        self.tree.column("status", width=300)
        self.tree.pack(fill="x")
        ttk.Button(box, text="Kiểm tra kết nối tất cả máy",
                   command=self.check_all).pack(anchor="w", pady=(8, 0))

        form = ttk.LabelFrame(outer, text="Tài nguyên muốn xin", padding=8)
        form.pack(fill="x", pady=(12, 0))
        self.fields: dict[str, tk.StringVar] = {}
        specs = [("gpu_vram_gb", "VRAM của GPU (GB)", "4"),
                 ("cpu_cores", "Số nhân CPU", "2"),
                 ("ram_gb", "RAM (GB)", "4"),
                 ("minutes", "Thời gian mượn (phút)", "30")]
        for i, (key, label, default) in enumerate(specs):
            ttk.Label(form, text=label + ":").grid(row=i // 2, column=(i % 2) * 2,
                                                   sticky="w", padx=(0, 8), pady=4)
            var = tk.StringVar(value=default)
            ttk.Entry(form, textvariable=var, width=12).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 24), pady=4)
            self.fields[key] = var

        ttk.Label(form, text="Công việc:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.job = tk.StringVar(value="Huấn luyện ResNet-20 trên CIFAR-10")
        ttk.Entry(form, textvariable=self.job, width=58).grid(
            row=2, column=1, columnspan=3, sticky="we", pady=4)

        ttk.Label(form, text="Ghi chú:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.note = tk.StringVar(value="Bạn có thể từ chối hoặc thu hồi bất cứ lúc nào.")
        ttk.Entry(form, textvariable=self.note, width=58).grid(
            row=3, column=1, columnspan=3, sticky="we", pady=4)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(12, 0))
        self.send_button = ttk.Button(actions, text="Gửi yêu cầu đến máy đã chọn",
                                      command=self.send_request)
        self.send_button.pack(side="left")
        self.status = ttk.Label(actions, text="Sẵn sàng.")
        self.status.pack(side="left", padx=12)

        log_box = ttk.LabelFrame(outer, text="Nhật ký", padding=6)
        log_box.pack(fill="both", expand=True, pady=(12, 0))
        self.log = tk.Text(log_box, height=10, wrap="word", state="disabled",
                           font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_box, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _fill_machines(self) -> None:
        self.machines = self.config.get("machines", [])
        for machine in self.machines:
            self.tree.insert("", "end", iid=machine["name"], text=machine["name"],
                             values=(f"{machine['address']}:{machine.get('port', 8765)}",
                                     "chưa kiểm tra"))
        if self.machines:
            self.tree.selection_set(self.machines[0]["name"])
        self.write_log(f"Đọc {len(self.machines)} máy từ {CONFIG_PATH.name}.")

    def write_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def selected_machine(self) -> dict | None:
        chosen = self.tree.selection()
        if not chosen:
            return None
        return next((m for m in self.machines if m["name"] == chosen[0]), None)

    def url_for(self, machine: dict, path: str) -> str:
        return f"http://{machine['address']}:{machine.get('port', 8765)}{path}"

    # ---------------------------------------------------------------- hanh dong

    def check_all(self) -> None:
        for machine in self.machines:
            self.tree.set(machine["name"], "status", "đang kiểm tra...")
            threading.Thread(target=self._check_one, args=(machine,), daemon=True).start()

    def _check_one(self, machine: dict) -> None:
        try:
            reply = get_json(self.url_for(machine, "/ping"), self.token, PING_TIMEOUT_S)
            text = f"sẵn sàng (agent: {reply.get('machine', '?')})"
        except Exception as exc:                      # loi mang nao cung chi bao o cot trang thai
            text = friendly_error(exc).split(".")[0]
        self.root.after(0, lambda: self.tree.set(machine["name"], "status", text))

    def collect_resources(self) -> dict | None:
        resources: dict[str, float] = {}
        for key, var in self.fields.items():
            raw = var.get().strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                messagebox.showerror("Giá trị không hợp lệ",
                                     f"'{raw}' không phải là số.")
                return None
            if value <= 0:
                messagebox.showerror("Giá trị không hợp lệ", "Các giá trị phải lớn hơn 0.")
                return None
            resources[key] = value
        if not resources:
            messagebox.showerror("Thiếu thông tin", "Cần ít nhất một loại tài nguyên.")
            return None
        return resources

    def send_request(self) -> None:
        if self.busy:
            return
        machine = self.selected_machine()
        if machine is None:
            messagebox.showinfo("Chưa chọn máy", "Hãy chọn một máy worker trong danh sách.")
            return
        resources = self.collect_resources()
        if resources is None:
            return

        payload = {"request_id": str(uuid.uuid4()),
                   "controller": self.controller_name,
                   "job": self.job.get().strip(),
                   "note": self.note.get().strip(),
                   "resources": resources,
                   "sent_at": datetime.now().isoformat(timespec="seconds")}

        self.busy = True
        self.send_button.state(["disabled"])
        self.tree.set(machine["name"], "status", "đang chờ trả lời...")
        self.write_log(f"Gửi yêu cầu đến {machine['name']}: "
                       + ", ".join(f"{k}={v:g}" for k, v in resources.items()))
        started = time.monotonic()

        def tick() -> None:
            if not self.busy:
                return
            self.set_status(f"Đang chờ chủ máy {machine['name']} trả lời... "
                            f"{int(time.monotonic() - started)}s")
            self.root.after(1000, tick)

        threading.Thread(target=self._send, args=(machine, payload), daemon=True).start()
        tick()

    def _send(self, machine: dict, payload: dict) -> None:
        try:
            reply = post_json(self.url_for(machine, "/request"), self.token,
                              payload, REQUEST_TIMEOUT_S)
            error = None
        except Exception as exc:
            reply, error = None, friendly_error(exc)
        self.root.after(0, lambda: self._finish(machine, reply, error))

    def _finish(self, machine: dict, reply: dict | None, error: str | None) -> None:
        self.busy = False
        self.send_button.state(["!disabled"])
        if error is not None:
            self.tree.set(machine["name"], "status", "lỗi")
            self.set_status("Không gửi được yêu cầu.")
            self.write_log(f"LỖI với {machine['name']}: {error}")
            messagebox.showerror("Không gửi được yêu cầu", error)
            return

        decision = (reply or {}).get("decision", "?")
        reason = (reply or {}).get("reason", "")
        label = {"allow": "ĐƯỢC CHO PHÉP", "deny": "BỊ TỪ CHỐI"}.get(decision, "HẾT THỜI GIAN CHỜ")
        self.tree.set(machine["name"], "status", label.lower())
        self.set_status(f"{machine['name']}: {label}")
        self.write_log(f"{machine['name']} trả lời: {label}. {reason}")
        if decision == "allow":
            messagebox.showinfo("Được cho phép",
                                f"{machine['name']} đồng ý cho mượn tài nguyên.\n\n"
                                "Đây là demo: chưa có công việc nào được chạy.")
        else:
            messagebox.showwarning("Không được cấp", f"{machine['name']}: {reason}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    try:
        ControllerApp().run()
    except json.JSONDecodeError as exc:
        print(f"{CONFIG_PATH} không phải JSON hợp lệ: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
