#!/usr/bin/env python3
"""Agent chay tren may worker: nhan yeu cau muon tai nguyen va hoi chu may.

    python agent.py --name may-a

Agent mo mot cong HTTP va cho may dieu phoi gui yeu cau. Moi yeu cau lam hien
mot hop thoai de chu may bam "Cho phep" hoac "Tu choi"; cau tra loi duoc gui
nguoc ve may dieu phoi.

Demo nay KHONG gioi han tai nguyen that va KHONG chay huan luyen. No chi minh
hoa buoc xin phep va nhuong quyen su dung GPU.

Chi dung thu vien chuan cua Python. Khong can cai them goi nao.
"""
from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import tkinter as tk
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import ttk

DEFAULT_PORT = 8765
DEFAULT_TOKEN = "dicom-demo"
DECISION_TIMEOUT_S = 120        # tu dong tu choi neu chu may khong tra loi
LOG_PATH = Path(__file__).with_name("agent-log.jsonl")


@dataclass
class PendingRequest:
    """Mot yeu cau dang cho chu may tra loi."""

    payload: dict
    peer: str
    answered: threading.Event = field(default_factory=threading.Event)
    decision: str = "timeout"
    reason: str = "Chủ máy không trả lời trong thời gian chờ."


def lan_ip() -> str:
    """Dia chi IPv4 ma cac may khac nhin thay. Khong gui goi tin nao."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def describe(resources: dict) -> list[tuple[str, str]]:
    """Doi dict tai nguyen thanh cac dong de hien trong hop thoai."""
    labels = [
        ("gpu_vram_gb", "VRAM của GPU", "GB"),
        ("cpu_cores", "Số nhân CPU", "nhân"),
        ("ram_gb", "RAM", "GB"),
        ("minutes", "Thời gian mượn", "phút"),
    ]
    rows = [(label, f"{resources[key]:g} {unit}")
            for key, label, unit in labels if resources.get(key) is not None]
    for key, value in resources.items():
        if key not in {k for k, _, _ in labels}:
            rows.append((key, str(value)))
    return rows


def make_handler(agent: "AgentApp"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, body: dict) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorised(self) -> bool:
            return self.headers.get("X-Demo-Token", "") == agent.token

        def do_GET(self) -> None:
            if self.path != "/ping":
                self._send_json(404, {"error": "không có đường dẫn này"})
                return
            if not self._authorised():
                self._send_json(401, {"error": "token không đúng"})
                return
            self._send_json(200, {"status": "ok", "machine": agent.name,
                                  "busy": agent.pending.qsize() > 0})

        def do_POST(self) -> None:
            if self.path != "/request":
                self._send_json(404, {"error": "không có đường dẫn này"})
                return
            if not self._authorised():
                self._send_json(401, {"error": "token không đúng"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("thân yêu cầu phải là một đối tượng JSON")
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": f"JSON không hợp lệ: {exc}"})
                return

            request = PendingRequest(payload=payload, peer=self.client_address[0])
            agent.pending.put(request)
            # Cho chu may bam nut. Hop thoai do luong giao dien tao ra.
            request.answered.wait(timeout=DECISION_TIMEOUT_S + 5)
            self._send_json(200, {
                "decision": request.decision,
                "reason": request.reason,
                "machine": agent.name,
                "request_id": payload.get("request_id", ""),
                "decided_at": datetime.now().isoformat(timespec="seconds"),
            })

        def log_message(self, fmt: str, *args) -> None:
            pass        # khong in ra console; nhat ky nam trong cua so agent

    return Handler


class AgentApp:
    def __init__(self, name: str, host: str, port: int, token: str):
        self.name = name
        self.host = host
        self.port = port
        self.token = token
        self.pending: queue.Queue[PendingRequest] = queue.Queue()
        self.server: ThreadingHTTPServer | None = None
        self._poll_timer: str | None = None
        self._close_dialog = None       # dat khi dang co hop thoai mo

        self.root = tk.Tk()
        self.root.title(f"Agent demo — {name}")
        self.root.geometry("640x420")
        self._build_ui()

    # ---------------------------------------------------------------- giao dien

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        info = ttk.LabelFrame(frame, text="Máy này", padding=10)
        info.pack(fill="x")
        rows = [("Tên máy", self.name),
                ("Địa chỉ cho máy điều phối", f"{lan_ip()}:{self.port}"),
                ("Lắng nghe trên", f"{self.host}:{self.port}")]
        for i, (label, value) in enumerate(rows):
            ttk.Label(info, text=label + ":").grid(row=i, column=0, sticky="w", padx=(0, 10))
            ttk.Label(info, text=value, font=("Segoe UI", 9, "bold")).grid(row=i, column=1, sticky="w")

        self.status = ttk.Label(frame, text="Đang chờ yêu cầu từ máy điều phối...",
                                font=("Segoe UI", 10))
        self.status.pack(fill="x", pady=(12, 6))

        log_box = ttk.LabelFrame(frame, text="Nhật ký", padding=6)
        log_box.pack(fill="both", expand=True)
        self.log = tk.Text(log_box, height=10, wrap="word", state="disabled",
                           font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_box, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Button(frame, text="Thoát", command=self.quit).pack(anchor="e", pady=(8, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def write_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{stamp}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- hop thoai

    def ask_owner(self, request: PendingRequest) -> None:
        payload = request.payload
        resources = payload.get("resources", {})
        controller = payload.get("controller", "không rõ")
        job = payload.get("job", "không ghi")

        win = tk.Toplevel(self.root)
        win.title("Yêu cầu mượn tài nguyên")
        win.resizable(False, False)
        win.transient(self.root)

        body = ttk.Frame(win, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Có máy xin dùng tài nguyên của máy bạn",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Separator(body).pack(fill="x", pady=8)

        grid = ttk.Frame(body)
        grid.pack(fill="x")
        rows = [("Máy điều phối", f"{controller} ({request.peer})"),
                ("Công việc", job)] + describe(resources)
        for i, (label, value) in enumerate(rows):
            ttk.Label(grid, text=label + ":").grid(row=i, column=0, sticky="nw", padx=(0, 12), pady=2)
            ttk.Label(grid, text=value, font=("Segoe UI", 9, "bold"),
                      wraplength=320, justify="left").grid(row=i, column=1, sticky="w", pady=2)

        if payload.get("note"):
            ttk.Label(body, text=payload["note"], wraplength=440,
                      justify="left").pack(anchor="w", pady=(10, 0))

        ttk.Label(body, text="Đây là bản demo: không có tài nguyên nào bị chiếm thật.",
                  foreground="#666").pack(anchor="w", pady=(10, 0))

        countdown = ttk.Label(body, text="")
        countdown.pack(anchor="w", pady=(6, 0))

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(14, 0))

        state: dict = {"done": False, "left": DECISION_TIMEOUT_S, "timers": []}

        def finish(decision: str, reason: str) -> None:
            if state["done"]:
                return
            state["done"] = True
            # Huy cac hen gio con cho, neu khong Tk se bao loi khi cua so da dong.
            for timer in state["timers"]:
                try:
                    win.after_cancel(timer)
                except tk.TclError:
                    pass
            self._close_dialog = None
            request.decision = decision
            request.reason = reason
            request.answered.set()
            self.record(request)
            win.grab_release()
            win.destroy()

        def tick() -> None:
            if state["done"]:
                return
            if state["left"] <= 0:
                finish("deny", "Hết thời gian chờ, tự động từ chối.")
                return
            countdown.configure(text=f"Tự động từ chối sau {state['left']} giây.")
            state["left"] -= 1
            state["timers"].append(win.after(1000, tick))

        ttk.Button(buttons, text="Từ chối",
                   command=lambda: finish("deny", "Chủ máy từ chối.")).pack(side="right")
        allow = ttk.Button(buttons, text="Cho phép",
                           command=lambda: finish("allow", "Chủ máy cho phép."))
        allow.pack(side="right", padx=(0, 8))

        win.protocol("WM_DELETE_WINDOW", lambda: finish("deny", "Chủ máy đóng cửa sổ."))
        self._close_dialog = finish
        win.grab_set()

        # Dua cua so len truoc mat chu may.
        self.root.deiconify()
        win.update_idletasks()
        win.lift()
        win.attributes("-topmost", True)
        state["timers"].append(win.after(400, lambda: win.attributes("-topmost", False)))
        allow.focus_force()
        win.bell()

        self.status.configure(text=f"Đang hỏi chủ máy về yêu cầu từ {controller}...")
        self.write_log(f"Nhận yêu cầu từ {controller} ({request.peer}).")
        tick()

    def record(self, request: PendingRequest) -> None:
        label = {"allow": "ĐÃ CHO PHÉP", "deny": "ĐÃ TỪ CHỐI"}.get(request.decision, "HẾT HẠN")
        self.write_log(f"{label}: {request.reason}")
        self.status.configure(text="Đang chờ yêu cầu từ máy điều phối...")
        entry = {"ts": datetime.now().isoformat(timespec="seconds"),
                 "machine": self.name, "peer": request.peer,
                 "decision": request.decision, "reason": request.reason,
                 "request": request.payload}
        try:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.write_log(f"Không ghi được nhật ký ra tệp: {exc}")

    # ---------------------------------------------------------------- vong lap

    def poll(self) -> None:
        try:
            request = self.pending.get_nowait()
        except queue.Empty:
            pass
        else:
            self.ask_owner(request)
        self._poll_timer = self.root.after(150, self.poll)

    def serve(self) -> None:
        handler = make_handler(self)
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.write_log(f"Agent sẵn sàng tại {lan_ip()}:{self.port}.")

    def quit(self) -> None:
        # Tra loi dut diem cho yeu cau dang cho, thay vi de may dieu phoi cho het gio.
        if self._close_dialog is not None:
            self._close_dialog("deny", "Agent đã đóng, tự động từ chối.")
        if self._poll_timer is not None:
            try:
                self.root.after_cancel(self._poll_timer)
            except tk.TclError:
                pass
            self._poll_timer = None
        if self.server:
            self.server.shutdown()
            self.server.server_close()      # tra lai cong, dong cac luong phuc vu
            self.server = None
        self.root.destroy()

    def run(self) -> None:
        self.serve()
        self._poll_timer = self.root.after(150, self.poll)
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent demo cho máy worker")
    parser.add_argument("--name", default=socket.gethostname(),
                        help="tên máy hiện trong giao diện (mặc định: tên máy tính)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="địa chỉ lắng nghe (mặc định: mọi card mạng)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=DEFAULT_TOKEN,
                        help="mã chia sẻ; máy điều phối phải gửi đúng mã này")
    args = parser.parse_args()

    try:
        AgentApp(args.name, args.host, args.port, args.token).run()
    except OSError as exc:
        print(f"Không mở được cổng {args.port}: {exc}")
        print("Có thể cổng đang bị một chương trình khác dùng; thử --port 8766.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
