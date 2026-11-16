"""The lab agent's window: what the machine owner sees and clicks.

Requests arrive on HTTP threads; the approver hands them to the Tk thread through a queue
and blocks until the owner answers, so every widget is touched from the Tk thread only.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Callable

from .agent_core import AgentCore, Approver
from .agent_server import make_server
from .protocol import APPROVAL_TIMEOUT_S, MODE_LABELS, describe_params, validate_params

POLL_MS = 150
REFRESH_MS = 1000


class AgentWindow:
    def __init__(self, core_factory: Callable[[Approver], AgentCore], host: str, port: int, token: str):
        self._pending: queue.Queue[dict] = queue.Queue()
        self._closing = False
        self._closed = False
        self._timers: dict[str, str] = {}
        self._close_dialog: Callable[[str, str], None] | None = None
        self._last_state: str | None = None

        self.core = core_factory(self._approve)
        self.server = make_server(self.core, host, port, token)
        self.port = self.server.server_address[1]

        self.root = tk.Tk()
        self.root.title(f"Agent LAB — {self.core.name}")
        self.root.geometry("660x440")
        self._build_ui()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        info = ttk.LabelFrame(frame, text="Máy này", padding=10)
        info.pack(fill="x")
        rows = [("Tên máy", self.core.name),
                ("Chủ máy", self.core.machine.owner or "—"),
                ("Địa chỉ cho máy điều phối", f"{self.core.machine.address or '?'}:{self.port}")]
        for i, (label, value) in enumerate(rows):
            ttk.Label(info, text=label + ":").grid(row=i, column=0, sticky="w", padx=(0, 10))
            ttk.Label(info, text=value, font=("Segoe UI", 9, "bold")).grid(row=i, column=1, sticky="w")

        self.status_label = ttk.Label(frame, text="", font=("Segoe UI", 10, "bold"))
        self.status_label.pack(fill="x", pady=(12, 6))

        log_box = ttk.LabelFrame(frame, text="Nhật ký", padding=6)
        log_box.pack(fill="both", expand=True)
        self.log = tk.Text(log_box, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_box, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Thoát", command=self._on_close).pack(side="right")
        self.stop_button = ttk.Button(buttons, text="Dừng huấn luyện", command=self._on_stop,
                                      state="disabled")
        self.stop_button.pack(side="right", padx=(0, 8))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_status()

    def write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- approval

    def _approve(self, payload: dict) -> tuple[str, str]:
        """Runs on an HTTP thread; blocks until the owner answers in the Tk thread."""
        if self._closing:
            return ("deny", "Agent đang đóng")
        pending = {"payload": payload, "event": threading.Event(),
                   "result": ("deny", "Hết thời gian chờ, tự động từ chối")}
        self._pending.put(pending)
        pending["event"].wait(APPROVAL_TIMEOUT_S + 5)
        return pending["result"]

    def _ask_owner(self, pending: dict) -> None:
        payload = pending["payload"]
        params = validate_params(payload["params"])  # already checked by the core
        others = [w for w in payload["workers"] if w != self.core.name]

        win = tk.Toplevel(self.root)
        win.title("Yêu cầu dùng GPU để huấn luyện")
        win.resizable(False, False)
        win.transient(self.root)
        body = ttk.Frame(win, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Máy điều phối xin dùng GPU của máy bạn",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Separator(body).pack(fill="x", pady=8)

        grid = ttk.Frame(body)
        grid.pack(fill="x")
        rows = [("Người yêu cầu", str(payload.get("controller", "không rõ")))]
        rows += describe_params(params)
        rows.append(("Cùng tham gia", ", ".join(others) or "—"))
        for i, (label, value) in enumerate(rows):
            ttk.Label(grid, text=label + ":").grid(row=i, column=0, sticky="nw", padx=(0, 12), pady=2)
            ttk.Label(grid, text=value, font=("Segoe UI", 9, "bold"), wraplength=320,
                      justify="left").grid(row=i, column=1, sticky="w", pady=2)
        ttk.Label(body, text="Bạn có thể dừng huấn luyện bất cứ lúc nào bằng nút "
                             "\"Dừng huấn luyện\" trên cửa sổ agent.",
                  foreground="#555", wraplength=420, justify="left").pack(anchor="w", pady=(10, 0))
        countdown = ttk.Label(body, text="")
        countdown.pack(anchor="w", pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(14, 0))

        state = {"done": False, "left": APPROVAL_TIMEOUT_S, "timers": []}

        def finish(decision: str, reason: str) -> None:
            if state["done"]:
                return
            state["done"] = True
            for timer in state["timers"]:   # pending callbacks would fire on a dead window
                try:
                    win.after_cancel(timer)
                except tk.TclError:
                    pass
            self._close_dialog = None
            pending["result"] = (decision, reason)
            pending["event"].set()
            self.write_log(("ĐÃ CHO PHÉP" if decision == "allow" else "ĐÃ TỪ CHỐI")
                           + f" lượt {payload['run_id']}: {reason}")
            try:
                win.grab_release()
                win.destroy()
            except tk.TclError:
                pass

        def tick() -> None:
            if state["done"]:
                return
            if state["left"] <= 0:
                finish("deny", "Hết thời gian chờ, tự động từ chối")
                return
            countdown.configure(text=f"Tự động từ chối sau {state['left']} giây.")
            state["left"] -= 1
            state["timers"].append(win.after(1000, tick))

        ttk.Button(buttons, text="Từ chối",
                   command=lambda: finish("deny", "Chủ máy từ chối")).pack(side="right")
        allow = ttk.Button(buttons, text="Cho phép", command=lambda: finish("allow", "Chủ máy cho phép"))
        allow.pack(side="right", padx=(0, 8))
        win.protocol("WM_DELETE_WINDOW", lambda: finish("deny", "Chủ máy đóng hộp thoại"))
        self._close_dialog = finish
        win.grab_set()

        self.root.deiconify()           # bring it in front of the owner
        win.update_idletasks()
        win.lift()
        win.attributes("-topmost", True)
        state["timers"].append(win.after(400, lambda: win.attributes("-topmost", False)))
        allow.focus_force()
        win.bell()
        self.write_log(f"Nhận yêu cầu lượt {payload['run_id']} từ {payload.get('controller', '?')}.")
        tick()

    # ---------------------------------------------------------------- status and stop

    def _status_text(self, state: str) -> str:
        run_id, params = self.core.run_id, self.core.params
        if state == "RUNNING" and params is not None:
            status = self.core.status(run_id)
            epoch = status["epoch"] or 0
            total = status["total_epochs"] or params.max_epochs
            return (f"Đang được dùng để huấn luyện: {MODE_LABELS[params.mode]}, {params.model}, "
                    f"epoch {epoch}/{total}")
        return {"IDLE": "Rảnh — đang chờ yêu cầu từ máy điều phối",
                "ASKING": "Đang chờ bạn quyết định",
                "APPROVED": "Đã cho phép — chờ máy điều phối bắt đầu",
                "FINISHED": f"Lượt {run_id} đã xong. Rảnh.",
                "FAILED": f"Lượt {run_id} gặp lỗi. Rảnh.",
                "STOPPED": f"Lượt {run_id} đã dừng. Rảnh."}.get(state, state)

    def _refresh_status(self) -> None:
        state = self.core.state
        self.status_label.configure(text=self._status_text(state))
        self.stop_button.configure(state="normal" if state == "RUNNING" else "disabled")
        if state != self._last_state and state in ("RUNNING", "FINISHED", "FAILED", "STOPPED"):
            self.write_log({"RUNNING": f"Bắt đầu huấn luyện lượt {self.core.run_id}.",
                            "FINISHED": "Huấn luyện xong.", "FAILED": "Huấn luyện gặp lỗi.",
                            "STOPPED": "Huấn luyện đã dừng."}[state])
        self._last_state = state
        self._timers["refresh"] = self.root.after(REFRESH_MS, self._refresh_status)

    def _on_stop(self) -> None:
        if self.core.state != "RUNNING":
            return
        if not messagebox.askyesno("Dừng huấn luyện",
                                   "Dừng huấn luyện trên máy này? Cả lượt chạy trên các máy khác "
                                   "cũng sẽ dừng.", parent=self.root):
            return
        self.core.stop(self.core.run_id, "Chủ máy dừng", by="owner")
        self.write_log("Bạn đã dừng huấn luyện.")
        self.stop_button.configure(state="disabled")

    def _on_close(self) -> None:
        if self.core.state == "RUNNING" and not messagebox.askyesno(
                "Thoát agent", "Máy đang huấn luyện. Thoát sẽ dừng cả lượt chạy. Thoát?",
                parent=self.root):
            return
        self.quit()

    # ---------------------------------------------------------------- loop

    def _poll(self) -> None:
        try:
            pending = self._pending.get_nowait()
        except queue.Empty:
            pass
        else:
            self._ask_owner(pending)
        self._timers["poll"] = self.root.after(POLL_MS, self._poll)

    def serve(self) -> None:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._timers["poll"] = self.root.after(POLL_MS, self._poll)
        self.write_log(f"Agent sẵn sàng tại {self.core.machine.address or '?'}:{self.port}.")

    def run(self) -> None:
        self.serve()
        self.root.mainloop()

    def quit(self) -> None:
        if self._closed:
            return
        self._closing = True
        if self._close_dialog is not None:
            self._close_dialog("deny", "Agent đã đóng, tự động từ chối")
        while True:     # answer requests that never got a dialog
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                break
            pending["result"] = ("deny", "Agent đã đóng, tự động từ chối")
            pending["event"].set()
        for timer in self._timers.values():
            try:
                self.root.after_cancel(timer)
            except tk.TclError:
                pass
        self.core.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self._closed = True
        self.root.destroy()
