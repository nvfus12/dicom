"""The lab controller window on machine C.

Network calls run on background threads; they hand results back through a queue that the
Tk thread drains, so widgets are only touched from the Tk thread.
"""
from __future__ import annotations

import csv
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from ..app.config_loader import PROJECT_DIR
from .controller_core import AgentClient, RunSession, load_lab
from .protocol import MODE_LABELS, POLL_INTERVAL_S, ProtocolError, RunParams, rank_order, validate_params

DATASETS = {"CIFAR-10": "cifar10"}
MODELS = {"ResNet-20": "resnet20", "ResNet-18": "resnet18"}
QUICK_DATASET, QUICK_MODEL = "Dữ liệu tổng hợp", "Tiny CNN"
STATE_LABELS = {"RUNNING": "đang chạy", "FINISHED": "xong", "FAILED": "lỗi", "STOPPED": "đã dừng",
                "UNREACHABLE": "mất liên lạc", "LOST": "mất lượt chạy", "IDLE": "rảnh",
                "ASKING": "đang hỏi chủ máy", "APPROVED": "đã cho phép"}
DRAIN_MS = 100


class ControllerWindow:
    def __init__(self, lab_path: Path, prototype_dir: Path = PROJECT_DIR,
                 session_factory=RunSession, client_factory=AgentClient,
                 results_root: Path | None = None, quick_root: Path | None = None,
                 poll_s: float = POLL_INTERVAL_S):
        self.prototype_dir = prototype_dir
        self.session_factory = session_factory
        self.results_root = results_root or prototype_dir / "results"
        self.quick_root = quick_root or prototype_dir / "results-thu"
        self.poll_s = poll_s
        refs, token = load_lab(lab_path)
        self.clients = [client_factory(ref, token) for ref in refs]
        self.selected = {c.name for c in self.clients}
        self.infos: dict[str, dict] = {}
        self.session = None
        self._busy = False          # a run is being prepared, running, or collected
        self._stop_requested = False
        self._closing = False
        self._inbox: queue.Queue[Callable[[], None]] = queue.Queue()
        self._drain_timer: str | None = None

        self.root = tk.Tk()
        self.root.title("Điều khiển huấn luyện LAB")
        self.root.geometry("980x860")
        self._build_ui()

    # ---------------------------------------------------------------- layout

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        workers = ttk.LabelFrame(outer, text="Worker (bấm vào dòng để chọn / bỏ chọn)", padding=6)
        workers.pack(fill="x")
        columns = ("use", "address", "gpu", "vram", "state", "rank")
        self.worker_tree = ttk.Treeview(workers, columns=columns, height=4)
        for col, title, width in (("#0", "Máy", 140), ("use", "Dùng", 50), ("address", "Địa chỉ", 150),
                                  ("gpu", "GPU", 220), ("vram", "VRAM", 80), ("state", "Trạng thái", 140),
                                  ("rank", "Rank", 50)):
            self.worker_tree.heading(col, text=title)
            self.worker_tree.column(col, width=width, anchor="w")
        for client in self.clients:
            self.worker_tree.insert("", "end", iid=client.name, text=client.name,
                                    values=("✓", f"{client.ref.address}:{client.ref.port}", "", "",
                                            "đang kiểm tra...", ""))
        self.worker_tree.pack(side="left", fill="x", expand=True)
        self.worker_tree.bind("<ButtonRelease-1>", self._toggle_worker)
        ttk.Button(workers, text="Làm mới", command=self.refresh_workers).pack(side="right", padx=6)

        form = ttk.LabelFrame(outer, text="Cấu hình lượt chạy", padding=8)
        form.pack(fill="x", pady=(8, 0))
        self.dataset_var = tk.StringVar(value="CIFAR-10")
        self.model_var = tk.StringVar(value="ResNet-20")
        self.mode_var = tk.StringVar(value="ddp")
        self.h_var = tk.StringVar(value="256")
        self.minutes_var = tk.StringVar(value="30")
        self.epochs_var = tk.StringVar(value="30")
        self.batch_var = tk.StringVar(value="128")
        self.lr_var = tk.StringVar(value="0.1")
        self.comm_var = tk.StringVar(value="fp32")
        self.quick_var = tk.BooleanVar(value=False)

        ttk.Label(form, text="Dataset").grid(row=0, column=0, sticky="w")
        self.dataset_box = ttk.Combobox(form, textvariable=self.dataset_var, values=list(DATASETS),
                                        state="readonly", width=18)
        self.dataset_box.grid(row=0, column=1, sticky="w", padx=(4, 16))
        ttk.Label(form, text="Model").grid(row=0, column=2, sticky="w")
        self.model_box = ttk.Combobox(form, textvariable=self.model_var, values=list(MODELS),
                                      state="readonly", width=14)
        self.model_box.grid(row=0, column=3, sticky="w", padx=(4, 16))

        ttk.Label(form, text="Chế độ").grid(row=1, column=0, sticky="w", pady=(6, 0))
        modes = ttk.Frame(form)
        modes.grid(row=1, column=1, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Radiobutton(modes, text="DDP", value="ddp", variable=self.mode_var).pack(side="left")
        ttk.Radiobutton(modes, text="Local SGD", value="localsgd",
                        variable=self.mode_var).pack(side="left", padx=(12, 0))
        ttk.Label(modes, text="   H (bước)").pack(side="left")
        self.h_entry = ttk.Entry(modes, textvariable=self.h_var, width=7)
        self.h_entry.pack(side="left", padx=(4, 0))

        ttk.Label(form, text="Thời gian tối đa (phút)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.minutes_var, width=8).grid(row=2, column=1, sticky="w",
                                                                     padx=(4, 0), pady=(6, 0))
        ttk.Label(form, text="Epoch tối đa").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.epochs_var, width=8).grid(row=2, column=3, sticky="w",
                                                                    padx=(4, 0), pady=(6, 0))

        self.advanced = ttk.Frame(form)
        ttk.Label(self.advanced, text="Batch mỗi máy").pack(side="left")
        ttk.Entry(self.advanced, textvariable=self.batch_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(self.advanced, text="LR").pack(side="left")
        ttk.Entry(self.advanced, textvariable=self.lr_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(self.advanced, text="Gửi qua mạng").pack(side="left")
        ttk.Combobox(self.advanced, textvariable=self.comm_var, values=["fp32", "fp16"], width=6,
                     state="readonly").pack(side="left", padx=(4, 12))
        ttk.Radiobutton(self.advanced, text="Sync (dự phòng khi DDP lỗi)", value="sync",
                        variable=self.mode_var).pack(side="left")
        self.advanced_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Nâng cao", variable=self.advanced_shown,
                        command=self._toggle_advanced).grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(form, text="Chạy thử nhanh (dữ liệu tổng hợp, 1–2 phút, không đưa vào báo cáo)",
                        variable=self.quick_var).grid(row=5, column=0, columnspan=4, sticky="w",
                                                      pady=(6, 0))

        actions = ttk.Frame(form)
        actions.grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.send_button = ttk.Button(actions, text="Gửi yêu cầu", command=self.send_request)
        self.send_button.pack(side="left")
        self.start_button = ttk.Button(actions, text="Start", command=self.start_run, state="disabled")
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(actions, text="Dừng", command=self.stop_run, state="disabled")
        self.stop_button.pack(side="left")

        self.mode_var.trace_add("write", lambda *_: self._sync_form())
        self.quick_var.trace_add("write", lambda *_: self._sync_form())
        self._sync_form()

        progress = ttk.LabelFrame(outer, text="Tiến độ", padding=6)
        progress.pack(fill="both", expand=True, pady=(8, 0))
        self.machines_label = ttk.Label(progress, text="Chưa có lượt chạy.")
        self.machines_label.pack(anchor="w")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(8, 2.4), dpi=90)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=progress)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._plot([])

        results = ttk.LabelFrame(outer, text="Kết quả các lượt (results/report-lab/runs.csv)", padding=6)
        results.pack(fill="x", pady=(8, 0))
        self.results_tree = ttk.Treeview(results, show="headings", height=4)
        self.results_tree.pack(side="left", fill="x", expand=True)
        ttk.Button(results, text="Mở thư mục kết quả", command=self.open_results).pack(side="right", padx=6)

        log_box = ttk.LabelFrame(outer, text="Nhật ký", padding=6)
        log_box.pack(fill="both", pady=(8, 0))
        self.log = tk.Text(log_box, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.load_results_table()

    def _toggle_advanced(self) -> None:
        if self.advanced_shown.get():
            self.advanced.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        else:
            self.advanced.grid_remove()

    def _sync_form(self) -> None:
        self.h_entry.configure(state="normal" if self.mode_var.get() == "localsgd" else "disabled")
        if self.quick_var.get():
            self.dataset_var.set(QUICK_DATASET)
            self.model_var.set(QUICK_MODEL)
            self.dataset_box.configure(state="disabled")
            self.model_box.configure(state="disabled")
        else:
            if self.dataset_var.get() not in DATASETS:
                self.dataset_var.set("CIFAR-10")
            if self.model_var.get() not in MODELS:
                self.model_var.set("ResNet-20")
            self.dataset_box.configure(state="readonly")
            self.model_box.configure(state="readonly")

    def write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- threading

    def _post(self, fn: Callable[[], None]) -> None:
        self._inbox.put(fn)

    def _drain(self) -> None:
        while True:
            try:
                fn = self._inbox.get_nowait()
            except queue.Empty:
                break
            if not self._closing:
                fn()
        if not self._closing:
            self._drain_timer = self.root.after(DRAIN_MS, self._drain)

    def _background(self, target: Callable[[], None]) -> None:
        threading.Thread(target=target, daemon=True).start()

    def _on_event(self, kind: str, data: dict) -> None:
        """Called by RunSession on background threads."""
        if kind == "decision":
            word = "ĐỒNG Ý" if data["decision"] == "allow" else "TỪ CHỐI"
            text = f"{data['machine']}: {word}" + (f" ({data['reason']})" if data.get("reason") else "")
            self._post(lambda: self.write_log(text))
        elif kind in ("stopping", "analyze_failed", "collect_failed", "stop_failed", "resume_missing"):
            text = {"stopping": lambda: f"Dừng lượt: {data['reason']}",
                    "analyze_failed": lambda: f"Phân tích lỗi: {data['output'][-500:]}",
                    "collect_failed": lambda: f"Không tải được kết quả của {data['machine']}: {data['error']}",
                    "stop_failed": lambda: f"Không gửi được lệnh dừng tới {data['machine']}: {data['error']}",
                    "resume_missing": lambda: f"Không có trong lab.yaml: {data['machines']}"}[kind]()
            self._post(lambda: self.write_log(text))

    # ---------------------------------------------------------------- workers

    def _toggle_worker(self, event) -> None:
        name = self.worker_tree.identify_row(event.y)
        if not name or self._busy:
            return
        if name in self.selected:
            self.selected.discard(name)
        else:
            self.selected.add(name)
        self._show_workers()

    def refresh_workers(self) -> None:
        def work():
            for client in self.clients:
                try:
                    info = client.info()
                except Exception as e:
                    info = {"name": client.name, "error": str(e)}
                self._post(lambda c=client, i=info: self._set_info(c.name, i))
        self._background(work)

    def _set_info(self, name: str, info: dict) -> None:
        self.infos[name] = info
        self._show_workers()

    def _show_workers(self) -> None:
        online = [i for n, i in self.infos.items() if n in self.selected and "error" not in i]
        ranks = {n: r for r, n in enumerate(rank_order(online))}
        for client in self.clients:
            info = self.infos.get(client.name, {})
            gpu = info.get("gpu") or {}
            vram = f"{gpu['vram_mb'] / 1024:.1f} GB" if gpu.get("vram_mb") else "—"
            state = ("không kết nối được" if "error" in info
                     else STATE_LABELS.get(info.get("state"), info.get("state", "đang kiểm tra...")))
            self.worker_tree.item(client.name, values=(
                "✓" if client.name in self.selected else "", f"{client.ref.address}:{client.ref.port}",
                gpu.get("name") or ("—" if info else ""), vram if info else "", state,
                ranks.get(client.name, "")))

    # ---------------------------------------------------------------- form

    def collect_params(self) -> RunParams:
        def number(var: tk.StringVar, label: str, kind=int):
            try:
                return kind(var.get().strip().replace(",", "."))
            except ValueError:
                raise ProtocolError(f"Ô \"{label}\" phải là số, đang là \"{var.get()}\"") from None
        quick = self.quick_var.get()
        data = {"mode": self.mode_var.get(), "quick": quick,
                "dataset": "synthetic" if quick else DATASETS.get(self.dataset_var.get(), "cifar10"),
                "model": "tiny_cnn" if quick else MODELS.get(self.model_var.get(), "resnet20"),
                "max_train_minutes": number(self.minutes_var, "Thời gian tối đa", float),
                "max_epochs": number(self.epochs_var, "Epoch tối đa"),
                "batch_size": number(self.batch_var, "Batch mỗi máy"),
                "lr": number(self.lr_var, "LR", float),
                "comm_dtype": self.comm_var.get()}
        if data["mode"] == "localsgd":
            data["local_steps"] = number(self.h_var, "H")
        return validate_params(data)

    def _set_buttons(self, send: bool, start: bool, stop: bool) -> None:
        self.send_button.configure(state="normal" if send else "disabled")
        self.start_button.configure(state="normal" if start else "disabled")
        self.stop_button.configure(state="normal" if stop else "disabled")

    # ---------------------------------------------------------------- run

    def send_request(self) -> None:
        try:
            params = self.collect_params()
        except ProtocolError as e:
            messagebox.showerror("Tham số không hợp lệ", str(e), parent=self.root)
            return
        clients = [c for c in self.clients if c.name in self.selected]
        if len(clients) < 2:
            messagebox.showerror("Chưa đủ máy", "Cần chọn ít nhất 2 máy.", parent=self.root)
            return
        self._release_unstarted()
        session = self.session_factory(clients, params, self.results_root, self.quick_root,
                                       self.prototype_dir, on_event=self._on_event)
        self.session = session
        self._busy = True
        self._set_buttons(False, False, False)
        mode = MODE_LABELS[params.mode] + (f", H={params.local_steps}" if params.mode == "localsgd" else "")
        self.write_log(f"Gửi yêu cầu: {mode}, {params.model}" + (" (chạy thử)" if params.quick else ""))

        def work():
            try:
                session.check()
                self._post(lambda: self.write_log(f"Lượt {session.run_id}, thứ tự rank: "
                                                  f"{', '.join(session.workers)}. Đang chờ chủ máy..."))
                session.request_all()
            except Exception as e:
                self._post(lambda: self._request_failed(str(e)))
                return
            self._post(lambda: self._request_done(session))
        self._background(work)

    def _release_unstarted(self, wait_s: float = 0) -> None:
        """An approved run that was never started keeps the machines busy; let them go."""
        old = self.session
        if old is None or not old.approved or getattr(old, "started", False):
            return
        old.approved = False
        thread = threading.Thread(target=lambda: old.cancel("Người điều phối huỷ lượt chưa bắt đầu"),
                                  daemon=True)
        thread.start()
        if wait_s:
            thread.join(wait_s)

    def _request_failed(self, message: str) -> None:
        self._busy = False
        self.write_log(f"Không gửi được yêu cầu: {message}")
        self._set_buttons(True, False, False)
        messagebox.showerror("Không gửi được yêu cầu", message, parent=self.root)

    def _request_done(self, session) -> None:
        self._busy = False
        if session is not self.session:
            return
        if session.approved:
            self.write_log("Mọi máy đã đồng ý. Bấm Start để bắt đầu.")
            self._set_buttons(True, True, False)
        else:
            self.write_log("Không đủ máy đồng ý; lượt bị huỷ.")
            self._set_buttons(True, False, False)

    def start_run(self) -> None:
        session = self.session
        if session is None or not session.approved:
            return
        self._busy = True
        self._stop_requested = False
        self._set_buttons(False, False, True)
        self.write_log(f"Bắt đầu lượt {session.run_id}.")

        def work():
            try:
                session.start()
            except Exception as e:
                self._post(lambda: self._run_failed(str(e)))
                return
            self._monitor(session)
        self._background(work)

    def _monitor(self, session) -> None:
        """Background loop: poll until every machine is done, then collect and analyze."""
        try:
            while not self._closing:
                statuses = session.poll_once()
                self._post(lambda s=statuses: self._show_progress(session, s))
                if session.done(statuses):
                    break
                time.sleep(self.poll_s)
            if self._closing:
                return
            self._post(lambda: self.write_log("Huấn luyện kết thúc, đang gom kết quả..."))
            run_dir = session.collect()
            report = session.analyze()
        except Exception as e:
            self._post(lambda: self._run_failed(str(e)))
            return
        self._post(lambda: self._run_done(run_dir, report))

    def _show_progress(self, session, statuses: dict) -> None:
        parts = []
        for name in session.workers:
            st = statuses.get(name, {})
            text = f"{name}: {STATE_LABELS.get(st.get('state'), st.get('state'))}"
            if st.get("epoch"):
                text += f", epoch {st['epoch']}/{st.get('total_epochs') or '?'}"
            if st.get("test_acc") is not None:
                text += f", acc {st['test_acc']:.3f}"
            parts.append(text)
        self.machines_label.configure(text="   |   ".join(parts))
        rank0 = statuses.get(session.workers[0], {}) if session.workers else {}
        self._plot(rank0.get("points") or [])

    def _plot(self, points: list) -> None:
        self.axes.clear()
        self.axes.set_xlabel("Thời gian huấn luyện (phút)")
        self.axes.set_ylabel("Accuracy (test)")
        self.axes.grid(True, alpha=0.3)
        pts = [(t / 60, a) for t, a in points if t is not None and a is not None]
        if pts:
            self.axes.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _run_failed(self, message: str) -> None:
        self._busy = False
        self.write_log(f"Lượt chạy lỗi: {message}")
        self._set_buttons(True, False, False)

    def _run_done(self, run_dir: Path, report: Path | None) -> None:
        self._busy = False
        self.write_log(f"Đã gom kết quả vào {run_dir}.")
        if report is not None:
            self.write_log(f"Đã phân tích, báo cáo ở {report}.")
            self.load_results_table()
        self._set_buttons(True, False, False)
        self.refresh_workers()

    def stop_run(self) -> None:
        session = self.session
        if session is None:
            return
        if not messagebox.askyesno("Dừng lượt chạy", "Dừng huấn luyện trên mọi máy?", parent=self.root):
            return
        self.stop_button.configure(state="disabled")
        self._background(lambda: session.stop_all("Người điều phối dừng lượt chạy"))

    # ---------------------------------------------------------------- results

    def load_results_table(self) -> None:
        path = self.results_root / "report-lab" / "runs.csv"
        self.results_tree.delete(*self.results_tree.get_children())
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        columns = list(rows[0])
        self.results_tree.configure(columns=columns)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=110, anchor="w")
        for row in rows:
            self.results_tree.insert("", "end", values=[row[c] for c in columns])

    def open_results(self) -> None:
        folder = self.results_root / "report-lab"
        if not folder.is_dir():
            folder = self.results_root
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # Windows only, like the rest of the lab setup

    # ---------------------------------------------------------------- lifecycle

    def start_background(self) -> None:
        """Starts the inbox loop, the first worker check, and picks up an unfinished run."""
        self._drain_timer = self.root.after(DRAIN_MS, self._drain)
        self.refresh_workers()

        def resume():
            session = self.session_factory.resume(self.clients, self.results_root, self.quick_root,
                                                  self.prototype_dir, on_event=self._on_event)
            if session is not None:
                self._post(lambda: self._resumed(session))
        self._background(resume)

    def _resumed(self, session) -> None:
        self.session = session
        self._busy = True
        self._set_buttons(False, False, True)
        self.write_log(f"Nối lại lượt {session.run_id} (các máy vẫn đang giữ lượt này).")
        self._background(lambda: self._monitor(session))

    def run(self) -> None:
        self.start_background()
        self.root.mainloop()

    def _on_close(self) -> None:
        if self._busy and not messagebox.askyesno(
                "Thoát", "Đang có lượt chạy. Các máy sẽ tiếp tục huấn luyện; mở lại giao diện để "
                         "gom kết quả. Thoát?", parent=self.root):
            return
        self.quit()

    def quit(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._release_unstarted(wait_s=5)
        if self._drain_timer is not None:
            try:
                self.root.after_cancel(self._drain_timer)
            except tk.TclError:
                pass
        self.root.destroy()
