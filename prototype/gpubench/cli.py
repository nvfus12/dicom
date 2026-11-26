"""Command line entry point: python -m gpubench <command> ..."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .app.config_loader import (CONFIGS_DIR, RESULTS_DIR, list_machines, load_experiment,
                                load_machine, parse_overrides)
from .domain.config import ConfigError


def cmd_run(args) -> int:
    from .app.bootstrap import run

    spec = load_experiment(args.experiment, parse_overrides(args.set), runner=args.machine)
    extra = {"results_dir": Path(args.results_dir)} if args.results_dir else {}
    run(args.machine, spec, run_id=args.run_id, master_addr=args.master_addr, **extra)
    return 0


def cmd_plan(args) -> int:
    """Shows ranks, rendezvous and the exact command each machine must run."""
    spec = load_experiment(args.experiment, parse_overrides(args.set), runner=args.machine)
    print(f"Thí nghiệm: {spec.name}  mode={spec.mode}  model={spec.model}  "
          f"batch/worker={spec.batch_size}  lr hiệu dụng={spec.effective_lr():g}  "
          f"fingerprint={spec.fingerprint()}")
    sets = " ".join(f"--set {s}" for s in args.set)
    problems = []
    for rank, name in enumerate(spec.workers):
        try:
            m = load_machine(name)
            addr = m.address or "CHƯA CÓ"
            if rank == 0 and not m.address and spec.world_size > 1:
                problems.append(f"rank 0 ({name}) chưa có address")
            limit = m.max_batch.get(spec.model)
            if limit is not None and spec.batch_size > limit:
                problems.append(f"{name}: batch {spec.batch_size} > max_batch {limit}")
            print(f"  rank {rank}: {name:<14} owner={m.owner or '-':<12} device={m.device} "
                  f"address={addr}")
        except ConfigError as e:
            problems.append(str(e))
            print(f"  rank {rank}: {name:<14} LỖI: {e}")
    if spec.world_size > 1:
        print(f"Rendezvous: rank 0, cổng TCP {spec.master_port} (mở inbound trên máy rank 0)")
    print("Lệnh chạy trên từng máy (thứ tự khởi động không quan trọng):")
    for name in spec.workers:
        print(f"  [{name}] python -m gpubench run --machine {name} --experiment {args.experiment} {sets}".rstrip())
    if problems:
        print("Cần sửa trước khi chạy:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


def cmd_check(args) -> int:
    """Smoke test of the local machine against its profile."""
    import torch

    from .adapters.device import build_device
    from .adapters.monitor import build_monitor

    m = load_machine(args.machine)
    print(f"Máy: {m.name}  owner={m.owner}  address={m.address}")
    print(f"Python {platform.python_version()}  torch {torch.__version__}  "
          f"cuda build {torch.version.cuda}  cuda available {torch.cuda.is_available()}")
    print(f"gloo available: {torch.distributed.is_gloo_available()}")
    try:
        # PyTorch 2.10 on Windows fails here for every address (seen on the controller laptop).
        torch.distributed.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        print("gloo device: OK")
    except Exception as e:
        print(f"gloo device: LỖI ({e}). Bản PyTorch này không chạy được nhiều máy; "
              "dùng bản đã kiểm chứng trong README (2.6.0).")
    device = build_device(m)
    print(f"Device: {device.describe()}")
    x = torch.randn(1024, 1024, device=device.torch_device)
    (x @ x).sum().item()
    print("Phép nhân ma trận thử: OK")
    if m.device == "cuda":
        print(f"NVML: {build_monitor(m.device, m.cuda_index).sample() or 'không có (cài nvidia-ml-py)'}")
    return 0


def cmd_profile(args) -> int:
    """Finds the largest batch that fits in memory and its throughput, per model."""
    import time

    import torch

    from .adapters.device import build_device
    from .adapters.models import build_model

    m = load_machine(args.machine)
    device = build_device(m, args.amp)
    print(f"Máy {m.name}: {device.describe()}")
    suggestion = {}
    rows: dict[str, list] = {}
    for model_name in args.models:
        rows[model_name] = []
        print(f"== {model_name} (amp={device.amp})")
        model = build_model(model_name, 10).to(device.torch_device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler(device.torch_device.type, enabled=device.amp)
        best = None
        for bs in args.batches:
            try:
                x = torch.randn(bs, 3, 32, 32, device=device.torch_device)
                y = torch.randint(0, 10, (bs,), device=device.torch_device)
                device.reset_peak_memory()
                for i in range(args.steps + 2):        # first 2 steps are warm-up
                    if i == 2:
                        device.synchronize()
                        t0 = time.perf_counter()
                    with torch.autocast(device.torch_device.type, enabled=device.amp):
                        loss = torch.nn.functional.cross_entropy(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                device.synchronize()
                rate = bs * args.steps / (time.perf_counter() - t0)
                peak = device.peak_memory_bytes()
                peak_txt = f"{peak / 2**20:.0f} MB" if peak is not None else "n/a"
                print(f"  batch {bs:>4}: {rate:8.0f} mẫu/s, VRAM peak {peak_txt}")
                rows[model_name].append({"batch": bs, "samples_per_s": round(rate, 1),
                                         "vram_peak_mb": round(peak / 2**20) if peak is not None else None})
                best = bs
            except torch.OutOfMemoryError:
                print(f"  batch {bs:>4}: hết VRAM")
                rows[model_name].append({"batch": bs, "out_of_memory": True})
                break
            finally:
                if device.torch_device.type == "cuda":
                    torch.cuda.empty_cache()
        if best is not None:
            suggestion[model_name] = best
    print("Gợi ý cho file máy (nên chừa dư VRAM cho chủ máy, chọn nhỏ hơn 1 bậc nếu cần):")
    print(f"  max_batch: {suggestion}")
    if args.json_out:
        import json
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"machine": m.name, "device": device.describe(), "models": rows, "suggestion": suggestion},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


def cmd_netcheck(args) -> int:
    from .app.netcheck import run_netcheck

    spec = load_experiment(args.experiment, parse_overrides(args.set))
    extra = {"results_dir": Path(args.results_dir)} if args.results_dir else {}
    run_netcheck(args.machine, spec, master_addr=args.master_addr, run_id=args.run_id, **extra)
    return 0


def cmd_glootest(args) -> int:
    from .app.glootest import run_glootest

    spec = load_experiment(args.experiment, parse_overrides(args.set))
    return run_glootest(args.machine, spec, args.master_addr, args.port_base, args.wait,
                        self_test=args.self_test, toward=args.toward,
                        json_out=Path(args.json_out) if args.json_out else None)


def cmd_porttest(args) -> int:
    from .app import porttest

    if args.porttest_action == "pair":
        return porttest.pair(args.machine, args.rank, args.peer, args.port, args.window,
                             Path(args.json_out) if args.json_out else None)
    if args.porttest_action == "listen":
        return porttest.listen(args.port, args.seconds)
    return porttest.probe(args.host, args.port, args.timeout)


def cmd_machines(args) -> int:
    for name in list_machines():
        m = load_machine(name)
        print(f"{name:<14} owner={m.owner or '-':<12} device={m.device:<5} "
              f"address={m.address or 'CHƯA CÓ'}  max_batch={m.max_batch or 'chưa profile'}")
    return 0


def cmd_agent(args) -> int:
    from .control.agent_core import AgentCore
    from .control.agent_ui import AgentWindow

    machine = load_machine(args.machine)
    print(f"Agent {machine.name}: máy điều phối gọi tới {machine.address or '?'}:{args.port}")
    print("Đang kiểm tra GPU (lần đầu có thể mất vài chục giây)...", flush=True)
    AgentWindow(lambda approver: AgentCore(machine.name, approver),
                args.host, args.port, args.token).run()
    return 0


def cmd_controller(args) -> int:
    from .control.controller_ui import ControllerWindow
    from .control.protocol import ProtocolError

    try:
        window = ControllerWindow(Path(args.lab))
    except ProtocolError as e:
        print(f"Lỗi cấu hình: {e}", file=sys.stderr)
        return 2
    window.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page that cannot print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="gpubench")
    sub = parser.add_subparsers(dest="command", required=True)

    def experiment_args(p):
        p.add_argument("--experiment", "-e", required=True,
                       help="tên file trong configs/experiments hoặc đường dẫn .yaml")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="ghi đè tham số thí nghiệm; phải giống nhau trên mọi máy")

    p = sub.add_parser("run", help="chạy thí nghiệm trên máy này")
    p.add_argument("--machine", "-m", required=True, help=f"tên file trong {CONFIGS_DIR / 'machines'}")
    experiment_args(p)
    p.add_argument("--run-id", default=None, help="mặc định: <tên thí nghiệm>-<thời gian> do rank 0 đặt")
    p.add_argument("--master-addr", default=None,
                   help="IP của máy rank 0; nếu có thì không cần file máy của rank 0")
    p.add_argument("--results-dir", default=None, help=f"mặc định: {RESULTS_DIR}")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("plan", help="xem rank, địa chỉ và lệnh cho từng máy")
    p.add_argument("--machine", "-m", default=None, help="chỉ cần cho thí nghiệm single")
    experiment_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("check", help="kiểm tra môi trường của máy này")
    p.add_argument("--machine", "-m", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("profile", help="đo batch tối đa vừa VRAM và tốc độ trên máy này")
    p.add_argument("--machine", "-m", required=True)
    p.add_argument("--models", nargs="+", default=["resnet20", "resnet18"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--amp", action="store_true", help="đo với mixed precision")
    p.add_argument("--json-out", default=None, help="ghi kết quả ra file JSON (dùng bởi agent)")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("netcheck", help="đo độ trễ và thông lượng giữa các máy của một thí nghiệm")
    p.add_argument("--machine", "-m", required=True)
    experiment_args(p)
    p.add_argument("--master-addr", default=None, help="IP của máy rank 0")
    p.add_argument("--run-id", default=None, help="mặc định: netcheck-<thời gian> do rank 0 đặt")
    p.add_argument("--results-dir", default=None, help=f"mặc định: {RESULTS_DIR}")
    p.set_defaults(func=cmd_netcheck)

    p = sub.add_parser("glootest", help="thử các cách mở kết nối gloo giữa các máy, in ĐẠT / LỖI")
    p.add_argument("--machine", "-m", required=True)
    p.add_argument("--experiment", "-e", default="run2_ddp_x2",
                   help="lấy danh sách máy (thứ tự rank) từ thí nghiệm này")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--master-addr", default=None,
                   help="IPv4 của máy rank 0 (giống nhau trên mọi máy); không cần khi --self")
    p.add_argument("--port-base", type=int, default=29600, help="mỗi cách dùng một cổng: port-base, +1, ...")
    p.add_argument("--wait", type=int, default=90, help="giây chờ máy kia ở mỗi cách")
    p.add_argument("--self", dest="self_test", action="store_true",
                   help="tự thử trên một máy: hai tiến trình nói chuyện qua IPv4 thật của máy này")
    p.add_argument("--toward", default="192.168.0.1",
                   help="với --self: lấy IPv4 của card dùng để đi tới địa chỉ này (mặc định router LAB)")
    p.add_argument("--json-out", default=None, help="ghi kết quả ra file JSON (dùng bởi agent)")
    p.set_defaults(func=cmd_glootest)

    p = sub.add_parser("porttest", help="thử máy kia có mở được kết nối TCP tới Python trên máy này")
    porttest_sub = p.add_subparsers(dest="porttest_action", required=True)
    q = porttest_sub.add_parser("listen", help="chạy trên máy được gọi tới")
    q.add_argument("--port", type=int, default=45000)
    q.add_argument("--seconds", type=int, default=180)
    q = porttest_sub.add_parser("probe", help="chạy trên máy gọi đi")
    q.add_argument("host", help="IPv4 của máy đang chạy listen")
    q.add_argument("--port", type=int, default=45000)
    q.add_argument("--timeout", type=float, default=5)
    q = porttest_sub.add_parser("pair", help="thử cả hai chiều cùng lúc (dùng bởi agent)")
    q.add_argument("--machine", "-m", required=True)
    q.add_argument("--rank", type=int, choices=[0, 1], required=True)
    q.add_argument("--peer", required=True, help="IPv4 của máy kia")
    q.add_argument("--port", type=int, default=45000, help="rank 0 nghe ở cổng này, rank 1 ở cổng +1")
    q.add_argument("--window", type=float, default=45)
    q.add_argument("--json-out", default=None)
    p.set_defaults(func=cmd_porttest)

    p = sub.add_parser("agent", help="chạy agent LAB trên worker (cửa sổ xin phép, nhận lượt chạy)")
    p.add_argument("--machine", "-m", required=True, help="tên file máy của chính máy này")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", default="lab-dicom", help="phải giống token trong configs/lab.yaml")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("controller", help="mở giao diện điều khiển LAB trên máy điều phối")
    p.add_argument("--lab", default=str(CONFIGS_DIR / "lab.yaml"), help="danh sách agent và token")
    p.set_defaults(func=cmd_controller)

    p = sub.add_parser("machines", help="liệt kê các máy đã cấu hình")
    p.set_defaults(func=cmd_machines)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"Lỗi cấu hình: {e}", file=sys.stderr)
        return 2
