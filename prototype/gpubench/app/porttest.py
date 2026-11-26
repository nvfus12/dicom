"""`gpubench porttest`: can the other machine open a TCP connection to this machine's Python.

gloo opens connections between the workers on random ports, so every worker must accept
inbound TCP from the others. This uses the same Python executable as training (the one in
.venv), so a firewall rule that blocks that executable shows up here too.

    on machine A:  python -m gpubench porttest listen
    on machine B:  python -m gpubench porttest probe <IPv4 of A>

`porttest pair` checks both directions at once (used by the lab agent): rank 0 listens on
--port and rank 1 on --port + 1, while each probes the other.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

from ..adapters.comm import local_ipv4_toward

DEFAULT_PORT = 45000
GREETING = b"gpubench-porttest OK\n"


def _own_ipv4(port: int) -> str:
    try:
        return local_ipv4_toward("192.168.0.1", port)
    except OSError:
        return "?"


def _listen_core(port: int, seconds: float, host: str = "0.0.0.0", stop_after_first: bool = False,
                 announce: bool = True) -> tuple[bool, str, list[str]]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.bind((host, port))
        except OSError as e:
            return False, f"LỖI: không mở được cổng {port} trên máy này ({e}).", []
        server.listen(8)
        server.settimeout(0.5)
        if announce:
            own = _own_ipv4(port)
            print(f"Đang nghe cổng {port}. IPv4 của máy này: {own}", flush=True)
            print(f"Trên máy kia chạy:  python -m gpubench porttest probe {own} --port {port}", flush=True)
            print(f"Tự dừng sau {seconds:g} giây (Ctrl+C để dừng sớm).\n", flush=True)
        peers: list[str] = []
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                try:
                    conn, peer = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.sendall(GREETING)
                peers.append(peer[0])
                print(f"Nhận kết nối từ {peer[0]}", flush=True)
                if stop_after_first:
                    break
        except KeyboardInterrupt:
            pass
    if peers:
        return True, (f"ĐẠT: đã nhận {len(peers)} kết nối (từ {', '.join(sorted(set(peers)))}). "
                      f"Máy kia mở được kết nối TCP tới Python trên máy này."), peers
    return False, (f"LỖI: không nhận được kết nối nào trên cổng {port} trong {seconds:g} giây. Nếu "
                   f"máy kia đã thử gọi tới đúng IP và cổng này, kết nối bị chặn trước khi tới đây "
                   f"(tường lửa máy này, hoặc router)."), peers


def _probe_core(host: str, port: int, timeout: float) -> tuple[bool, str, str]:
    """(ok, message, kind) with kind in ok | refused | blocked | other."""
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            reply = conn.recv(64)
    except ConnectionRefusedError:
        return False, (f"LỖI: {host}:{port} từ chối kết nối. Đường mạng THÔNG nhưng bên kia chưa "
                       f"nghe trên cổng này (chưa chạy listen, hoặc sai cổng/IP)."), "refused"
    except socket.timeout:
        return False, (f"LỖI: {host}:{port} không phản hồi sau {timeout:g} giây: bị chặn "
                       f"(tường lửa của máy {host}, hoặc router chặn các máy nói chuyện với nhau)."), \
            "blocked"
    except OSError as e:
        return False, f"LỖI: không kết nối được tới {host}:{port}: {e}", "other"
    ms = (time.perf_counter() - start) * 1000
    if reply.startswith(GREETING.strip()):
        return True, f"ĐẠT: kết nối được tới Python trên {host}:{port} sau {ms:.0f} ms.", "ok"
    return False, (f"LỖI: {host}:{port} trả lời lạ ({reply!r}); có chương trình khác đang dùng "
                   f"cổng này."), "other"


def listen(port: int, seconds: int, host: str = "0.0.0.0") -> int:
    ok, message, _ = _listen_core(port, seconds, host)
    print("\n" + message, flush=True)
    return 0 if ok else 1


def probe(host: str, port: int, timeout: float) -> int:
    ok, message, _ = _probe_core(host, port, timeout)
    print(message, flush=True)
    return 0 if ok else 1


def _probe_until(host: str, port: int, seconds: float, timeout: float = 4) -> tuple[bool, str]:
    """Keeps trying while the other side may not be listening yet; reports the last attempt."""
    deadline = time.monotonic() + seconds
    while True:
        ok, message, kind = _probe_core(host, port, timeout)
        if ok or time.monotonic() >= deadline:
            return ok, message
        time.sleep(0.5 if kind == "refused" else 0.1)


def pair(machine_name: str, rank: int, peer: str, port: int, window: float,
         json_out: Path | None = None) -> int:
    """Both directions at once: this machine listens on its own port while probing the other's."""
    mine, theirs = (port, port + 1) if rank == 0 else (port + 1, port)
    print(f"Máy {machine_name} (rank {rank}): nghe cổng {mine}, gọi {peer}:{theirs}, "
          f"tối đa {window:g} giây.", flush=True)
    heard: dict = {}

    def run_listener():
        ok, message, peers = _listen_core(mine, window, stop_after_first=True, announce=False)
        heard.update(ok=ok, message=message, peers=peers)
    listener = threading.Thread(target=run_listener, daemon=True)
    listener.start()
    probe_ok, probe_message = _probe_until(peer, theirs, max(1.0, window - 1))
    print(probe_message, flush=True)
    listener.join(window + 5)
    print(heard.get("message", "LỖI: bên nghe không kết thúc"), flush=True)
    result = {"machine": machine_name, "rank": rank, "peer": peer,
              "listen": {"port": mine, "ok": heard.get("ok", False), "from": heard.get("peers", []),
                         "message": heard.get("message", "")},
              "probe": {"port": theirs, "ok": probe_ok, "message": probe_message}}
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if probe_ok and heard.get("ok") else 1
