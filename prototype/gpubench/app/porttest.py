"""`gpubench porttest`: can the other machine open a TCP connection to this machine's Python.

gloo opens connections between the workers on random ports, so every worker must accept
inbound TCP from the others. This uses the same Python executable as training (the one in
.venv), so a firewall rule that blocks that executable shows up here too.

    on machine A:  python -m gpubench porttest listen
    on machine B:  python -m gpubench porttest probe <IPv4 of A>
"""
from __future__ import annotations

import socket
import time

from ..adapters.comm import local_ipv4_toward

DEFAULT_PORT = 45000
GREETING = b"gpubench-porttest OK\n"


def listen(port: int, seconds: int, host: str = "0.0.0.0") -> int:
    try:
        own = local_ipv4_toward("192.168.0.1", port)
    except OSError:
        own = "?"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.bind((host, port))
        except OSError as e:
            print(f"LỖI: không mở được cổng {port} trên máy này ({e}). Thử --port khác.")
            return 1
        server.listen(8)
        server.settimeout(1.0)
        print(f"Đang nghe cổng {port}. IPv4 của máy này: {own}", flush=True)
        print(f"Trên máy kia chạy:  python -m gpubench porttest probe {own} --port {port}", flush=True)
        print(f"Tự dừng sau {seconds} giây (Ctrl+C để dừng sớm).\n", flush=True)
        received = 0
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                try:
                    conn, peer = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.sendall(GREETING)
                received += 1
                print(f"Nhận kết nối từ {peer[0]}", flush=True)
        except KeyboardInterrupt:
            pass
    if received:
        print(f"\nĐẠT: đã nhận {received} kết nối. Máy kia mở được kết nối TCP tới Python trên máy này.")
        return 0
    print("\nKhông nhận được kết nối nào. Nếu máy kia đã chạy probe tới đúng IP và cổng này, "
          "kết nối bị chặn trước khi tới đây (tường lửa máy này, hoặc router).")
    return 1


def probe(host: str, port: int, timeout: float) -> int:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            reply = conn.recv(64)
    except ConnectionRefusedError:
        print(f"LỖI: {host}:{port} từ chối kết nối. Đường mạng THÔNG nhưng bên kia chưa chạy "
              f"`porttest listen` (hoặc sai cổng). Chạy listen trên máy kia rồi thử lại.")
        return 1
    except socket.timeout:
        print(f"LỖI: {host}:{port} không phản hồi sau {timeout:g} giây: bị chặn "
              f"(tường lửa của máy {host}, hoặc router chặn các máy nói chuyện với nhau).")
        return 1
    except OSError as e:
        print(f"LỖI: không kết nối được tới {host}:{port}: {e}")
        return 1
    ms = (time.perf_counter() - start) * 1000
    if reply.startswith(GREETING.strip()):
        print(f"ĐẠT: kết nối được tới Python trên {host}:{port} sau {ms:.0f} ms.")
        return 0
    print(f"LỖI: {host}:{port} trả lời lạ ({reply!r}); có chương trình khác đang dùng cổng này.")
    return 1
