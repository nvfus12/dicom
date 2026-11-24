"""Pinning the gloo device to an IPv4 address, and the `glootest` diagnostic.

On the real lab laptops, gloo chose its address from the Wi-Fi adapter's name and both
machines crashed while connecting. These tests run on localhost only.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed.distributed_c10d as c10d  # noqa: E402

from gpubench.adapters import comm  # noqa: E402
from gpubench.app import bootstrap  # noqa: E402

PROTOTYPE = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def free_port_pair() -> int:
    """A port p such that p and p+1 are both free: glootest uses one port per way it tries."""
    for _ in range(50):
        port = free_port()
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
    raise RuntimeError("không tìm được hai cổng liền nhau còn trống")


class LocalAddressTest(unittest.TestCase):
    def test_dia_chi_di_toi_loopback_la_loopback(self):
        self.assertEqual(comm.local_ipv4_toward("127.0.0.1", 29500), "127.0.0.1")

    def test_dia_chi_di_toi_mot_ip_la_ipv4(self):
        address = comm.local_ipv4_toward("192.168.0.1", 29500)
        self.assertEqual(len(address.split(".")), 4)
        self.assertNotIn(":", address)


class BindPolicyTest(unittest.TestCase):
    def test_mac_dinh_gan_theo_ipv4(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GPUBENCH_GLOO_DEVICE", None)
            self.assertEqual(bootstrap.gloo_bind_address("127.0.0.1", 29500), "127.0.0.1")

    def test_bien_moi_truong_quay_ve_cach_cu(self):
        with mock.patch.dict(os.environ, {"GPUBENCH_GLOO_DEVICE": "interface"}):
            self.assertIsNone(bootstrap.gloo_bind_address("127.0.0.1", 29500))


class PinnedDeviceTest(unittest.TestCase):
    def test_tao_thiet_bi_theo_ip_va_tra_lai_lop_goc(self):
        original = c10d.ProcessGroupGloo
        made = []
        real_create = dist.ProcessGroupGloo.create_device

        def spy(**kwargs):
            made.append(kwargs)
            return real_create(**kwargs)
        with mock.patch.object(dist.ProcessGroupGloo, "create_device", side_effect=spy):
            g = comm.GlooComm("127.0.0.1", free_port(), 0, 1, 30, bind_addr="127.0.0.1")
        try:
            self.assertEqual(made, [{"hostname": "127.0.0.1"}])
            self.assertIs(c10d.ProcessGroupGloo, original)
            t = torch.ones(3)
            g.all_reduce_mean_(t)
            self.assertEqual(t.tolist(), [1.0, 1.0, 1.0])
        finally:
            g.close()


class GlooTestCommandTest(unittest.TestCase):
    def test_hai_may_chay_glootest(self):
        port = free_port_pair()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        procs = [subprocess.Popen([sys.executable, "-m", "gpubench", "glootest", "-m", name,
                                   "--master-addr", "127.0.0.1", "--port-base", str(port),
                                   "-e", "smoke_dist", "--set", "workers=[sim-0,sim-1]",
                                   "--wait", "60"],
                                  cwd=PROTOTYPE, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                 for name in ("sim-0", "sim-1")]
        outputs = [p.communicate(timeout=300)[0] for p in procs]
        for out in outputs:
            self.assertIn("theo-ip", out)
            self.assertRegex(out, r"theo-ip\s+ĐẠT")
            self.assertIn("theo-ten-card", out)      # sims have no adapter name: reported, not run
            self.assertIn("IPv4 dùng để đi tới rank 0: 127.0.0.1", out)
        self.assertEqual([p.returncode for p in procs], [0, 0])


if __name__ == "__main__":
    unittest.main()
