# gpubench: benchmarking distributed training on lab laptop GPUs

Benchmark code that compares single GPU, DDP and local SGD by **time to reach a target accuracy**.

## What has been verified

| Item | Status |
|---|---|
| 3 simulated CPU workers over gloo (TCP on localhost), all 4 modes: single / ddp / sync / localsgd | Runs end to end, PyTorch 2.6.0 |
| Synchronisation tests (`tests/`): models stay identical; `avg` equals the exact mean of the workers; DiLoCo outer step equals `torch.optim.SGD(nesterov=True)`; fp16/bf16 transfer | 23/23 pass, PyTorch 2.6.0 |
| DDP with the bf16 compression hook | **Not tested** (fp16 hook is tested) |
| `GLOO_SOCKET_IFNAME` (`network_interface`) on Windows | Works with a real adapter name; a wrong name fails with `Unable to find address for: <name>` |
| Machines with mismatched configs → all stop together with a clear error, no hang | Tried |
| `netcheck` between 3 simulated workers on localhost | Runs; parsing of `tailscale ping` output is unit-tested, but the real `tailscale` CLI was never run |
| Running over a VPN/internet | **Not yet** |
| Real GPUs, real CIFAR-10, real network between 3 machines | **Not yet** (no access to the machines) |
| DDP with CUDA tensors over gloo on Windows | **Not yet**. If it fails, use `--set mode=sync` |

**Version warning:** PyTorch **2.10** on Windows could not create a gloo device (`makeDeviceForHostname(): unsupported gloo device`, for every address including 127.0.0.1). **2.6.0** works. Versions 2.7 to 2.9 were not tried. The `check` command detects this problem.

## Architecture

```
gpubench/
  domain/      core; knows nothing about YAML, network, NVML or torchvision
    config.py    MachineProfile (differs per machine), ExperimentSpec (identical on all machines)
    ports.py     interfaces: Device, Communicator, DatasetProvider, MetricsSink, GpuMonitor
    sync.py      synchronisation strategies: single, ddp, sync, localsgd (avg or DiLoCo outer step)
    trainer.py   training loop, timing, stop when the target is reached
  adapters/    implementations of the interfaces
    device.py    CudaDevice, CpuDevice
    comm.py      GlooComm (torch.distributed), LocalComm (single worker)
    data.py      CIFAR-10, synthetic data
    models.py    resnet20, resnet18 (CIFAR variants), tiny_cnn
    metrics.py   JSONL
    monitor.py   NVML (optional)
  app/         composition: read YAML, pick adapters, run
  cli.py       python -m gpubench run|plan|check|profile|netcheck|machines
configs/
  machines/    ONE FILE PER MACHINE
  experiments/ one file per experiment
```

**A machine is a configuration file, not a class.** Everything that differs between machines (IP, which GPU, maximum batch size, data directory, network adapter, environment variables) is data in `configs/machines/<name>.yaml`. The code only has adapters for each *kind of infrastructure* (CUDA/CPU, gloo/single worker, CIFAR/synthetic). Adding a 4th or 10th machine means adding one YAML file and listing its name under `workers` in an experiment; no code changes.

A new adapter is only needed for a new *kind* of infrastructure, for example AMD GPUs (ROCm), Linux with NCCL, or another dataset.

Files for real machines are git-ignored because they hold owners' names and IPs. Copy `configs/machines/machine.yaml.example` and fill it in. The `sim-*` profiles used for CPU simulation are included.

## Workflow with real machines

Do this on **each worker machine** (copy the whole `prototype/` folder over):

1. Install Python 3.10+ and create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```
2. Install PyTorch 2.6.0 built for CUDA 12.4. Check the "CUDA Version" line of `nvidia-smi` first; the driver must support CUDA 12.4:
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. Check the environment. You should see `cuda available True` and `gloo device: OK`:
   ```bash
   python -m gpubench check -m <machine>
   ```
4. Measure the largest batch that fits, then put it into `max_batch` in the machine file:
   ```bash
   python -m gpubench profile -m <machine>
   ```
5. Fill in `address` (IPv4 on the network used for the run) in the machine file. **Every** machine must accept inbound TCP from the other workers: gloo opens connections between every pair, on random ports, not only to rank 0's `master_port`. Allow inbound TCP from the VPN range only (see below), or switch the network profile to Private on a trusted network.
   Alternatively pass `--master-addr <rank 0 IP>` to `run`/`netcheck`; then each machine only needs its own machine file.
6. Download CIFAR-10 once beforehand so it is not counted in run time (running E1 on each machine downloads it automatically).

When done, **copy the same `configs/` folder to every machine**: each machine must read the others' files to find rank 0.

### Running over the internet (machines on different networks)

Training uses gloo over direct TCP connections, and **every machine must reach every other machine** (full mesh); rank 0 is only the first meeting point. Laptops on home or campus networks sit behind NAT and cannot do this on their own. Port forwarding is usually impossible (CGNAT, campus firewalls), and the gloo port must **never be exposed to the public internet**: gloo has no encryption or authentication.

Use a mesh VPN such as Tailscale, ZeroTier or WireGuard. Each machine gets a virtual IP and the code runs unchanged:

1. Every owner installs the VPN client and joins the same virtual network (each owner does this themselves).
2. In each machine file set `address` to the **VPN IP**, and `network_interface` to the VPN adapter name (for Tailscale on Windows this is normally `Tailscale`; check the adapter list on the machine). Without it, gloo may advertise a different adapter's address to the peers. This has not been tested on real machines.
3. In Windows Defender Firewall allow inbound TCP **only from the VPN range** on every worker (Tailscale uses 100.64.0.0/10), for example in an administrator PowerShell: `New-NetFirewallRule -DisplayName gpubench-vpn -Direction Inbound -Protocol TCP -RemoteAddress 100.64.0.0/10 -Action Allow`, and remove it afterwards with `Remove-NetFirewallRule -DisplayName gpubench-vpn`.
4. Measure the link with the same transport training will use:
   ```bash
   python -m gpubench netcheck -m <machine> -e e3_localsgd_x3
   ```
   Run it on every worker like a normal experiment. It prints round-trip latency and throughput per machine pair, the time of one all-reduce at the size of ResNet-20 (~1 MB) and ResNet-18 (~45 MB), and, if the `tailscale` CLI is installed, whether each peer is reached directly or through a relay. A relayed path is much slower and distorts the benchmark, so record it with every result. Output is also written to `results/netcheck-*/`.

**Expect poor numbers for per-step synchronisation.** Residential upload bandwidth is often a few tens of Mbit/s. A 45 MB all-reduce may then take tens of seconds per step, so `ddp`/`sync` with ResNet-18 will very likely be slower than one GPU; this is a hypothesis to confirm with `netcheck`, not a measurement. `localsgd` pays the cost only every H steps, so sweep large H (up to 512). A slower-than-one-GPU result is still a valid result if it is measured and explained.

No controller is needed for this: all workers stay online for the whole run, and rank 0 only hosts the rendezvous. A controller becomes necessary for workers that join or leave mid-run, or if a VPN cannot be used (then machines would have to call out to a relay server).

### Running experiments

```bash
python -m gpubench plan -e e3_localsgd_x3 --set local_steps=16
```

`plan` prints ranks, addresses, the exact command to run on each machine and any missing fields. Then run exactly those commands, one per machine, in any order; machines wait for each other for up to `timeout_s` seconds.

| Experiment | Command (on each machine, replace `<machine>`) |
|---|---|
| Minimal benchmark, run 1: one GPU | `python -m gpubench run -m <machine> -e run1_single` |
| Minimal benchmark, run 2: DDP on 3 machines | `python -m gpubench run -m <machine> -e run2_ddp_x3 --master-addr <rank 0 IP>` |
| Minimal benchmark, run 3: local SGD H=256 | `python -m gpubench run -m <machine> -e run3_localsgd_x3 --master-addr <rank 0 IP>` |
| E1 baseline | `python -m gpubench run -m <machine> -e e1_single` (add `--set model=resnet18`) |
| E2 DDP, 2 identical GPUs | `python -m gpubench run -m <machine> -e e2_ddp_t1000x2` |
| E2 DDP, 3 machines | `python -m gpubench run -m <machine> -e e2_ddp_x3` |
| E3 local SGD, plain averaging | `python -m gpubench run -m <machine> -e e3_localsgd_x3 --set local_steps=4` (then 16, 64, 256, 512) |
| E3 local SGD, DiLoCo | add `--set outer_optimizer=nesterov --set outer_lr=0.7 --set outer_momentum=0.9` (sweep momentum 0, 0.5, 0.9) |
| Half-precision transfer | add `--set comm_dtype=fp16` (or `bf16`) to any distributed experiment |
| Network check | `python -m gpubench netcheck -m <machine> -e e3_localsgd_x3` |

`--set` must be identical on all machines. If it differs, every machine stops together and reports a fingerprint mismatch.

After running E1, set the same `target_acc` in every experiment file.

### Analysis

Collect the `results/` folder of **every** machine into one place (each run is a subfolder; the folders of one distributed run hold one file per rank and merge cleanly), then:

```bash
python analyze/analyze.py --acc 0.6 0.7 0.8 --at-minutes 10 30
```

It reads every rank's log and the `netcheck-*` folders, and writes to `results/report/`:

| File | Content |
|---|---|
| `runs.csv`, `table_runs.tex` | One row per run: time to each accuracy level, speed-up against the baseline (default: the fastest single-GPU run, or `--baseline <part of run id>`), accuracy at given minutes, group throughput, scaling efficiency, estimated synchronisation overhead per step, sync rounds, data sent |
| `ranks.csv`, `table_ranks.tex` | One row per machine and run: per-step data / compute / communication time, VRAM peak, max GPU temperature, min SM clock |
| `network.csv`, `table_network.tex` | Netcheck: RTT and throughput per pair, direct or relayed, all-reduce time per size |
| `accuracy_vs_time.png`, `accuracy_vs_epoch.png` | Wall-clock view and statistical-efficiency view |
| `time_breakdown.png`, `gpu_thermal.png` | Per-step time split per machine; temperature and clock over time (needs NVML) |

How the comparisons are computed:

- **Scaling efficiency** = group throughput / sum of the single-GPU throughputs of the same machines. Valid only if those single runs ran on otherwise idle machines; in the CPU simulation they share one CPU, so values above 1 can appear.
- **Synchronisation overhead per step** = step time of the distributed run − step time of the slowest machine alone (same batch). An indirect estimate, needed because DDP does not report communication separately.
- **DDP data sent** is estimated as steps × parameters × 4 bytes; `sync` and `localsgd` report it directly.
- Single-GPU runs started at the same moment on several machines no longer collide: the default run id contains rank 0's machine name.

## Trying it without a GPU

```bash
python scripts/simulate.py -e smoke_dist --set mode=sync
```

```bash
python -m unittest discover -s tests -v
```

On Windows this needs a PyTorch version other than 2.10 (see the version warning).

## Reading the numbers correctly

- `train_time_s` excludes evaluation on the test set and start-up time.
- `comm_s` includes time **waiting for the slowest machine**, because an all-reduce only completes when every machine arrives.
- Mode `ddp` overlaps communication with the backward pass, so `comm_s` is 0 and communication time is inside `compute_s`. To separate communication, use `sync` (same operation, no overlap).
- `comm_bytes` is the amount of data fed into all-reduce on one machine, not the number of bytes that actually crossed the network.
- Local SGD keeps an fp32 copy of the global model. At each synchronisation every machine sends its pseudo-gradient (global minus local weights); the averaged pseudo-gradient drives the outer step:
  - `outer_optimizer: avg` (default): the new global model is exactly the mean of the workers' models (FedAvg). This is the baseline.
  - `outer_optimizer: nesterov`: SGD with Nesterov momentum on the pseudo-gradient, as in DiLoCo (Douillard et al., 2023). With `outer_lr: 1` and `outer_momentum: 0` it equals `avg`. DiLoCo used lr 0.7 and momentum 0.9 for language models; for ResNet-20 on CIFAR, Lin et al. (ICLR 2020) report that high global momentum can hurt accuracy, so sweep it.
  - BatchNorm statistics are always plainly averaged; inner optimizer momentum stays local. The inner learning rate is not scaled by the number of workers.
  - `sync_rounds` in `train_end` counts outer synchronisations.
- `amp` (mixed precision) is an **experiment** setting, identical on all workers: with `ddp`/`sync` a worker using AMP sends loss-scaled gradients, so mixing AMP and non-AMP workers would silently corrupt the average. Machine files no longer have an `amp` key.
- Local SGD evaluates the **global model** (the state after the last outer step), as DiLoCo does; evaluation never forces an extra synchronisation, so H keeps its meaning even when an epoch is shorter than H. Before the first synchronisation the evaluated model is the initial one.
- Time to reach any accuracy level can be computed after the run: `python analyze/analyze.py --acc 0.6 0.7 0.8`.
- `comm_dtype` (`fp32` / `fp16` / `bf16`) only changes what crosses the network; all model arithmetic stays fp32. In `localsgd` only the pseudo-gradient is rounded, never the weights. In `sync`, gradients are sent after AMP loss scaling, so fp16 may overflow and make the GradScaler skip steps; prefer `bf16` there when AMP is on. In `ddp` it uses PyTorch's DDP compression hooks.
- DDP/sync: the learning rate is scaled linearly with the number of workers (`lr_scaling: linear`), with warmup.
