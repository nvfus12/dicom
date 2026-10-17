# gpubench: benchmarking distributed training on lab laptop GPUs

Benchmark code that compares single GPU, DDP and local SGD by **time to reach a target accuracy**.

## What has been verified

| Item | Status |
|---|---|
| 3 simulated CPU workers over gloo (TCP on localhost), all 4 modes: single / ddp / sync / localsgd | Runs end to end, PyTorch 2.6.0 |
| Parameter-synchronisation tests (`tests/`) | 8/8 pass, PyTorch 2.6.0 |
| Machines with mismatched configs → all stop together with a clear error, no hang | Tried |
| Real GPUs, real CIFAR-10, real network between 3 machines | **Not yet** (no access to the machines) |
| DDP with CUDA tensors over gloo on Windows | **Not yet**. If it fails, use `--set mode=sync` |

**Version warning:** PyTorch **2.10** on Windows could not create a gloo device (`makeDeviceForHostname(): unsupported gloo device`, for every address including 127.0.0.1). **2.6.0** works. Versions 2.7 to 2.9 were not tried. The `check` command detects this problem.

## Architecture

```
gpubench/
  domain/      core; knows nothing about YAML, network, NVML or torchvision
    config.py    MachineProfile (differs per machine), ExperimentSpec (identical on all machines)
    ports.py     interfaces: Device, Communicator, DatasetProvider, MetricsSink, GpuMonitor
    sync.py      synchronisation strategies: single, ddp, sync, localsgd
    trainer.py   training loop, timing, stop when the target is reached
  adapters/    implementations of the interfaces
    device.py    CudaDevice, CpuDevice
    comm.py      GlooComm (torch.distributed), LocalComm (single worker)
    data.py      CIFAR-10, synthetic data
    models.py    resnet20, resnet18 (CIFAR variants), tiny_cnn
    metrics.py   JSONL
    monitor.py   NVML (optional)
  app/         composition: read YAML, pick adapters, run
  cli.py       python -m gpubench run|plan|check|profile|machines
configs/
  machines/    ONE FILE PER MACHINE
  experiments/ one file per experiment
```

**A machine is a configuration file, not a class.** Everything that differs between machines (IP, which GPU, AMP on/off, maximum batch size, data directory, network adapter, environment variables) is data in `configs/machines/<name>.yaml`. The code only has adapters for each *kind of infrastructure* (CUDA/CPU, gloo/single worker, CIFAR/synthetic). Adding a 4th or 10th machine means adding one YAML file and listing its name under `workers` in an experiment; no code changes.

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
5. Fill in `address` (IPv4 on the network used for the run) in the machine file. On the rank 0 machine, open the inbound TCP port `master_port` (default 29500) in Windows Defender Firewall, or switch the network profile to Private.
6. Download CIFAR-10 once beforehand so it is not counted in run time (running E1 on each machine downloads it automatically).

When done, **copy the same `configs/` folder to every machine**: each machine must read the others' files to find rank 0.

### Running experiments

```bash
python -m gpubench plan -e e3_localsgd_x3 --set local_steps=16
```

`plan` prints ranks, addresses, the exact command to run on each machine and any missing fields. Then run exactly those commands, one per machine, in any order; machines wait for each other for up to `timeout_s` seconds.

| Experiment | Command (on each machine, replace `<machine>`) |
|---|---|
| E1 baseline | `python -m gpubench run -m <machine> -e e1_single` (add `--set model=resnet18`) |
| E2 DDP, 2 identical GPUs | `python -m gpubench run -m <machine> -e e2_ddp_t1000x2` |
| E2 DDP, 3 machines | `python -m gpubench run -m <machine> -e e2_ddp_x3` |
| E3 local SGD | `python -m gpubench run -m <machine> -e e3_localsgd_x3 --set local_steps=4` (then 16, 64) |

`--set` must be identical on all machines. If it differs, every machine stops together and reports a fingerprint mismatch.

After running E1, set the same `target_acc` in every experiment file.

### Analysis

Collect the `results/` folder from the rank 0 machine of each run into one place, then:

```bash
python analyze/analyze.py
```

Output: `results/summary.csv` and `results/accuracy_vs_time.png`.

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
- Local SGD: optimizer momentum stays local to each machine; only parameters and BatchNorm statistics are averaged. The learning rate is not scaled by the number of workers.
- DDP/sync: the learning rate is scaled linearly with the number of workers (`lr_scaling: linear`), with warmup.
