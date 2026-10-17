# dicom: training AI models on idle lab GPUs

A student research project on using the idle GPUs of laptops in a university lab to train one model together. A controller machine coordinates; worker machines contribute their GPU when it is free.

**Status: design stage plus a tested benchmark harness.** No results have been measured on real lab GPUs or over a real network yet. Everything the harness was run on so far was CPU simulation (3 local processes talking over gloo).

## Research question

With a few heterogeneous Windows laptops (GPUs with roughly 4 to 6 GB of VRAM) on a shared, probably wireless network, does distributed training reduce the **time to reach a target accuracy** compared with one GPU, and under which conditions (model size, synchronisation frequency, speed gap between GPUs, network)?

The first benchmark round assumes stable conditions: no machine is reclaimed mid-run. GPU reclaiming, checkpoint/restart and elastic membership are future work.

## Repository layout

| Path | Content |
|---|---|
| `prototype/` | The benchmark harness `gpubench` (Python/PyTorch). See [prototype/README.md](prototype/README.md). |

## The prototype in short

- Training modes: `single` (one GPU), `ddp` (PyTorch DistributedDataParallel), `sync` (manual gradient all-reduce, separates communication time) and `localsgd` (average parameters every H steps).
- Backend: gloo over TCP, the only multi-node option on Windows. Workloads: CIFAR-10 with ResNet-20 or ResNet-18.
- Clean architecture: a `domain` core that only sees interfaces, `adapters` for CUDA/CPU, gloo, datasets and logging, and an `app` layer that wires them together.
- **Every machine is one YAML file** in `prototype/configs/machines/`. Adding a machine means adding a file and listing its name in an experiment; no code changes.
- Per-run JSONL logs record compute, communication and wait time, bytes, accuracy and GPU telemetry. `analyze/analyze.py` produces a summary table and an accuracy-versus-time plot.

### Quick start (CPU simulation, no GPU needed)

```bash
cd prototype
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/simulate.py -e smoke_dist --set mode=sync
```

### Known issues

- PyTorch 2.10 on Windows could not create a gloo device (`makeDeviceForHostname(): unsupported gloo device`) in our test. PyTorch 2.6.0 works. Versions 2.7 to 2.9 were not tried. `python -m gpubench check -m <machine>` detects the problem.
- DDP with CUDA tensors over gloo on Windows has not been tested on real GPUs. If it fails, use `--set mode=sync`.

### Machine files are not in this repository

Files for real machines contain the owners' names and network addresses and are git-ignored. To add one, copy `prototype/configs/machines/machine.yaml.example` to `prototype/configs/machines/<name>.yaml` and fill it in. The `sim-*` profiles used for CPU simulation are included.

## Language

Project documents (most of the prototype documentation) are written in Vietnamese for the supervising lecturer. This README is the English entry point.
