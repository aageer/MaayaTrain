# ⚡ MaayaTrain

**Cross-platform distributed ML training using the DiLoCo algorithm.**

Train machine learning models across multiple devices — Mac, Windows, Linux — over regular Wi-Fi. Uses the DiLoCo algorithm to reduce network traffic by **500×** compared to traditional distributed training, making collaborative training practical on consumer hardware.

> Created by **Akhil Ageer**.

---

## Table of Contents

1. [What's New in v0.2](#whats-new-in-v02)
2. [Installation](#installation)
3. [Quick Start (Single Device)](#quick-start-single-device)
4. [Multi-Device Training (Step-by-Step)](#multi-device-training-step-by-step)
5. [Same-Machine Multi-Process Training](#same-machine-multi-process-training)
6. [SOTA Features](#sota-features)
7. [Supported Platforms](#supported-platforms)
8. [CLI Reference](#cli-reference)
9. [Configuration](#configuration)
10. [How DiLoCo Works](#how-diloco-works)
11. [Project Architecture](#project-architecture)
12. [Troubleshooting](#troubleshooting)

---

## What's New in v0.2

| Feature | What it Does | Research Basis |
|---------|-------------|----------------|
| 🔥 **Mixed Precision (AMP)** | 2× faster training, 50% less memory | PyTorch AMP, bfloat16 auto-detect |
| 📈 **Cosine Warmup LR** | Better convergence, avoids early divergence | GPT-3, Chinchilla training recipes |
| 🔄 **Gradient Accumulation** | Train with larger effective batch sizes on small GPUs | Standard DDP technique |
| 📡 **Streaming Sync** | 100× less peak bandwidth during sync | Streaming DiLoCo (DeepMind, 2025) |
| 🗜️ **INT8 Compression** | 8–12× gradient compression (vs 4-6× for FP16) | Per-tensor affine quantization |
| 🛡️ **Byzantine Tolerance** | Tolerates up to 1/3 faulty/malicious workers | Coordinate-wise median aggregation |
| 📝 **BPE Tokenizer** | Production-quality text tokenization | Sennrich et al. (2016) BPE algorithm |
| 📊 **GPU Memory Tracking** | Monitor VRAM usage during training | PyTorch peak memory stats |

---

## Installation

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **pip** (check with `pip --version` or `pip3 --version`)
- **Git** (optional, for cloning)

### Option A: Install from source (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MaayaTrain.git
cd MaayaTrain

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# 4. Install MaayaTrain + all dependencies
pip install .

# 5. Verify it works
maayatrain status
```

### Option B: Install in editable mode (for development)

```bash
pip install -e ".[dev]"
```

### What gets installed automatically

MaayaTrain installs everything it needs — no manual dependency management:

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch ML framework (auto-detects your GPU) |
| `safetensors` | Checkpoint format |
| `typer` + `rich` | Beautiful CLI interface |
| `pydantic` | Configuration validation |
| `zeroconf` | Automatic device discovery on LAN |
| `fastapi` + `uvicorn` | Dashboard server |
| `httpx` | HTTP client for relay discovery |
| `numpy` | Tensor serialization |

---

## Quick Start (Single Device)

Train a GPT-2 model on a single device to verify everything works:

```bash
# 1. Initialize config (creates maayatrain.toml)
maayatrain init

# 2. Create a sample dataset
mkdir -p data
curl -o data/wikitext.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 3. Start training (with live dashboard)
maayatrain start --model gpt2-small --dataset ./data/wikitext.txt --dashboard
```

The dashboard opens at **http://localhost:8471** — watch loss decrease in real-time.

Press `Ctrl+C` to stop. A checkpoint is saved automatically.

---

## Multi-Device Training (Step-by-Step)

This is the core feature of MaayaTrain. You can train across **any combination** of devices: Mac + Windows, Linux + Mac, 2 Linux boxes with NVIDIA GPUs, etc.

### What you need

- 2+ devices on the **same Wi-Fi/LAN network** (or use the relay for internet)
- MaayaTrain installed on each device
- The **same dataset file** on each device
- The **same model name** on each device (e.g., `gpt2-small`)

### Step 1: Prepare the dataset on ALL devices

Every device needs a local copy of the training data. The file must be a plain text file.

```bash
# On EVERY device:
mkdir -p data
curl -o data/wikitext.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

> **Important:** Each device trains on its own **random batches** from this file. They don't need different data — DiLoCo handles the diversity through independent training paths.

### Step 2: Start the Coordinator (Device 1)

Pick **one device** to be the coordinator. This device manages synchronization.

```bash
# On Device 1 (the coordinator):
maayatrain start \
    --model gpt2-small \
    --dataset ./data/wikitext.txt \
    --dashboard
```

You'll see output like:

```
⚡ MaayaTrain v0.2.0
Device: Apple M4 | 16.0 GB | ~4.6 TFLOPS | MPS | Darwin
Model: gpt2-small (28.5M parameters)
Mixed precision: float16 (AMP enabled)
LR schedule: linear warmup (50 steps) → cosine decay (min 10.0%)
Dashboard: http://localhost:8471
Listening on port 7471 — waiting for workers…
```

**Note the IP address** of this device (find it via `ipconfig` on Windows or `ifconfig`/`ip addr` on Mac/Linux).

### Step 3: Join from Workers (Device 2, 3, …)

On each additional device, join the training session:

#### Option A: Auto-discovery (devices on same LAN)

```bash
# On Device 2, 3, etc. — discovers coordinator automatically via mDNS:
maayatrain join auto --dataset ./data/wikitext.txt
```

#### Option B: Manual address (if mDNS doesn't work)

```bash
# Replace with your coordinator's actual IP address:
maayatrain join 192.168.1.100:7471 --dataset ./data/wikitext.txt
```

You'll see:

```
⚡ MaayaTrain v0.2.0
Device: NVIDIA RTX 4090 | 24.0 GB | ~82.6 TFLOPS | CUDA | Linux
Mixed precision: bfloat16 (AMP enabled)
Found coordinator: gpt2-small at 192.168.1.100:7471
Connecting…
```

### Step 4: Watch training progress

Open **http://[coordinator-ip]:8471** in any browser to see:
- 📈 Live loss curve
- 👥 Connected peers (all devices)
- 💾 Checkpoint timeline
- ⚡ Tokens/second throughput
- 💾 GPU memory usage

### Step 5: Stop training

Press `Ctrl+C` on **any device** — it saves a local checkpoint and gracefully disconnects. The remaining devices continue training.

Press `Ctrl+C` on the **coordinator** to stop the entire session. A final checkpoint is saved.

### Step 6: Resume training later

```bash
# Resume from the last checkpoint:
maayatrain start \
    --model gpt2-small \
    --dataset ./data/wikitext.txt \
    --resume ./checkpoints/step-005000 \
    --dashboard
```

---

## Same-Machine Multi-Process Training

You can test multi-device training on **a single machine** by running two terminals:

```bash
# Terminal 1 — Coordinator
maayatrain start --model gpt2-small --dataset ./data/wikitext.txt --port 7471 --dashboard

# Terminal 2 — Worker (different port is handled automatically)
maayatrain join localhost:7471 --dataset ./data/wikitext.txt
```

Both processes train independently and sync every 500 steps.

---

## SOTA Features

### Mixed Precision (AMP)

MaayaTrain auto-detects the best precision for your hardware:

| Hardware | Auto-Detected Precision | Speedup |
|----------|------------------------|---------|
| NVIDIA Ampere+ (A100, RTX 3090/4090) | **bfloat16** | ~2× |
| NVIDIA Volta/Turing (V100, T4) | **float16** + GradScaler | ~1.8× |
| Apple Silicon (M1–M5) | **float16** | ~1.5× |
| CPU | Off (FP32) | — |

Override with `mixed_precision = "bf16"`, `"fp16"`, or `"off"` in config.

### Cosine Warmup Learning Rate

Every training run uses the gold-standard LLM schedule:
1. **Linear warmup** (first 10% of steps): gradually increases LR to prevent early instability
2. **Cosine decay** (remaining 90%): smoothly decreases to `min_lr_ratio × peak_lr`

```
LR ▲
   │    ╭──╮
   │   ╱    ╲
   │  ╱      ╲
   │ ╱        ╲
   │╱          ╲___
   ╰──────────────── Steps →
   warmup    cosine decay
```

### Gradient Accumulation

Simulate large batch sizes without extra GPU memory:

```toml
[training]
batch_size = 4                      # Micro-batch per step
gradient_accumulation_steps = 8     # Effective batch = 4 × 8 = 32
```

### Streaming Sync (Streaming DiLoCo)

Instead of syncing all 124M parameters at once (bandwidth spike), split into K shards and sync them sequentially during training:

```toml
[diloco]
streaming_shards = 4   # Sync 1/4 of params at a time
```

Reduces **peak bandwidth** by K× while maintaining convergence.

### INT8 Gradient Compression

Enable aggressive compression for very slow networks (the algorithm
uses per-tensor min-max affine quantization):

```toml
[diloco]
compress_int8 = true   # 8-12× compression (vs 4-6× with FP16)
```

| Mode | Compression | Quality Impact |
|------|------------|----------------|
| FP16 + gzip | ~4–6× | Negligible |
| INT8 + gzip | ~8–12× | Very small (~1% error) |

### Byzantine Fault Tolerance

Protect against faulty or malicious workers with coordinate-wise median aggregation:

```toml
[diloco]
aggregation = "median"   # Tolerates up to 1/3 bad workers
```

### BPE Tokenizer

Use production-quality byte-pair encoding instead of character-level tokenization:

```python
from maayatrain.training.tokenizer import BPETokenizer

tok = BPETokenizer(vocab_size=4096)
tok.train(open("data/wikitext.txt").read())
tok.save("tokenizer.json")

ids = tok.encode("hello world", add_bos=True)
text = tok.decode(ids)
```

---

## Supported Platforms

| Platform | GPU Backend | Install Command | Notes |
|----------|------------|-----------------|-------|
| **macOS** (Apple Silicon) | MPS (Metal) | `pip install .` | M1/M2/M3/M4/M5 all supported |
| **Linux** (x86_64) | NVIDIA CUDA | `pip install .` | Requires CUDA 11.8+ drivers |
| **Linux** (x86_64) | AMD ROCm | `pip install .` | Requires ROCm 5.4+ |
| **Windows 10/11** | NVIDIA CUDA | `pip install .` | Requires CUDA 11.8+ drivers |
| **Any OS** | CPU fallback | `pip install .` | Works but slow — useful for testing |
| **Linux** (x86_64) | Intel XPU (Arc) | `pip install .` | Experimental |

> **PyTorch auto-detects your GPU.** You don't need to install anything extra for GPU support — `pip install .` handles it.

### OS-Specific Notes

**Windows:**
```powershell
# Use PowerShell (not CMD) for best experience
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install .
maayatrain status   # Verify GPU is detected
```

**Linux (NVIDIA):**
```bash
# Verify CUDA is available before installing
nvidia-smi   # Should show your GPU
python3 -m venv .venv && source .venv/bin/activate
pip install .
maayatrain status   # Should show "CUDA" backend
```

**macOS (Apple Silicon):**
```bash
# MPS backend is automatic on M-series Macs
python3 -m venv .venv && source .venv/bin/activate
pip install .
maayatrain status   # Should show "MPS" backend
```

---

## CLI Reference

| Command | What it does |
|---------|-------------|
| `maayatrain init` | Create a default `maayatrain.toml` config file |
| `maayatrain status` | Show hardware info, detected GPU, available models |
| `maayatrain start` | Start training as the **coordinator** |
| `maayatrain join <target>` | Join a session as a **worker** (`auto` or `host:port`) |
| `maayatrain version` | Show version |
| `maayatrain relay export` | Export checkpoint for async handoff |
| `maayatrain relay import` | Import a relay checkpoint |

### `maayatrain start` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | `gpt2-small` | Model architecture name |
| `--dataset`, `-d` | `./data/wikitext.txt` | Path to training text file |
| `--dashboard` | `false` | Start the web monitoring dashboard |
| `--port`, `-p` | `7471` | TCP port for peer connections |
| `--resume` | — | Resume from a checkpoint directory |
| `--max-steps` | `100000` | Maximum training steps |
| `--verbose`, `-v` | `false` | Enable debug logging |

---

## Configuration

Run `maayatrain init` to create `maayatrain.toml`:

```toml
[model]
name = "gpt2-small"                # Choose: gpt2-small, gpt2-medium, gpt2-large

[dataset]
path = "./data/wikitext.txt"        # Path to training text file
seq_length = 512                    # Tokens per training sequence

[training]
batch_size = 8                      # Per-device micro-batch size
max_steps = 100000                  # Total training steps
checkpoint_dir = "./checkpoints"
checkpoint_every = 1000
log_every = 10
seed = 42
mixed_precision = "auto"            # "auto" | "fp16" | "bf16" | "off"
gradient_accumulation_steps = 1     # Effective batch = batch_size × this
warmup_steps = 0                    # 0 = auto (10% of inner_steps)
min_lr_ratio = 0.1                  # Min LR as fraction of peak (cosine)
max_grad_norm = 1.0                 # Gradient clipping

[diloco]
inner_steps = 500                   # Local steps before sync (H)
inner_lr = 0.0003                   # Inner AdamW learning rate
outer_lr = 0.7                      # Outer Nesterov SGD learning rate
outer_momentum = 0.9                # Momentum coefficient
nesterov = true                     # Use Nesterov momentum
gradient_compression = true         # Compress pseudo-gradients
compress_fp16 = true                # FP16 compression (default)
compress_int8 = false               # INT8 compression (more aggressive)
streaming_shards = 1                # 1 = off, 4+ = streaming sync
aggregation = "mean"                # "mean" or "median" (Byzantine-tolerant)

[network]
port = 7471
heartbeat_interval = 5

[dashboard]
port = 8471
enabled = false
```

---

## How DiLoCo Works

Traditional distributed training (DDP) synchronizes gradients after **every** step — requiring high-bandwidth interconnects:

```
Traditional DDP:  sync every step    → needs ~50 GB/s  → requires InfiniBand
MaayaTrain:       sync every 500     → needs ~0.1 GB/s → works over Wi-Fi ✅
```

The DiLoCo algorithm splits training into two loops:

**Inner loop** (runs on each device independently):
```
θ_local = θ_global          # Start from shared weights
for step in range(500):     # Train locally for 500 steps
    loss = model(batch)
    θ_local -= α · AdamW(∇loss)
```

**Outer loop** (coordinator averages all devices):
```
Δθ_avg = mean(θ_global − θ_local_1, ..., θ_global − θ_local_n)
v = β·v + Δθ_avg            # Momentum
θ_global -= η·(Δθ_avg + β·v) # Nesterov update
```

**Result:** Each device explores a different region of the loss landscape, then they combine their findings. It converges similarly to standard distributed training while using 500× less bandwidth.

---

## Project Architecture

```
MaayaTrain/
├── maayatrain/
│   ├── __init__.py             # Package version + author
│   ├── app.py                  # Typer CLI (init, start, join, status)
│   ├── settings.py             # Pydantic v2 config + TOML loader
│   ├── hardware.py             # Cross-platform GPU detection
│   ├── comms/
│   │   ├── wire_format.py      # Binary protocol (header + payload)
│   │   ├── tensor_codec.py     # FP16/INT8 + gzip compression
│   │   └── tcp_channel.py      # Async TCP server/client
│   ├── training/
│   │   ├── diloco.py           # DiLoCo algorithm + streaming sync
│   │   ├── loop.py             # Training loop (AMP, grad accum, cosine LR)
│   │   ├── lr_schedule.py      # Cosine warmup scheduler
│   │   ├── tokenizer.py        # BPE tokenizer
│   │   ├── orchestrator.py     # Coordinator node logic
│   │   ├── participant.py      # Worker node logic
│   │   ├── snapshots.py        # SafeTensors checkpointing
│   │   └── cluster_info.py     # Cluster state tracking
│   ├── architectures/
│   │   ├── gpt2.py             # GPT-2 model (124M/355M/774M)
│   │   └── catalog.py          # Model registry
│   ├── discovery/
│   │   ├── zeroconf_service.py # mDNS advertiser + browser
│   │   ├── relay_client.py     # HTTP relay for WAN discovery
│   │   └── roster.py           # Thread-safe peer registry
│   └── monitor/
│       ├── server.py           # FastAPI + WebSocket dashboard
│       └── static/index.html   # Dashboard UI
├── tests/                      # 51 unit tests
├── pyproject.toml              # Package config
├── maayatrain.toml             # Default training config
├── LICENSE                     # MIT
└── README.md
```

---

## Troubleshooting

### `maayatrain: command not found`
Your virtual environment isn't activated. Run:
```bash
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\Activate.ps1     # Windows PowerShell
```

### `ModuleNotFoundError: No module named 'maayatrain'`
You haven't installed the package. Run `pip install .` inside the project directory.

### Worker can't find coordinator (mDNS)
Some corporate Wi-Fi blocks multicast. Use manual addressing:
```bash
maayatrain join 192.168.1.100:7471 --dataset ./data/wikitext.txt
```

### `Dataset not found`
Every device needs its **own local copy** of the dataset. The dataset is not shared over the network.

### GPU not detected
Run `maayatrain status` to check. If it shows "CPU", verify:
- **NVIDIA:** `nvidia-smi` should show your GPU. Install CUDA drivers if not.
- **Apple:** M-series Macs should show "MPS" automatically.
- **AMD:** Ensure ROCm is installed (`rocminfo`).

### `CUDA out of memory`
Reduce `batch_size` in your config, or use gradient accumulation:
```toml
[training]
batch_size = 2
gradient_accumulation_steps = 16   # Effective batch = 32
mixed_precision = "auto"           # Halves memory usage
```

### Training is slow on CPU
CPU training is ~100× slower than GPU. It works for testing but isn't practical for real training. Use a GPU-equipped device as the coordinator.

### Port already in use
Change the port: `maayatrain start --port 7472`

---

## License

MIT License — see [LICENSE](LICENSE).

Built with ❤️ by **Akhil Ageer**.
