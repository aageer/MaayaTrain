# MaayaTrain — Two-Laptop Training Guide

> **What this is:** Step-by-step instructions to train an AI model across two MacBooks over Wi-Fi.  
> **Your laptop:** MacBook Pro M4 (the "Coordinator" — the boss)  
> **Friend's laptop:** MacBook Air (the "Worker" — the helper)  
> **What you need:** Both laptops on the **same Wi-Fi network**. That's it.

---

## How It Works (30-Second Explanation)

```
┌──────────────────────┐         Wi-Fi          ┌──────────────────────┐
│   YOUR MacBook Pro   │ ◄──────────────────────►│  FRIEND'S MacBook Air│
│   (Coordinator)      │    Training data        │  (Worker)            │
│                      │    flows back & forth    │                      │
│   • Runs the show    │    every 500 steps       │   • Trains locally   │
│   • Collects results │    (~60 seconds)         │   • Sends results    │
│   • Saves checkpoints│                          │   • Gets updates     │
└──────────────────────┘                          └──────────────────────┘
```

Each laptop trains the AI independently on its own copy of the data. Every ~60 seconds, they sync up — your laptop collects the work, averages it together, and sends back the improved model. This is the **DiLoCo** algorithm. The magic is that they only talk every 500 steps instead of every single step, so slow Wi-Fi is totally fine.

---

## PART 1: Set Up YOUR Laptop (Coordinator) — 15 minutes

You already have MaayaTrain installed. Let's make sure everything is ready.

### Step 1.1 — Open Terminal
- Press `Cmd + Space`, type **Terminal**, press Enter

### Step 1.2 — Go to the Project Folder
```bash
cd ~/Kean/Projects/MaayaTrain
```

### Step 1.3 — Make Sure the Virtual Environment Works
```bash
source .venv/bin/activate
```
You should see `(.venv)` appear at the start of your terminal line. That means Python is ready.

### Step 1.4 — Prepare Training Data
You need a text file for the AI to learn from. Let's create a simple one:
```bash
mkdir -p data
```

**Option A — Use sample data (easiest, works immediately):**
```bash
python3 -c "
import random
random.seed(42)
words = ['the', 'model', 'training', 'distributed', 'compute', 'gradient',
         'network', 'worker', 'parameter', 'optimization', 'learning', 'data',
         'batch', 'loss', 'convergence', 'synchronization', 'quantization',
         'compression', 'tensor', 'weights', 'momentum', 'architecture',
         'transformer', 'attention', 'embedding', 'language', 'generation',
         'inference', 'latency', 'throughput', 'bandwidth', 'protocol',
         'deep', 'neural', 'function', 'layer', 'output', 'input', 'hidden']
lines = []
for _ in range(10000):
    line = ' '.join(random.choices(words, k=random.randint(10, 40)))
    lines.append(line)
with open('data/training_text.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'Created {len(lines)} lines of training text.')
"
```

**Option B — Use real text (better results, but optional):**
```bash
# Download a small chunk of Wikipedia (~2MB)
curl -L "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
     -o data/training_text.txt
```

### Step 1.5 — Find Your IP Address
Your friend's laptop needs to know where to connect. Run:
```bash
ipconfig getifaddr en0
```
This will print something like `192.168.1.42`. **Write this number down** — you'll give it to your friend.

> **Note:** If `en0` shows nothing, try `en1`:
> ```bash
> ipconfig getifaddr en1
> ```

### Step 1.6 — Start the Coordinator
```bash
python -m maayatrain start \
    --model gpt2-small \
    --dataset ./data/training_text.txt \
    --dashboard \
    --max-steps 5000
```

You'll see output like:
```
╭─────────────────────────────────────────╮
│ ⚡ MaayaTrain v0.1.0                    │
│ Cross-platform distributed ML training  │
╰─────────────────────────────────────────╯
Device: Apple M4 Pro (MPS) — 16.0 GB
Model: gpt2-small (124.4M parameters)
Dashboard: http://localhost:8471
Listening on port 7471 — waiting for workers…
```

**🎉 Your laptop is now the Coordinator. It's waiting for your friend to connect.**

**Leave this terminal window open. Don't close it. Don't press anything.**

### Step 1.7 — Open the Dashboard (Optional but Cool)
- Open Safari or Chrome
- Go to: `http://localhost:8471`
- You'll see a live dashboard showing training progress
- Keep this open to watch the magic happen

---

## PART 2: Set Up FRIEND'S Laptop (Worker) — 20 minutes

Your friend's MacBook Air needs MaayaTrain installed and the same training data.

### Step 2.1 — Install Python (if not already installed)
On your friend's MacBook Air, open Terminal and check:
```bash
python3 --version
```
If it shows `Python 3.11` or higher, you're good. If not:
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Python
brew install python@3.13
```

### Step 2.2 — Copy the Project to Friend's Laptop
You have three options. Pick the easiest one:

**Option A — AirDrop (easiest):**
1. On YOUR Mac, right-click the `MaayaTrain` folder
2. Click **Share** → **AirDrop**
3. Send it to your friend's MacBook Air
4. On friend's Mac, move it to a convenient location:
```bash
mv ~/Downloads/MaayaTrain ~/MaayaTrain
```

**Option B — USB Drive:**
1. Copy the `MaayaTrain` folder to a USB stick
2. Plug it into friend's MacBook Air
3. Copy it:
```bash
cp -r /Volumes/USB_DRIVE/MaayaTrain ~/MaayaTrain
```

**Option C — GitHub (if your repo is pushed):**
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/MaayaTrain.git
```

### Step 2.3 — Set Up the Virtual Environment
On your **friend's** laptop:
```bash
cd ~/MaayaTrain

# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -e .
```

> **If `pip install -e .` fails**, try:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -e .
> ```

### Step 2.4 — Copy the Same Training Data
The worker needs the **exact same** training data file. You can either:

**Option A — AirDrop the data file:**
- Send `data/training_text.txt` from your Mac to friend's Mac via AirDrop
- Put it in the same place:
```bash
mkdir -p ~/MaayaTrain/data
mv ~/Downloads/training_text.txt ~/MaayaTrain/data/
```

**Option B — Generate the same data:**
If you used the sample data script in Step 1.4 (Option A), just run the **exact same script** on your friend's laptop:
```bash
cd ~/MaayaTrain
mkdir -p data
python3 -c "
import random
random.seed(42)
words = ['the', 'model', 'training', 'distributed', 'compute', 'gradient',
         'network', 'worker', 'parameter', 'optimization', 'learning', 'data',
         'batch', 'loss', 'convergence', 'synchronization', 'quantization',
         'compression', 'tensor', 'weights', 'momentum', 'architecture',
         'transformer', 'attention', 'embedding', 'language', 'generation',
         'inference', 'latency', 'throughput', 'bandwidth', 'protocol',
         'deep', 'neural', 'function', 'layer', 'output', 'input', 'hidden']
lines = []
for _ in range(10000):
    line = ' '.join(random.choices(words, k=random.randint(10, 40)))
    lines.append(line)
with open('data/training_text.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'Created {len(lines)} lines of training text.')
"
```
> **Why same seed?** `random.seed(42)` makes both scripts produce identical text, so both laptops train on the same data.

### Step 2.5 — Make Sure Both Laptops Are on the Same Wi-Fi
- On YOUR laptop: Click the Wi-Fi icon → note the network name (e.g., "HomeWiFi")
- On FRIEND's laptop: Make sure they're connected to the **exact same** network

### Step 2.6 — Connect the Worker to Your Coordinator

**Option A — Automatic discovery (easiest, usually works):**
```bash
cd ~/MaayaTrain
source .venv/bin/activate
python -m maayatrain join auto --dataset ./data/training_text.txt
```
The worker will search for your coordinator via mDNS and connect automatically.

**Option B — Manual connection (if auto doesn't work):**
Remember the IP address from Step 1.5? Use it here:
```bash
cd ~/MaayaTrain
source .venv/bin/activate
python -m maayatrain join 192.168.1.42:7471 --dataset ./data/training_text.txt
```
> ⚠️ Replace `192.168.1.42` with YOUR actual IP address from Step 1.5!

You should see:
```
╭─────────────────────────────────────────╮
│ ⚡ MaayaTrain v0.1.0                    │
│ Cross-platform distributed ML training  │
╰─────────────────────────────────────────╯
Device: Apple M3 (MPS) — 8.0 GB
Connecting to 192.168.1.42:7471…

Received model weights from coordinator (124.4M parameters)
Starting inner training loop — 500 steps per round…
```

**🎉 Both laptops are now training the same AI model together!**

---

## PART 3: Watch It Run (The Fun Part)

### What You'll See on YOUR Laptop (Coordinator)
```
[Round 1] Received pseudo-gradient from MacBookAir (local_steps=312)
[Round 1] Local training: 500 steps, loss=8.234 → 6.102
[Round 1] Aggregating 2 workers (weighted by compute)
[Round 1] Outer step complete. Broadcasting updated weights.
─────────────────────────────────────────
[Round 2] Received pseudo-gradient from MacBookAir (local_steps=308)
[Round 2] Local training: 500 steps, loss=6.102 → 5.411
...
```

### What You'll See on FRIEND'S Laptop (Worker)
```
[Round 1] Inner training: 312 steps complete (loss=8.501 → 6.340)
[Round 1] Sending pseudo-gradient to coordinator…
[Round 1] Received updated weights from coordinator.
─────────────────────────────────────────
[Round 2] Inner training: 308 steps complete (loss=6.340 → 5.520)
...
```

### The Dashboard (on YOUR Laptop)
- Open `http://localhost:8471` in a browser
- You'll see:
  - 📉 A **loss curve** going down over time (the AI is learning!)
  - 🖥️ **Connected peers** showing both laptops
  - ⚡ **Tokens/sec** for each device
  - 🌐 **Network RTT** between the two laptops

### Understanding the Numbers
- **Loss going down** = good! The AI is learning.
- **Your M4 Pro will do ~500 steps** per round (it's faster)
- **Friend's Air will do ~200-350 steps** per round (it's slower, that's OK!)
- **MaayaTrain automatically handles this** — the faster laptop's work counts more (compute-proportional weighting). No work is wasted.

---

## PART 4: Stop Training & Save Results

### When to Stop
- Let it run for **at least 15 minutes** (about 10-15 rounds)
- For paper-quality results, run for **1-2 hours** if possible
- Watch the loss number — it should steadily decrease

### How to Stop
1. **On your friend's laptop:** Press `Ctrl+C` in the terminal
2. **On your laptop:** Press `Ctrl+C` in the terminal
3. The coordinator automatically saves a checkpoint before shutting down

### Where the Results Are Saved
On YOUR laptop (the coordinator):
```
~/Kean/Projects/MaayaTrain/
├── checkpoints/
│   ├── step-500/           ← Saved model at step 500
│   ├── step-1000/          ← Saved model at step 1000
│   ├── ...
│   └── telemetry.csv       ← All metrics for your paper graphs!
```

### Check Telemetry Data
```bash
cat checkpoints/telemetry.csv
```
You'll see:
```
global_step,loss,compute_hours,cluster_rtt_ms,active_peers,streaming_shards
500,6.102,0.02,12.5,1,4
1000,5.411,0.04,15.2,1,4
1500,4.890,0.06,11.8,1,4
...
```
**This is the real data you'll use for your paper graphs!**

---

## Troubleshooting

### "Connection refused" when worker tries to join
**Cause:** Firewall is blocking the connection.
**Fix on YOUR laptop:**
```bash
# On macOS, allow Python through the firewall
# System Settings → Network → Firewall → Options → Add Python
```
Or temporarily disable the firewall:
- System Settings → Network → Firewall → Turn Off
- (Remember to turn it back on after training!)

### "No coordinator found on LAN" with `join auto`
**Cause:** mDNS/Bonjour is blocked on the Wi-Fi network (common in universities/coffee shops).
**Fix:** Use the manual IP address method instead:
```bash
python -m maayatrain join 192.168.1.42:7471 --dataset ./data/training_text.txt
```

### Worker connects but then disconnects
**Cause:** The training data file is missing or different.
**Fix:** Make sure both laptops have the **exact same** `training_text.txt` file. The file must have the same content and be in `data/training_text.txt`.

### "ModuleNotFoundError: No module named 'maayatrain'"
**Cause:** The virtual environment isn't activated, or the package isn't installed.
**Fix:**
```bash
cd ~/MaayaTrain          # or ~/Kean/Projects/MaayaTrain on your laptop
source .venv/bin/activate
pip install -e .
```

### "CUDA not available" / "MPS not available"
**Cause:** This is normal on MacBook Air without dedicated GPU.
**Fix:** MaayaTrain automatically falls back to CPU. It'll be slower but works perfectly. The whole point of MaayaTrain is to handle this!

### Very slow training on the MacBook Air
**Expected behavior!** Your M4 Pro is ~3-5× faster than the Air. MaayaTrain handles this automatically through compute-proportional weighting — the Air does 200 steps while your Pro does 500 steps, and MaayaTrain weights them accordingly. No compute is wasted.

### Wi-Fi drops during training
**MaayaTrain handles this!** If the connection drops:
- The coordinator will wait for the worker to reconnect
- Training continues on your laptop in the meantime
- When the worker reconnects, it gets the latest weights and resumes

---

## The "Coffee Shop Sabotage" Experiment (For Your Paper!)

This makes an AMAZING figure for your research paper:

1. Start training normally between both laptops
2. Let it run for 5 minutes (about 5 rounds) — everything is smooth
3. **On your laptop, start downloading something huge** (like a 4K movie trailer from YouTube)
4. Watch the dashboard — you'll see:
   - 📈 RTT (latency) spike from ~15ms to ~200ms+
   - 🔄 Dynamic Sharding kick in (shards jump from 4 → 8 → 16)
   - ✅ Training continues without crashing!
5. Stop the download — RTT drops, shards decrease back
6. The `telemetry.csv` captures all of this automatically

This real-world chaos test proves your Dynamic Sharding algorithm works. Plot this data in your paper → instant reviewer approval.

---

## Quick Command Reference

| Action | Command (Your Laptop) |
|--------|----------------------|
| Start coordinator | `python -m maayatrain start --model gpt2-small --dataset ./data/training_text.txt --dashboard` |
| Check your IP | `ipconfig getifaddr en0` |
| View dashboard | Open `http://localhost:8471` in browser |
| Check telemetry | `cat checkpoints/telemetry.csv` |

| Action | Command (Friend's Laptop) |
|--------|--------------------------|
| Auto-connect worker | `python -m maayatrain join auto --dataset ./data/training_text.txt` |
| Manual-connect worker | `python -m maayatrain join YOUR_IP:7471 --dataset ./data/training_text.txt` |

---

*You're about to train an AI across two laptops over Wi-Fi. That's literally what billion-dollar companies do, just with fancier hardware. Let's go! 🚀*
