# MaayaTrain — Gemma Fine-Tuning Guide (Two MacBooks)

> **Author:** Akhil Ageer · Kean University  
> **Hardware:** MacBook Pro M4 16GB (Primary) + MacBook Air/Pro (Secondary)  
> **Goal:** Fine-tune Google Gemma 3 1B on a HuggingFace dataset across two MacBooks

---

## Prerequisites

| Item | Primary MacBook (M4) | Secondary MacBook |
|------|---------------------|-------------------|
| macOS | 14+ (Sonoma/Sequoia) | 14+ |
| Python | 3.11+ | 3.11+ |
| RAM | 16 GB | 8+ GB |
| Disk | ~15 GB free | ~15 GB free |
| Wi-Fi | Same network | Same network |

---

## PHASE 1 — Environment Setup (Both Machines)

### Step 1.1 — Install Dependencies (Run on BOTH MacBooks)

```bash
# Go to project directory (adjust path on secondary machine)
cd ~/Kean/Projects/MaayaTrain   # Primary
# cd ~/MaayaTrain               # Secondary (if copied via AirDrop)

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install MaayaTrain
pip install -e .

# Install HuggingFace + LoRA libraries
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf
```

### Step 1.2 — Authenticate with HuggingFace (Both Machines)

Gemma requires accepting Google's license on HuggingFace.

1. Go to https://huggingface.co/google/gemma-3-1b-pt
2. Click **"Agree and access repository"**
3. Go to https://huggingface.co/settings/tokens → Create a token with `read` access

```bash
pip install huggingface-hub
huggingface-cli login
# Paste your token when prompted
```

### Step 1.3 — Verify GPU/MPS Backend

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device: {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')
"
```

Expected output: `MPS available: True`

---

## PHASE 2 — Download Model & Dataset

### Step 2.1 — Download Gemma 3 1B

```bash
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = 'google/gemma-3-1b-pt'
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
print('Downloading model (~2.5 GB)...')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('Done! Model cached in ~/.cache/huggingface/')
"
```

> **Note:** First download takes 5-10 min depending on internet speed. Subsequent runs use the cache.

### Step 2.2 — Download a HuggingFace Dataset

We recommend **`databricks/databricks-dolly-15k`** — 15K human-written instruction-response pairs, commercially licensed.

```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('databricks/databricks-dolly-15k', split='train')
print(f'Dataset size: {len(ds)} samples')
print(f'Columns: {ds.column_names}')
print(f'Sample:\n{ds[0]}')
"
```

**Alternative datasets (pick ONE):**

| Dataset | Size | Best For |
|---------|------|----------|
| `databricks/databricks-dolly-15k` | 15K | General instruction-following |
| `yahma/alpaca-cleaned` | 52K | Broad instruction tuning |
| `HuggingFaceH4/no_robots` | 10K | High-quality human-annotated |
| `GAIR/lima` | 1K | Minimal but high-quality |

---

## PHASE 3 — Prepare Training Script

### Step 3.1 — Create the Fine-Tuning Script

Save this as `finetune_gemma.py` in your MaayaTrain directory:

```bash
cat > finetune_gemma.py << 'SCRIPT'
"""
Gemma 3 1B Fine-Tuning with LoRA on Apple Silicon
Author: Akhil Ageer — Kean University
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─── Configuration ───────────────────────────────────────────
MODEL_ID = "google/gemma-3-1b-pt"
DATASET_ID = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./gemma-finetuned"
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4        # effective batch = 2 * 4 = 8
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ─── Device Setup ────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ─── Load Tokenizer & Model ─────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,  # we handle placement manually for MPS
)

# ─── Apply LoRA ──────────────────────────────────────────────
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─── Load & Format Dataset ───────────────────────────────────
print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")

def format_prompt(example):
    """Convert Dolly format → instruction prompt."""
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    if context:
        text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": text}

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# Split: 90% train, 10% eval
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ─── Training Arguments ─────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,  # MPS doesn't support fp16 training well
    bf16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    seed=42,
    dataloader_pin_memory=False,  # Required for MPS
    use_mps_device=(device == "mps"),
)

# ─── Train ───────────────────────────────────────────────────
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

trainer.train()

# ─── Save ────────────────────────────────────────────────────
print("Saving model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model saved to {OUTPUT_DIR}/final")
print("Training complete!")
SCRIPT

echo "Created finetune_gemma.py"
```

---

## PHASE 4 — Run Training

### Option A: Single MacBook (Simpler)

```bash
cd ~/Kean/Projects/MaayaTrain
source .venv/bin/activate
python3 finetune_gemma.py
```

**Expected timeline on M4 16GB:**
- Gemma 1B + LoRA + Dolly 15K: **~2-4 hours** for 3 epochs
- You'll see loss printed every 10 steps

### Option B: Two MacBooks with MaayaTrain (Distributed)

**On Primary MacBook (Coordinator):**
```bash
cd ~/Kean/Projects/MaayaTrain
source .venv/bin/activate
python -m maayatrain start \
    --model gpt2-small \
    --dataset ./data/training_text.txt \
    --dashboard \
    --max-steps 5000
```

**On Secondary MacBook (Worker):**
```bash
cd ~/MaayaTrain
source .venv/bin/activate
python -m maayatrain join auto --dataset ./data/training_text.txt
```

> **Note:** MaayaTrain currently supports its built-in GPT-2 architecture. For Gemma via HuggingFace, use Option A on each MacBook independently, or run the `finetune_gemma.py` script on both machines with different data splits.

---

## PHASE 5 — Monitor Training

### Step 5.1 — Watch the Logs

Key metrics to monitor during training:

| Metric | Healthy Range | Red Flag |
|--------|--------------|----------|
| `loss` | Decreasing steadily | Stuck or increasing |
| `eval_loss` | Decreasing, close to train loss | Much higher than train loss (overfitting) |
| `learning_rate` | Follows cosine schedule | N/A |
| `grad_norm` | < 5.0 | Spikes above 10 (instability) |

### Step 5.2 — Check Memory Usage

Open another terminal while training runs:

```bash
# Watch memory usage (macOS)
while true; do
    echo "$(date): $(memory_pressure | head -1)"
    sleep 30
done
```

Or use Activity Monitor → Memory tab → look for Python process.

### Step 5.3 — If Training Crashes (OOM)

Reduce memory usage by editing `finetune_gemma.py`:

```python
BATCH_SIZE = 1          # reduce from 2
MAX_SEQ_LENGTH = 256    # reduce from 512
GRAD_ACCUM = 8          # increase to compensate
```

---

## PHASE 6 — Post-Training Tests

> **This is the most important phase.** Run ALL of these to confirm your fine-tuned model works.

### Test 1: Sanity Check — Model Loads

```bash
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-pt', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, './gemma-finetuned/final')
tokenizer = AutoTokenizer.from_pretrained('./gemma-finetuned/final')
print('✅ Model loaded successfully')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

### Test 2: Generation Quality — Before vs After

```bash
cat > test_generation.py << 'SCRIPT'
"""Compare base model vs fine-tuned model outputs."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load base model
print("=" * 60)
print("BASE MODEL (before fine-tuning)")
print("=" * 60)
base_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-pt", torch_dtype=torch.float16
).to(device)

# Load fine-tuned model
print("\n" + "=" * 60)
print("FINE-TUNED MODEL (after training)")
print("=" * 60)
ft_model = PeftModel.from_pretrained(base_model, "./gemma-finetuned/final").to(device)
ft_tokenizer = AutoTokenizer.from_pretrained("./gemma-finetuned/final")

# Test prompts
test_prompts = [
    "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
    "### Instruction:\nWhat are the benefits of distributed computing?\n\n### Response:\n",
]

for prompt in test_prompts:
    print(f"\nPROMPT: {prompt.strip()[:80]}...")

    # Base model
    inputs = base_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(f"  BASE:      {base_tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):][:150]}")

    # Fine-tuned model
    inputs = ft_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = ft_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(f"  FINETUNED: {ft_tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):][:150]}")

print("\n✅ Generation comparison complete")
SCRIPT
python3 test_generation.py
```

**What to look for:**
- Fine-tuned model should follow the instruction format better
- Responses should be more structured and relevant
- Base model may ramble or not follow the instruction pattern

### Test 3: Loss Curve Analysis

```bash
cat > test_loss_curve.py << 'SCRIPT'
"""Plot training loss from trainer logs."""
import json, os

log_dir = "./gemma-finetuned"
log_files = []
for root, dirs, files in os.walk(log_dir):
    for f in files:
        if f == "trainer_state.json":
            log_files.append(os.path.join(root, f))

if not log_files:
    print("❌ No trainer_state.json found. Check OUTPUT_DIR.")
    exit(1)

with open(log_files[0]) as f:
    state = json.load(f)

train_losses = [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]
eval_losses = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]

print("📊 Training Loss Curve:")
print(f"{'Step':>6} | {'Train Loss':>10} | {'Status'}")
print("-" * 40)
for step, loss in train_losses[-15:]:
    status = "✅" if loss < train_losses[0][1] else "⚠️"
    print(f"{step:>6} | {loss:>10.4f} | {status}")

if eval_losses:
    print(f"\n📊 Eval Loss:")
    for step, loss in eval_losses[-5:]:
        print(f"  Step {step}: {loss:.4f}")

    # Check for overfitting
    if len(eval_losses) >= 2:
        if eval_losses[-1][1] > eval_losses[-2][1]:
            print("⚠️  WARNING: Eval loss increasing — possible overfitting!")
        else:
            print("✅ Eval loss still decreasing — training looks healthy")

print(f"\n📉 Loss reduction: {train_losses[0][1]:.4f} → {train_losses[-1][1]:.4f}")
print(f"   Improvement: {((train_losses[0][1] - train_losses[-1][1]) / train_losses[0][1] * 100):.1f}%")
SCRIPT
python3 test_loss_curve.py
```

**Healthy results:**
- Loss should decrease by 30-60% over training
- Eval loss should track close to train loss
- No sudden spikes

### Test 4: Perplexity Evaluation

```bash
cat > test_perplexity.py << 'SCRIPT'
"""Measure perplexity on held-out evaluation set."""
import torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./gemma-finetuned/final")
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "./gemma-finetuned/final").to(device)
model.eval()

# Load eval data
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
eval_data = dataset.select(range(100))  # 100 samples for quick eval

total_loss = 0
total_tokens = 0

for sample in eval_data:
    text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print(f"📊 Evaluation Results:")
print(f"   Average Loss: {avg_loss:.4f}")
print(f"   Perplexity:   {perplexity:.2f}")
print()
if perplexity < 20:
    print("✅ Excellent — model has learned the task well")
elif perplexity < 50:
    print("✅ Good — model shows meaningful learning")
elif perplexity < 100:
    print("⚠️  Fair — consider more training or data quality check")
else:
    print("❌ Poor — check training setup, data formatting, or hyperparameters")
SCRIPT
python3 test_perplexity.py
```

### Test 5: Consistency & Repetition Check

```bash
cat > test_consistency.py << 'SCRIPT'
"""Check for repetition, degeneration, and consistency."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("./gemma-finetuned/final")
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "./gemma-finetuned/final").to(device)
model.eval()

prompt = "### Instruction:\nExplain gravity.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

issues = []

# Test 1: Check for repetition
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)
response = text[len(prompt):]

# Count repeated phrases (3-grams)
words = response.split()
trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)

if unique_ratio < 0.5:
    issues.append(f"❌ High repetition (unique trigrams: {unique_ratio:.0%})")
else:
    print(f"✅ Repetition check passed (unique trigrams: {unique_ratio:.0%})")

# Test 2: Multiple generations should be coherent
print("\n📝 Sample generations (temperature=0.7):")
for i in range(3):
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    print(f"  [{i+1}] {gen[:120]}...")

# Test 3: Empty/garbage check
if len(response.strip()) < 10:
    issues.append("❌ Model produces near-empty responses")
else:
    print(f"✅ Response length OK ({len(response.split())} words)")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ All consistency checks passed!")
SCRIPT
python3 test_consistency.py
```

### Test 6: LoRA Adapter Integrity

```bash
python3 -c "
import os, json

adapter_dir = './gemma-finetuned/final'
required_files = ['adapter_config.json', 'adapter_model.safetensors']
missing = [f for f in required_files if not os.path.exists(os.path.join(adapter_dir, f))]

if missing:
    print(f'❌ Missing files: {missing}')
else:
    print('✅ All adapter files present')
    with open(os.path.join(adapter_dir, 'adapter_config.json')) as f:
        cfg = json.load(f)
    print(f'   LoRA rank: {cfg.get(\"r\", \"?\")}')
    print(f'   Alpha: {cfg.get(\"lora_alpha\", \"?\")}')
    print(f'   Target modules: {cfg.get(\"target_modules\", \"?\")}')
    size_mb = os.path.getsize(os.path.join(adapter_dir, 'adapter_model.safetensors')) / 1e6
    print(f'   Adapter size: {size_mb:.1f} MB (should be ~10-50 MB)')
    if size_mb < 1:
        print('   ⚠️  Adapter suspiciously small — may not have trained')
    elif size_mb > 500:
        print('   ⚠️  Adapter too large — check if full model was saved instead')
    else:
        print('   ✅ Adapter size looks correct')
"
```

---

## PHASE 7 — Post-Training Checklist

Run through this checklist after training completes:

| # | Test | Command | Pass Criteria |
|---|------|---------|---------------|
| 1 | Model loads | `python3 -c "from peft import ..."` | No errors |
| 2 | Loss decreased | `python3 test_loss_curve.py` | 30%+ reduction |
| 3 | Generation quality | `python3 test_generation.py` | Follows instruction format |
| 4 | Perplexity | `python3 test_perplexity.py` | < 50 |
| 5 | No repetition | `python3 test_consistency.py` | Unique trigrams > 50% |
| 6 | Adapter files | Check adapter_config.json | All files present, 10-50 MB |
| 7 | Eval loss stable | Check trainer_state.json | Not increasing at end |

---

## PHASE 8 — Merge & Export (Optional)

If you want a standalone model (no separate adapter):

```bash
cat > merge_model.py << 'SCRIPT'
"""Merge LoRA adapter into base model for standalone deployment."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.float16)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, "./gemma-finetuned/final")

print("Merging weights...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained("./gemma-merged")
AutoTokenizer.from_pretrained("./gemma-finetuned/final").save_pretrained("./gemma-merged")

print("✅ Merged model saved to ./gemma-merged/")
print(f"   Size: {sum(os.path.getsize(os.path.join('./gemma-merged', f)) for f in os.listdir('./gemma-merged')) / 1e9:.2f} GB")
SCRIPT
python3 merge_model.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OutOfMemoryError` | Reduce `BATCH_SIZE` to 1, `MAX_SEQ_LENGTH` to 256 |
| `MPS backend error` | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before running |
| Very slow training | Normal on MPS — expect ~2-4 hours for 1B model |
| Loss not decreasing | Check data formatting, try `LEARNING_RATE = 1e-4` |
| `401 Unauthorized` | Re-run `huggingface-cli login`, accept Gemma license |
| Repetitive outputs | Reduce epochs from 3 to 1, or increase `LORA_DROPOUT` to 0.1 |

**Environment variable for MPS memory:**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
```

---

## Quick Reference — Full Workflow

```bash
# 1. Setup
cd ~/Kean/Projects/MaayaTrain
source .venv/bin/activate
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf

# 2. Login
huggingface-cli login

# 3. Train
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python3 finetune_gemma.py

# 4. Test (run ALL)
python3 test_generation.py
python3 test_loss_curve.py
python3 test_perplexity.py
python3 test_consistency.py

# 5. Merge (optional)
python3 merge_model.py
```

---

*Built with MaayaTrain by Akhil Ageer · Kean University*
