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
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ─── Configuration ───────────────────────────────────────────
MODEL_ID = "google/gemma-3-1b-pt"
DATASET_ID = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./gemma-finetuned"
MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8        # effective batch = 1 * 8 = 8
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
    dtype=torch.float32,  # MPS requires float32 for stable training
    device_map=None,  # we handle placement manually for MPS
)

# ─── LoRA Config ─────────────────────────────────────────────
print("Configuring LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

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

# ─── SFT Training Config ────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=50,
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
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

# ─── Train ───────────────────────────────────────────────────
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

# ─── Save ────────────────────────────────────────────────────
print("Saving model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model saved to {OUTPUT_DIR}/final")
print("Training complete!")
