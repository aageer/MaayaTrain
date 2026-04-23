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
