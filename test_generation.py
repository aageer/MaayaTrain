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
