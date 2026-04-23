"""Merge LoRA adapter into base model for standalone deployment."""
import os
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
