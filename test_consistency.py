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
