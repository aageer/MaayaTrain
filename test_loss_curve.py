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
