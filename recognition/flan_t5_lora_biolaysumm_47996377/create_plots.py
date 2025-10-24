"""
Create training visualization plots for the README.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load training history
checkpoint_path = 'checkpoints/checkpoint-13/trainer_state.json'
with open(checkpoint_path, 'r') as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']

# Extract training loss
train_steps = []
train_losses = []
for entry in log_history:
    if 'loss' in entry:
        train_steps.append(entry['step'])
        train_losses.append(entry['loss'])

# Extract evaluation metrics
eval_steps = []
eval_losses = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
rougeLsum_scores = []

for entry in log_history:
    if 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_losses.append(entry['eval_loss'])
        rouge1_scores.append(entry.get('eval_rouge1', 0))
        rouge2_scores.append(entry.get('eval_rouge2', 0))
        rougeL_scores.append(entry.get('eval_rougeL', 0))
        rougeLsum_scores.append(entry.get('eval_rougeLsum', 0))

# Create output directory
output_dir = 'assets'
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
plt.plot(eval_steps, eval_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/training_loss.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {output_dir}/training_loss.png")
plt.close()

# Plot 2: ROUGE Metrics
plt.figure(figsize=(10, 6))
plt.plot(eval_steps, rouge1_scores, 'g-o', label='ROUGE-1', linewidth=2, markersize=8)
plt.plot(eval_steps, rouge2_scores, 'b-s', label='ROUGE-2', linewidth=2, markersize=8)
plt.plot(eval_steps, rougeL_scores, 'r-^', label='ROUGE-L', linewidth=2, markersize=8)
plt.plot(eval_steps, rougeLsum_scores, 'm-d', label='ROUGE-Lsum', linewidth=2, markersize=8)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('ROUGE Score', fontsize=12)
plt.title('ROUGE Metrics on Validation Set', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 0.4])
plt.tight_layout()
plt.savefig(f'{output_dir}/rouge_metrics.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {output_dir}/rouge_metrics.png")
plt.close()

# Plot 3: Combined view
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Loss
ax1.plot(train_steps, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=8)
ax1.plot(eval_steps, eval_losses, 'r-s', label='Val Loss', linewidth=2, markersize=8)
ax1.set_xlabel('Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Progress: Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: ROUGE
ax2.plot(eval_steps, rouge1_scores, 'g-o', label='ROUGE-1', linewidth=2, markersize=8)
ax2.plot(eval_steps, rouge2_scores, 'b-s', label='ROUGE-2', linewidth=2, markersize=8)
ax2.plot(eval_steps, rougeL_scores, 'r-^', label='ROUGE-L', linewidth=2, markersize=8)
ax2.plot(eval_steps, rougeLsum_scores, 'm-d', label='ROUGE-Lsum', linewidth=2, markersize=8)
ax2.set_xlabel('Steps', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Training Progress: ROUGE Metrics', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.4])

plt.tight_layout()
plt.savefig(f'{output_dir}/training_overview.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {output_dir}/training_overview.png")
plt.close()

print("\n[SUCCESS] All plots created successfully!")

