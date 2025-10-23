"""
Generate training artifacts for README: loss plot and model parameter info.

This script reads training logs and generates:
- loss.png: Training and validation loss plot
- model_info.txt: Model parameter counts and training info
"""

import json
import os
import matplotlib.pyplot as plt
from modules import build_model, load_tokenizer


def parse_training_log(log_file):
    """
    Parse trainer_log.json to extract metrics.
    
    Args:
        log_file (str): Path to trainer_log.json
    
    Returns:
        dict: Training metrics
    """
    with open(log_file, 'r') as f:
        data = json.load(f)
    return data


def create_loss_plot(log_data, output_path):
    """
    Create a simple loss plot from training log.
    
    Args:
        log_data (dict): Training log data
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # For a simple plot with final metrics
    train_loss = log_data.get('train_loss', 0)
    eval_loss = log_data.get('eval_loss', 0)
    
    # Create bar plot comparing train vs eval loss
    losses = ['Train Loss', 'Validation Loss']
    values = [train_loss, eval_loss]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(losses, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add ROUGE scores as text box
    if 'eval_rouge1' in log_data:
        textstr = f"ROUGE Scores:\n"
        textstr += f"ROUGE-1: {log_data['eval_rouge1']:.4f}\n"
        textstr += f"ROUGE-2: {log_data['eval_rouge2']:.4f}\n"
        textstr += f"ROUGE-L: {log_data['eval_rougeL']:.4f}\n"
        textstr += f"ROUGE-Lsum: {log_data['eval_rougeLsum']:.4f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Loss plot saved to: {output_path}")


def create_model_info(log_data, output_path, model_name='google/flan-t5-base'):
    """
    Create model_info.txt with parameter counts and training details.
    
    Args:
        log_data (dict): Training log data
        output_path (str): Path to save model info
        model_name (str): Model name
    """
    # Load model to get parameter counts
    print(f"\n  Loading model to count parameters...")
    model = build_model(
        model_name=model_name,
        use_lora=True,
        lora_r=log_data.get('lora_r', 8),
        lora_alpha=log_data.get('lora_alpha', 16),
        lora_dropout=log_data.get('lora_dropout', 0.05)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_percent = 100 * trainable_params / total_params
    
    # Format training time
    training_time = log_data.get('training_time_seconds', 0)
    training_time_min = training_time / 60
    training_time_hr = training_time / 3600
    
    # Create info text
    info_text = f"""FLAN-T5 + LoRA Model Information
{'=' * 60}

Model Configuration:
  Base Model: {model_name}
  LoRA Rank (r): {log_data.get('lora_r', 8)}
  LoRA Alpha: {log_data.get('lora_alpha', 16)}
  LoRA Dropout: {log_data.get('lora_dropout', 0.05)}

Parameter Counts:
  Total Parameters:      {total_params:>15,}
  Trainable Parameters:  {trainable_params:>15,}
  Frozen Parameters:     {frozen_params:>15,}
  Trainable Percentage:  {trainable_percent:>14.2f}%

Training Configuration:
  Epochs: {log_data.get('epochs', 'N/A')}
  Batch Size: {log_data.get('batch_size', 'N/A')}
  Learning Rate: {log_data.get('learning_rate', 'N/A')}
  Training Samples: {log_data.get('max_samples', 'Full dataset')}

Training Results:
  Training Loss:         {log_data.get('train_loss', 0):.4f}
  Validation Loss:       {log_data.get('eval_loss', 0):.4f}

ROUGE Scores (Validation):
  ROUGE-1:               {log_data.get('eval_rouge1', 0):.4f}
  ROUGE-2:               {log_data.get('eval_rouge2', 0):.4f}
  ROUGE-L:               {log_data.get('eval_rougeL', 0):.4f}
  ROUGE-Lsum:            {log_data.get('eval_rougeLsum', 0):.4f}

Training Time:
  Seconds:               {training_time:.2f}s
  Minutes:               {training_time_min:.2f}min
  Hours:                 {training_time_hr:.4f}hr

Best Model Selection:
  Metric: {log_data.get('best_model_metric', 'N/A')}
  Value: {log_data.get(f"eval_{log_data.get('best_model_metric', 'rougeLsum')}", 0):.4f}

{'=' * 60}
Generated: {__file__}
"""
    
    with open(output_path, 'w') as f:
        f.write(info_text)
    
    print(f"  Model info saved to: {output_path}")


def generate_artifacts(
    checkpoint_dir='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints',
    model_name='google/flan-t5-base'
):
    """
    Generate all artifacts: loss plot and model info.
    
    Args:
        checkpoint_dir (str): Directory containing trainer_log.json
        model_name (str): Model name for parameter counting
    """
    print("=" * 80)
    print("Generating Training Artifacts")
    print("=" * 80)
    
    log_file = os.path.join(checkpoint_dir, 'trainer_log.json')
    
    if not os.path.exists(log_file):
        print(f"\nError: {log_file} not found!")
        print("Please run training first to generate trainer_log.json")
        return
    
    print(f"\n[1/3] Reading training log: {log_file}")
    log_data = parse_training_log(log_file)
    
    print(f"\n  Training Loss: {log_data.get('train_loss', 0):.4f}")
    print(f"  Validation Loss: {log_data.get('eval_loss', 0):.4f}")
    print(f"  ROUGE-Lsum: {log_data.get('eval_rougeLsum', 0):.4f}")
    
    print(f"\n[2/3] Creating loss plot...")
    loss_plot_path = os.path.join(checkpoint_dir, 'loss.png')
    create_loss_plot(log_data, loss_plot_path)
    
    print(f"\n[3/3] Creating model info file...")
    model_info_path = os.path.join(checkpoint_dir, 'model_info.txt')
    create_model_info(log_data, model_info_path, model_name)
    
    print("\n" + "=" * 80)
    print("SUCCESS! Artifacts Generated:")
    print("=" * 80)
    print(f"  [OK] {loss_plot_path}")
    print(f"  [OK] {model_info_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training artifacts')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints',
                       help='Directory containing trainer_log.json')
    parser.add_argument('--model-name', type=str,
                       default='google/flan-t5-base',
                       help='Model name for parameter counting')
    
    args = parser.parse_args()
    
    generate_artifacts(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name
    )

