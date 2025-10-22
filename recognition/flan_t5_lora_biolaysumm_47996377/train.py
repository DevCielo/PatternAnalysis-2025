"""
Training script for FLAN-T5 LoRA fine-tuning on BioLaySumm dataset.

This script handles model training, validation, and checkpoint saving.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import torch
from dataset import BioLaySummDS, load_tokenizer
from modules import build_model, print_model_info


def compute_metrics(eval_pred):
    """
    Compute ROUGE metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        dict: Dictionary of metric scores
    """
    # TODO: Implement full ROUGE computation in MP4
    # For now, return a stub to prove the training loop works
    predictions, labels = eval_pred
    
    # Simple metrics stub
    metrics = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0
    }
    
    return metrics


def train(
    train_csv='recognition/flan_t5_lora_biolaysumm_47996377/data/train.csv',
    val_csv='recognition/flan_t5_lora_biolaysumm_47996377/data/val.csv',
    model_name='google/flan-t5-base',
    output_dir='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints',
    max_samples=None,
    epochs=1,
    batch_size=8,
    learning_rate=1e-3,
    max_input_length=512,
    max_target_length=128,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    save_steps=100,
    eval_steps=100,
    logging_steps=10
):
    """
    Train FLAN-T5 with LoRA on BioLaySumm dataset.
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        model_name (str): HuggingFace model name
        output_dir (str): Directory to save checkpoints
        max_samples (int): Max samples to use (for dry-run testing)
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        max_input_length (int): Max input sequence length
        max_target_length (int): Max target sequence length
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha
        lora_dropout (float): LoRA dropout
        save_steps (int): Save checkpoint every N steps
        eval_steps (int): Evaluate every N steps
        logging_steps (int): Log every N steps
    """
    print("=" * 80)
    print("FLAN-T5 LoRA Training - BioLaySumm Dataset")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    
    # Load datasets
    print("\n[2/5] Loading datasets...")
    train_dataset = BioLaySummDS(
        csv_path=train_csv,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    val_dataset = BioLaySummDS(
        csv_path=val_csv,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    # Limit dataset size for dry-run
    if max_samples is not None:
        print(f"\n  [DRY-RUN MODE] Limiting to {max_samples} samples")
        train_dataset.data = train_dataset.data.head(max_samples)
        val_dataset.data = val_dataset.data.head(min(max_samples // 5, len(val_dataset.data)))
        print(f"    Train samples: {len(train_dataset.data)}")
        print(f"    Val samples: {len(val_dataset.data)}")
    
    # Build model with LoRA
    print("\n[3/5] Building model with LoRA...")
    model = build_model(
        model_name=model_name,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    print_model_info(model)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    print("\n[4/5] Configuring training arguments...")
    
    # Check GPU compatibility
    use_gpu = False  # Force CPU for compatibility
    if torch.cuda.is_available():
        print("  Note: GPU detected but forcing CPU due to compatibility issues")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=False,  # Will implement in MP4
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard for now
        seed=42,
        fp16=False,  # Disable for CPU compatibility
        use_cpu=True,  # Force CPU
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False
    )
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: CPU")
    print(f"  Mixed precision (FP16): False")
    
    # Initialize Trainer
    print("\n[5/5] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # ROUGE stub for now
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"  Total time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  Final train loss: {train_result.training_loss:.4f}")
    
    trainer.save_model(f"{output_dir}/final")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Running Validation...")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    print("\nValidation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")
    
    # Save training log
    log_data = {
        'train_loss': float(train_result.training_loss),
        'eval_loss': float(eval_results['eval_loss']),
        'training_time_seconds': training_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'max_samples': max_samples
    }
    
    with open(f"{output_dir}/training_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n[SUCCESS] Training log saved to {output_dir}/training_log.json")
    print(f"[SUCCESS] Model saved to {output_dir}/final")
    
    return trainer, eval_results


if __name__ == "__main__":
    """
    Dry-run training script with minimal samples.
    """
    parser = argparse.ArgumentParser(description='Train FLAN-T5 with LoRA on BioLaySumm')
    
    parser.add_argument('--dry-run', action='store_true', help='Run with 50 samples for testing')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    
    args = parser.parse_args()
    
    # Dry-run defaults
    if args.dry_run:
        print("\n[DRY-RUN MODE ENABLED]")
        max_samples = 50
        epochs = 1
        batch_size = 4
        save_steps = 10
        eval_steps = 10
        logging_steps = 5
    else:
        max_samples = args.max_samples
        epochs = args.epochs
        batch_size = args.batch_size
        save_steps = 500
        eval_steps = 500
        logging_steps = 50
    
    # Run training
    trainer, eval_results = train(
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps
    )
