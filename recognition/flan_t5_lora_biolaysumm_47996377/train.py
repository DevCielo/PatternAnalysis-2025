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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import torch
import evaluate
from dataset import BioLaySummDS, load_tokenizer
from modules import build_model, print_model_info

# Load ROUGE metric
rouge_metric = evaluate.load('rouge')


def compute_metrics(eval_pred, tokenizer):
    """
    Compute ROUGE metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        tokenizer: Tokenizer for decoding
    
    Returns:
        dict: Dictionary of ROUGE scores
    """
    predictions, labels = eval_pred
    
    # Handle predictions: they might be logits or generated IDs
    # If logits, take argmax; if IDs, use directly
    if len(predictions.shape) == 3:
        # predictions are logits (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 padding in predictions with pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects newline after each sentence for proper scoring
    decoded_preds = ["\n".join(pred.strip().split(".")) if pred.strip() else "empty" for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split(".")) if label.strip() else "empty" for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True
    )
    
    # Extract the metrics we need
    metrics = {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL'],
        'rougeLsum': result['rougeLsum']
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
    
    training_args = Seq2SeqTrainingArguments(
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
        load_best_model_at_end=True,  # Load best model based on ROUGE
        metric_for_best_model="rougeLsum",  # Use ROUGE-Lsum for best model selection
        greater_is_better=True,  # Higher ROUGE is better
        report_to="none",  # Disable wandb/tensorboard for now
        seed=42,
        fp16=False,  # Disable for CPU compatibility
        use_cpu=True,  # Force CPU
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False,
        predict_with_generate=True,  # Enable text generation for evaluation
        generation_max_length=128  # Max length for generated summaries
    )
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: CPU")
    print(f"  Mixed precision (FP16): False")
    print(f"  Predict with generate: True")
    print(f"  Generation max length: 128")
    
    # Initialize Trainer
    print("\n[5/5] Initializing Trainer...")
    
    # Create compute_metrics closure with tokenizer
    def compute_metrics_with_tokenizer(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_with_tokenizer  # ROUGE evaluation
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    
    # Save best and final models
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"  Total time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  Final train loss: {train_result.training_loss:.4f}")
    
    # Save final model
    trainer.save_model(f"{output_dir}/final")
    print(f"\n  Saved final model to: {output_dir}/final")
    
    # Save best model (based on rougeLsum)
    # The trainer already loaded the best model at the end
    best_model_dir = f"{output_dir}/best"
    trainer.save_model(best_model_dir)
    print(f"  Saved best model to: {best_model_dir}")
    
    # Evaluate on validation set with best model
    print("\n" + "=" * 80)
    print("Running Final Validation with Best Model...")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    print("\nValidation Results (Best Model):")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save training log with ROUGE scores
    log_data = {
        'train_loss': float(train_result.training_loss),
        'eval_loss': float(eval_results['eval_loss']),
        'eval_rouge1': float(eval_results.get('eval_rouge1', 0.0)),
        'eval_rouge2': float(eval_results.get('eval_rouge2', 0.0)),
        'eval_rougeL': float(eval_results.get('eval_rougeL', 0.0)),
        'eval_rougeLsum': float(eval_results.get('eval_rougeLsum', 0.0)),
        'training_time_seconds': training_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'max_samples': max_samples,
        'best_model_metric': 'rougeLsum'
    }
    
    log_file = f"{output_dir}/trainer_log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n[SUCCESS] Training log saved to {log_file}")
    print(f"[SUCCESS] Best model saved to {best_model_dir}")
    print(f"[SUCCESS] Final model saved to {output_dir}/final")
    
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
