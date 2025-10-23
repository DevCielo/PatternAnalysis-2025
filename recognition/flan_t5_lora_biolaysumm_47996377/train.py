"""
Training script for FLAN-T5 LoRA fine-tuning on BioLaySumm dataset.

This script handles model training, validation, and checkpoint saving.
"""

import os
import json
import argparse
import random
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


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  Random seed set to: {seed}")


def get_hardware_info():
    """
    Get hardware information (GPU name, VRAM, etc.).
    
    Returns:
        dict: Hardware information
    """
    info = {
        'device': 'CPU',
        'gpu_name': None,
        'gpu_vram_total_mb': None,
        'gpu_vram_available_mb': None,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__
    }
    
    if torch.cuda.is_available():
        try:
            info['device'] = 'GPU'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            # Get VRAM info
            props = torch.cuda.get_device_properties(0)
            info['gpu_vram_total_mb'] = props.total_memory / (1024**2)
            # Current memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            info['gpu_vram_allocated_mb'] = allocated
            info['gpu_vram_reserved_mb'] = reserved
            info['gpu_vram_available_mb'] = info['gpu_vram_total_mb'] - allocated
        except Exception as e:
            info['gpu_error'] = str(e)
    
    return info


def print_hardware_info(info):
    """
    Print hardware information in a formatted way.
    
    Args:
        info (dict): Hardware information
    """
    print("\n" + "=" * 80)
    print("Hardware Information")
    print("=" * 80)
    print(f"  PyTorch Version: {info['pytorch_version']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  CUDA Version: {info['cuda_version']}")
        print(f"  Device: {info['device']}")
        print(f"  GPU Name: {info['gpu_name']}")
        if info['gpu_vram_total_mb']:
            print(f"  GPU VRAM Total: {info['gpu_vram_total_mb']:.2f} MB")
            print(f"  GPU VRAM Available: {info['gpu_vram_available_mb']:.2f} MB")
    else:
        print(f"  Device: CPU")
    print("=" * 80)


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
    max_target_length=160,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    seed=42
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
        seed (int): Random seed for reproducibility
    """
    print("=" * 80)
    print("FLAN-T5 LoRA Training - BioLaySumm Dataset")
    print("=" * 80)
    
    # Set random seed for reproducibility
    print("\n[Reproducibility] Setting random seed...")
    set_seed(seed)
    
    # Get and print hardware information
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    
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
        generation_max_length=160  # Max length for generated summaries
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
    
    # Save run info (hardware + hyperparameters)
    run_info = {
        'hardware': hardware_info,
        'hyperparameters': {
            'model_name': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_input_length': max_input_length,
            'max_target_length': max_target_length,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'seed': seed
        },
        'results': log_data,
        'timestamp': datetime.now().isoformat()
    }
    
    run_info_file = f"{output_dir}/run_info.txt"
    with open(run_info_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Run Information\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hardware Information:\n")
        f.write("-" * 80 + "\n")
        for key, value in hardware_info.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Hyperparameters:\n")
        f.write("-" * 80 + "\n")
        for key, value in run_info['hyperparameters'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Training Results:\n")
        f.write("-" * 80 + "\n")
        for key, value in log_data.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Timestamp: {run_info['timestamp']}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n[SUCCESS] Training log saved to {log_file}")
    print(f"[SUCCESS] Run info saved to {run_info_file}")
    print(f"[SUCCESS] Best model saved to {best_model_dir}")
    print(f"[SUCCESS] Final model saved to {output_dir}/final")
    
    return trainer, eval_results


if __name__ == "__main__":
    """
    Training script with comprehensive CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train FLAN-T5 with LoRA on BioLaySumm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-csv', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/data/train.csv',
                       help='Path to training CSV')
    parser.add_argument('--val-csv', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/data/val.csv',
                       help='Path to validation CSV')
    parser.add_argument('--output-dir', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints',
                       help='Directory to save checkpoints')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='google/flan-t5-base',
                       help='HuggingFace model name')
    parser.add_argument('--max-input-length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max-target-length', type=int, default=160,
                       help='Maximum target sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max training samples (for testing)')
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    
    # Logging arguments
    parser.add_argument('--save-steps', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--logging-steps', type=int, default=50, help='Log every N steps')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Quick modes
    parser.add_argument('--dry-run', action='store_true',
                       help='Quick test with 50 samples for 1 epoch')
    
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
        save_steps = args.save_steps
        eval_steps = args.eval_steps
        logging_steps = args.logging_steps
    
    # Run training
    trainer, eval_results = train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=args.lr,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seed=args.seed
    )
