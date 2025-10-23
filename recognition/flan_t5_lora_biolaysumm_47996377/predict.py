"""
Inference script for generating layperson summaries from radiology reports.

This script loads the trained model and runs predictions on the test set,
computing ROUGE scores and saving example predictions.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import evaluate
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from modules import load_tokenizer
from dataset import BioLaySummDS


# Load ROUGE metric
rouge_metric = evaluate.load('rouge')


def load_model(checkpoint_path):
    """
    Load trained LoRA model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint directory
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load model with PEFT
    from transformers import AutoModelForSeq2SeqLM
    
    # Get base model name from config
    config = PeftConfig.from_pretrained(checkpoint_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    print(f"  Model loaded successfully")
    print(f"  Base model: {config.base_model_name_or_path}")
    
    return model, tokenizer


def generate_predictions(model, tokenizer, test_dataset, batch_size=4, max_length=128):
    """
    Generate predictions for the test dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        batch_size (int): Batch size for inference
        max_length (int): Maximum generation length
    
    Returns:
        list: List of generated summaries
        list: List of reference summaries
        list: List of input texts
    """
    model.eval()
    device = torch.device('cpu')  # Force CPU for compatibility
    model.to(device)
    
    predictions = []
    references = []
    inputs = []
    
    print(f"\nGenerating predictions for {len(test_dataset)} samples...")
    print(f"  Batch size: {batch_size}")
    print(f"  Max generation length: {max_length}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Predicting"):
            batch_end = min(i + batch_size, len(test_dataset))
            batch = [test_dataset[j] for j in range(i, batch_end)]
            
            # Get input_ids and attention_mask
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode predictions
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)
            
            # Decode references (from labels)
            for item in batch:
                labels = item['labels'].numpy()
                # Replace -100 with pad_token_id
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                reference = tokenizer.decode(labels, skip_special_tokens=True)
                references.append(reference)
                
                # Decode input for context
                input_ids_np = item['input_ids'].numpy()
                input_text = tokenizer.decode(input_ids_np, skip_special_tokens=True)
                inputs.append(input_text)
    
    return predictions, references, inputs


def compute_rouge_scores(predictions, references):
    """
    Compute ROUGE scores for predictions.
    
    Args:
        predictions (list): List of predicted summaries
        references (list): List of reference summaries
    
    Returns:
        dict: ROUGE scores
    """
    print("\nComputing ROUGE scores...")
    
    # ROUGE expects newline after each sentence
    formatted_preds = ["\n".join(pred.strip().split(".")) if pred.strip() else "empty" 
                       for pred in predictions]
    formatted_refs = ["\n".join(ref.strip().split(".")) if ref.strip() else "empty" 
                      for ref in references]
    
    result = rouge_metric.compute(
        predictions=formatted_preds,
        references=formatted_refs,
        use_stemmer=True,
        use_aggregator=True
    )
    
    scores = {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL'],
        'rougeLsum': result['rougeLsum']
    }
    
    return scores


def save_predictions(predictions, references, inputs, rouge_scores, output_dir, num_samples=5):
    """
    Save predictions and ROUGE scores to files.
    
    Args:
        predictions (list): List of predicted summaries
        references (list): List of reference summaries
        inputs (list): List of input texts
        rouge_scores (dict): ROUGE scores
        output_dir (str): Directory to save predictions
        num_samples (int): Number of example samples to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ROUGE scores
    rouge_file = os.path.join(output_dir, 'rouge.json')
    with open(rouge_file, 'w') as f:
        json.dump(rouge_scores, f, indent=2)
    print(f"\n  Saved ROUGE scores to: {rouge_file}")
    
    # Save sample predictions (first N examples)
    samples_file = os.path.join(output_dir, 'samples.jsonl')
    with open(samples_file, 'w') as f:
        for i in range(min(num_samples, len(predictions))):
            sample = {
                'index': i,
                'input': inputs[i],
                'prediction': predictions[i],
                'reference': references[i]
            }
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved {min(num_samples, len(predictions))} samples to: {samples_file}")


def predict(
    checkpoint_path='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints/best',
    test_csv='recognition/flan_t5_lora_biolaysumm_47996377/data/test.csv',
    output_dir='recognition/flan_t5_lora_biolaysumm_47996377/predictions',
    batch_size=4,
    max_length=160,
    num_samples=5,
    max_test_samples=None
):
    """
    Run inference on test set and save results.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        test_csv (str): Path to test CSV
        output_dir (str): Directory to save predictions
        batch_size (int): Batch size for inference
        max_length (int): Maximum generation length
        num_samples (int): Number of example samples to save
        max_test_samples (int): Maximum test samples to use (for quick testing)
    """
    print("=" * 80)
    print("FLAN-T5 LoRA Inference - Test Set Evaluation")
    print("=" * 80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model(checkpoint_path)
    
    # Load test dataset
    print("\n[2/5] Loading test dataset...")
    test_dataset = BioLaySummDS(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=128
    )
    
    # Limit test samples if specified
    if max_test_samples is not None:
        print(f"  Limiting to {max_test_samples} test samples for quick evaluation")
        test_dataset.data = test_dataset.data.head(max_test_samples)
    
    # Generate predictions
    print("\n[3/5] Generating predictions...")
    predictions, references, inputs = generate_predictions(
        model, tokenizer, test_dataset, batch_size, max_length
    )
    
    # Compute ROUGE scores
    print("\n[4/5] Computing ROUGE scores...")
    rouge_scores = compute_rouge_scores(predictions, references)
    
    # Print ROUGE scores
    print("\n" + "=" * 80)
    print("Test Set ROUGE Scores:")
    print("=" * 80)
    print(f"  ROUGE-1:    {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2:    {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L:    {rouge_scores['rougeL']:.4f}")
    print(f"  ROUGE-Lsum: {rouge_scores['rougeLsum']:.4f}")
    print("=" * 80)
    
    # Save predictions
    print("\n[5/5] Saving predictions...")
    save_predictions(predictions, references, inputs, rouge_scores, output_dir, num_samples)
    
    print("\n" + "=" * 80)
    print("SUCCESS! Inference Complete")
    print("=" * 80)
    print(f"  Total predictions: {len(predictions)}")
    print(f"  ROUGE scores saved: {output_dir}/rouge.json")
    print(f"  Example samples saved: {output_dir}/samples.jsonl")
    print("=" * 80)
    
    return rouge_scores, predictions, references


if __name__ == "__main__":
    """
    Run inference on test set.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('--checkpoint', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/checkpoints/best',
                       help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/data/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--output-dir', type=str,
                       default='recognition/flan_t5_lora_biolaysumm_47996377/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--max-length', type=int, default=160,
                       help='Maximum generation length')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of example samples to save')
    parser.add_argument('--max-test-samples', type=int, default=None,
                       help='Maximum test samples to use (for quick testing)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 20 samples')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("\n[QUICK TEST MODE] Using 20 test samples")
        max_test_samples = 20
    else:
        max_test_samples = args.max_test_samples
    
    # Run inference
    rouge_scores, predictions, references = predict(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_samples=args.num_samples,
        max_test_samples=max_test_samples
    )
