"""
Model components for FLAN-T5 with LoRA fine-tuning.

This module contains functions to load the tokenizer and build the FLAN-T5 model
with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType


def load_tokenizer(model_name='google/flan-t5-base'):
    """
    Load the FLAN-T5 tokenizer.
    
    Args:
        model_name (str): HuggingFace model name (default: google/flan-t5-base)
    
    Returns:
        tokenizer: HuggingFace tokenizer
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    return tokenizer


def build_model(
    model_name='google/flan-t5-base',
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None
):
    """
    Build FLAN-T5 model with optional LoRA for parameter-efficient fine-tuning.
    
    Args:
        model_name (str): HuggingFace model name (default: google/flan-t5-base)
        use_lora (bool): Whether to apply LoRA (default: True)
        lora_r (int): LoRA rank (default: 8)
        lora_alpha (int): LoRA alpha scaling parameter (default: 16)
        lora_dropout (float): LoRA dropout rate (default: 0.05)
        target_modules (list): Modules to apply LoRA to (default: q, v projection layers)
    
    Returns:
        model: HuggingFace model (with LoRA if use_lora=True)
    """
    print(f"\nLoading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Print base model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    if use_lora:
        print(f"\nApplying LoRA configuration:")
        print(f"  Rank (r): {lora_r}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Dropout: {lora_dropout}")
        
        # Default target modules for T5: query and value projections
        if target_modules is None:
            target_modules = ["q", "v"]
        
        print(f"  Target modules: {target_modules}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


def print_model_info(model):
    """
    Print detailed model parameter information.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 60)
    print("Model Parameter Summary")
    print("=" * 60)
    print(f"Total parameters:      {total_params:>15,}")
    print(f"Trainable parameters:  {trainable_params:>15,}")
    print(f"Frozen parameters:     {frozen_params:>15,}")
    print(f"Trainable percentage:  {100 * trainable_params / total_params:>14.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    """
    Test module loading.
    """
    print("=" * 60)
    print("Testing FLAN-T5 + LoRA Model Loading")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Build model with LoRA
    model = build_model(
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    # Print info
    print_model_info(model)
    
    print("\n[SUCCESS] Model and tokenizer loaded successfully!")
