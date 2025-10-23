"""
Dataset loader and preprocessor for BioLaySumm data.

This module implements the dataset class for loading and tokenizing
radiology reports and their layperson summaries.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BioLaySummDS(Dataset):
    """
    Dataset class for BioLaySumm radiology report to layperson summary translation.
    
    Args:
        csv_path (str): Path to the CSV file containing report_text and lay_summary columns
        tokenizer: HuggingFace tokenizer (e.g., FLAN-T5 tokenizer)
        max_input_length (int): Maximum length for input sequences (default: 512)
        max_target_length (int): Maximum length for target sequences (default: 128)
        prefix (str): Prefix prompt to add before each input report
    """
    
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_input_length=512,
        max_target_length=128,
        prefix="Explain the following medical report in simple terms that a patient can understand: "
    ):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        
        print(f"Loaded dataset from {csv_path}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Max input length: {max_input_length}")
        print(f"  Max target length: {max_target_length}")
        print(f"  Prefix: '{prefix}'")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized example.
        
        Returns:
            dict: Contains 'input_ids', 'attention_mask', and 'labels'
        """
        row = self.data.iloc[idx]
        report_text = str(row['report_text'])
        lay_summary = str(row['lay_summary'])
        
        # Add prefix to input
        input_text = self.prefix + report_text
        
        # Tokenize input (report)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (summary)
        target_encoding = self.tokenizer(
            lay_summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract tensors and squeeze batch dimension
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        
        # Replace padding token id with -100 for labels
        # -100 is ignored by PyTorch CrossEntropyLoss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_tokenizer(model_name='google/flan-t5-base'):
    """
    Load the FLAN-T5 tokenizer.
    
    Args:
        model_name (str): HuggingFace model name
    
    Returns:
        tokenizer: HuggingFace tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


if __name__ == "__main__":
    """
    Smoke test: Load one sample and print tensor shapes.
    """
    print("=" * 60)
    print("BioLaySummDS Dataset Smoke Test")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/3] Loading FLAN-T5 tokenizer...")
    tokenizer = load_tokenizer('google/flan-t5-base')
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    
    # Load dataset
    print("\n[2/3] Loading dataset...")
    dataset = BioLaySummDS(
        csv_path='recognition/flan_t5_lora_biolaysumm_47996377/data/train.csv',
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=128
    )
    
    # Get first sample
    print("\n[3/3] Fetching first sample...")
    sample = dataset[0]
    
    print("\n" + "=" * 60)
    print("Sample Tensor Shapes:")
    print("=" * 60)
    print(f"  input_ids: {list(sample['input_ids'].shape)}")
    print(f"  attention_mask: {list(sample['attention_mask'].shape)}")
    print(f"  labels: {list(sample['labels'].shape)}")
    
    print("\n" + "=" * 60)
    print("Sample Content (decoded):")
    print("=" * 60)
    
    # Decode input (skip padding)
    input_ids = sample['input_ids']
    non_pad_input = input_ids[input_ids != tokenizer.pad_token_id]
    decoded_input = tokenizer.decode(non_pad_input, skip_special_tokens=True)
    print(f"\nInput (first 200 chars):\n{decoded_input[:200]}...")
    
    # Decode labels (skip -100 padding)
    labels = sample['labels']
    valid_labels = labels[labels != -100]
    decoded_output = tokenizer.decode(valid_labels, skip_special_tokens=True)
    print(f"\nTarget summary (first 200 chars):\n{decoded_output[:200]}...")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Dataset loader working correctly!")
    print("=" * 60)
