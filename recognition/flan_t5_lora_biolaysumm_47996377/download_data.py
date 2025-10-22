"""
Download and prepare BioLaySumm2025 dataset from Hugging Face.
Saves as train.csv, val.csv, test.csv with columns: report_text, lay_summary
"""
import os
from datasets import load_dataset
import pandas as pd

# Create data directory
os.makedirs('recognition/flan_t5_lora_biolaysumm_47996377/data', exist_ok=True)

print("Downloading BioLaySumm2025 dataset from Hugging Face...")
dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

print(f"\nDataset loaded:")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Validation: {len(dataset['validation'])} samples")
print(f"  Test: {len(dataset['test'])} samples")

# Convert each split to DataFrame and rename columns
for split_name, output_name in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
    df = dataset[split_name].to_pandas()
    
    # Rename columns: radiology_report -> report_text, layman_report -> lay_summary
    df_clean = pd.DataFrame({
        'report_text': df['radiology_report'],
        'lay_summary': df['layman_report']
    })
    
    # Save to CSV
    output_path = f'recognition/flan_t5_lora_biolaysumm_47996377/data/{output_name}.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {output_name}.csv ({len(df_clean)} rows)")
    print(f"  Columns: {list(df_clean.columns)}")
    print(f"  Sample report_text: {df_clean['report_text'].iloc[0][:80]}...")
    print(f"  Sample lay_summary: {df_clean['lay_summary'].iloc[0][:80]}...")

print("\n[SUCCESS] All splits saved successfully!")
print("\nFile locations:")
print("  - recognition/flan_t5_lora_biolaysumm_47996377/data/train.csv")
print("  - recognition/flan_t5_lora_biolaysumm_47996377/data/val.csv")
print("  - recognition/flan_t5_lora_biolaysumm_47996377/data/test.csv")

