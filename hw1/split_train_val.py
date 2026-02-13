#!/usr/bin/env python3
"""
Split trainval.csv into train.csv and val.csv with stratified sampling
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split_dataset(input_csv, train_output, val_output, val_ratio=0.15, random_state=42):
    """
    Split dataset with stratified sampling to maintain class distribution

    Args:
        input_csv: path to trainval.csv
        train_output: path to save train.csv
        val_output: path to save val.csv
        val_ratio: validation set ratio (default 0.15 = 15%)
        random_state: random seed for reproducibility
    """
    # Read data
    df = pd.read_csv(input_csv)

    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['Category'].value_counts().sort_index())

    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df['Category'],  # Keep class distribution
        random_state=random_state
    )

    # Save to CSV
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    print(f"\n✓ Train set: {len(train_df)} samples -> {train_output}")
    print(f"✓ Validation set: {len(val_df)} samples -> {val_output}")

    print(f"\nTrain class distribution:")
    print(train_df['Category'].value_counts().sort_index())
    print(f"\nValidation class distribution:")
    print(val_df['Category'].value_counts().sort_index())

    # Verify proportions
    print("\nClass proportions verification:")
    for cat in sorted(df['Category'].unique()):
        original_prop = (df['Category'] == cat).mean()
        train_prop = (train_df['Category'] == cat).mean()
        val_prop = (val_df['Category'] == cat).mean()
        print(f"Class {cat}: Original={original_prop:.3f}, Train={train_prop:.3f}, Val={val_prop:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split trainval.csv into train and validation sets')
    parser.add_argument('--input', default='labels/trainval.csv', help='Input trainval.csv path')
    parser.add_argument('--train_output', default='labels/train.csv', help='Output train.csv path')
    parser.add_argument('--val_output', default='labels/val.csv', help='Output val.csv path')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    split_dataset(
        args.input,
        args.train_output,
        args.val_output,
        args.val_ratio,
        args.seed
    )
