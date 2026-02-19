#!/usr/bin/env python3
"""
Rerun testing only (without retraining models)
Use this when models are already trained but you need to regenerate predictions
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_test(cluster_num=50):
    """Rerun testing on validation and test sets"""

    work_dir = '/home/YueChang/phd_ws/AIAA5032/homework/hw1'
    os.chdir(work_dir)

    bof_dir = 'features/bof'
    models_dir = 'models'
    labels_dir = 'labels'

    # Find the latest experiment directory or create new one
    exp_dir = os.path.join(work_dir, 'experiments', f'k{cluster_num}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_retest')
    os.makedirs(exp_dir, exist_ok=True)

    print("=" * 70)
    print(f"Rerunning tests for cluster_num={cluster_num}")
    print(f"Output directory: {exp_dir}")
    print("=" * 70)

    # Check if models exist
    required_models = [
        f'{models_dir}/mfcc-{cluster_num}.svm.model',
        f'{models_dir}/mfcc-{cluster_num}.lr.model',
        f'{models_dir}/mfcc-{cluster_num}.mlp.model'
    ]

    for model in required_models:
        if not os.path.exists(model):
            print(f"ERROR: Model not found: {model}")
            print("Please run training first: python run_baseline.py")
            return False

    print("\n✓ All models found")

    # Run tests
    classifiers = ['svm', 'lr', 'mlp']
    test_scripts = {
        'svm': 'test_svm_multiclass.py',
        'lr': 'test_LR.py',
        'mlp': 'test_mlp.py'
    }

    print("\nTesting on validation set...")
    print("-" * 70)
    for clf in classifiers:
        cmd = f"python {test_scripts[clf]} {models_dir}/mfcc-{cluster_num}.{clf}.model {bof_dir}/ {cluster_num} {labels_dir}/val.csv {exp_dir}/val_{clf}.csv"
        print(f"  {clf.upper()}...", end=' ', flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓")
        else:
            print(f"✗\n{result.stderr}")

    print("\nTesting on Kaggle test set...")
    print("-" * 70)
    for clf in classifiers:
        cmd = f"python {test_scripts[clf]} {models_dir}/mfcc-{cluster_num}.{clf}.model {bof_dir}/ {cluster_num} {labels_dir}/test_for_student.label {exp_dir}/test_{clf}.csv"
        print(f"  {clf.upper()}...", end=' ', flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓")
        else:
            print(f"✗\n{result.stderr}")

    # Calculate validation accuracy
    print("\nCalculating validation accuracy...")
    print("-" * 70)

    import pandas as pd

    val_gt = pd.read_csv(f'{labels_dir}/val.csv')
    gt_dict = dict(zip(val_gt['Id'], val_gt['Category']))

    for clf in classifiers:
        pred_file = f'{exp_dir}/val_{clf}.csv'
        if not os.path.exists(pred_file):
            print(f"  {clf.upper()}: Prediction file not found")
            continue

        pred_df = pd.read_csv(pred_file)
        correct = 0
        total = 0

        for _, row in pred_df.iterrows():
            video_id = row['Id']
            pred_class = row['Category']

            if video_id in gt_dict:
                total += 1
                if gt_dict[video_id] == pred_class:
                    correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"  {clf.upper()}: {accuracy*100:.2f}% ({correct}/{total})")

    print("\n" + "=" * 70)
    print("Testing complete!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 70)

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rerun testing without retraining')
    parser.add_argument('--clusters', type=int, default=50,
                       help='Number of clusters (must match trained models)')

    args = parser.parse_args()

    success = run_test(cluster_num=args.clusters)
    sys.exit(0 if success else 1)
