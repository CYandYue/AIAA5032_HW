#!/usr/bin/env python3
"""
Run Fisher Vector pipeline on existing GMM model
Fisher Vector typically achieves 5-8% improvement over Bag-of-Features
"""

import os
import sys
import subprocess
from datetime import datetime

def run_fisher_vector_pipeline(gmm_exp_dir, cluster_num):
    """
    Run Fisher Vector extraction and training on existing GMM experiment

    Args:
        gmm_exp_dir: Path to GMM experiment directory (contains models/ and features/)
        cluster_num: Number of GMM components
    """
    work_dir = '/home/YueChang/phd_ws/AIAA5032/homework/hw1'
    os.chdir(work_dir)

    print("=" * 70)
    print("FISHER VECTOR PIPELINE")
    print("=" * 70)
    print(f"Base GMM experiment: {gmm_exp_dir}")
    print(f"GMM components: {cluster_num}")
    print("=" * 70)

    # Check GMM model exists
    gmm_model = f'{gmm_exp_dir}/models/gmm.{cluster_num}.model'
    if not os.path.exists(gmm_model):
        print(f"ERROR: GMM model not found: {gmm_model}")
        return False

    # Create new experiment directory for Fisher Vector
    exp_name = f'fisher_k{cluster_num}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir = f'experiments/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)

    features_dir = f'{exp_dir}/features'
    models_dir = f'{exp_dir}/models'
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n✓ Created experiment directory: {exp_dir}")

    # Step 1: Extract Fisher Vectors
    print("\n" + "=" * 70)
    print("STEP 1: Extracting Fisher Vectors")
    print("=" * 70)
    print("This will take ~10-15 minutes...")

    cmd = f"python get_fisher_vector.py {gmm_model} data/videos.name.lst " \
          f"--mfcc_path data/mfcc/ --output_path {features_dir}/"

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("ERROR: Fisher Vector extraction failed")
        return False

    # Calculate feature dimension
    fv_dim = 2 * cluster_num * 13  # 2 * K * d (assuming 13 MFCC dimensions)
    print(f"\n✓ Fisher Vector dimension: {fv_dim}")

    # Step 2: Train classifiers
    print("\n" + "=" * 70)
    print("STEP 2: Training classifiers")
    print("=" * 70)

    classifiers = {
        'svm': f'python train_svm_multiclass.py {features_dir}/ {fv_dim} labels/train.csv {models_dir}/fisher-{cluster_num}.svm.model',
        'lr': f'python train_LR.py {features_dir}/ {fv_dim} labels/train.csv {models_dir}/fisher-{cluster_num}.lr.model',
        'mlp': f'python train_mlp.py {features_dir}/ {fv_dim} labels/train.csv {models_dir}/fisher-{cluster_num}.mlp.model'
    }

    for clf, cmd in classifiers.items():
        print(f"\n  Training {clf.upper()}...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  WARNING: {clf.upper()} training failed")

    # Step 3: Test on validation set
    print("\n" + "=" * 70)
    print("STEP 3: Testing on validation set")
    print("=" * 70)

    test_scripts = {
        'svm': 'test_svm_multiclass.py',
        'lr': 'test_LR.py',
        'mlp': 'test_mlp.py'
    }

    for clf, script in test_scripts.items():
        model_file = f'{models_dir}/fisher-{cluster_num}.{clf}.model'
        output_file = f'{exp_dir}/val_{clf}.csv'
        cmd = f'python {script} {model_file} {features_dir}/ {fv_dim} labels/val.csv {output_file}'

        print(f"  {clf.upper()}...", end=' ', flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            print("✓")
        else:
            print("✗")

    # Step 4: Test on Kaggle test set
    print("\n" + "=" * 70)
    print("STEP 4: Testing on Kaggle test set")
    print("=" * 70)

    for clf, script in test_scripts.items():
        model_file = f'{models_dir}/fisher-{cluster_num}.{clf}.model'
        output_file = f'{exp_dir}/test_{clf}.csv'
        cmd = f'python {script} {model_file} {features_dir}/ {fv_dim} labels/test_for_student.label {output_file}'

        print(f"  {clf.upper()}...", end=' ', flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            print("✓")
        else:
            print("✗")

    # Step 5: Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION: Fisher Vector vs GMM BoF")
    print("=" * 70)

    import pandas as pd

    val_gt = pd.read_csv('labels/val.csv')
    gt_dict = dict(zip(val_gt['Id'], val_gt['Category']))

    print("\nFisher Vector Results:")
    fv_results = {}
    for clf in ['svm', 'lr', 'mlp']:
        pred_file = f'{exp_dir}/val_{clf}.csv'
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)
            correct = sum(1 for _, row in pred_df.iterrows()
                         if row['Id'] in gt_dict and gt_dict[row['Id']] == row['Category'])
            total = len([1 for _, row in pred_df.iterrows() if row['Id'] in gt_dict])
            acc = correct / total * 100
            fv_results[clf] = acc
            print(f"  {clf.upper()}: {acc:.2f}% ({correct}/{total})")

    # Compare with GMM BoF
    print("\nGMM BoF Results (reference):")
    bof_results = {}
    for clf in ['svm', 'lr', 'mlp']:
        pred_file = f'{gmm_exp_dir}/val_{clf}.csv'
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)
            correct = sum(1 for _, row in pred_df.iterrows()
                         if row['Id'] in gt_dict and gt_dict[row['Id']] == row['Category'])
            total = len([1 for _, row in pred_df.iterrows() if row['Id'] in gt_dict])
            acc = correct / total * 100
            bof_results[clf] = acc
            print(f"  {clf.upper()}: {acc:.2f}% ({correct}/{total})")

    print("\n" + "-" * 70)
    print("IMPROVEMENT:")
    print("-" * 70)
    for clf in ['svm', 'lr', 'mlp']:
        if clf in fv_results and clf in bof_results:
            improvement = fv_results[clf] - bof_results[clf]
            marker = "📈" if improvement > 0 else "📉"
            print(f"  {clf.upper()}: {improvement:+.2f}% {marker}")

    # Find best
    if fv_results:
        best_clf = max(fv_results.items(), key=lambda x: x[1])
        print(f"\n🏆 Best Fisher Vector model: {best_clf[0].upper()} - {best_clf[1]:.2f}%")

    print("\n" + "=" * 70)
    print(f"✓ COMPLETE! Results saved to: {exp_dir}")
    print("=" * 70)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Fisher Vector pipeline')
    parser.add_argument('--gmm_exp_dir', required=True,
                       help='GMM experiment directory (e.g., experiments/gmm_k200_soft_xxx)')
    parser.add_argument('--clusters', type=int, required=True,
                       help='Number of GMM components')

    args = parser.parse_args()

    success = run_fisher_vector_pipeline(args.gmm_exp_dir, args.clusters)
    sys.exit(0 if success else 1)
