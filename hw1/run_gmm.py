#!/usr/bin/env python3
"""
Run GMM-based baseline (more advanced than K-Means)
GMM typically performs better for audio features
"""

import os
import sys
import subprocess
import argparse

def run_gmm_pipeline(cluster_num=50, use_soft=True):
    """Run complete GMM pipeline"""

    work_dir = '/home/YueChang/phd_ws/AIAA5032/homework/hw1'
    os.chdir(work_dir)

    print("=" * 70)
    print(f"GMM BASELINE PIPELINE (clusters={cluster_num}, soft={use_soft})")
    print("=" * 70)

    # Directories
    temp_dir = 'temp'
    labels_dir = 'labels'
    exp_suffix = 'soft' if use_soft else 'hard'

    # Create experiment directory first
    from datetime import datetime
    exp_name = f'gmm_k{cluster_num}_{exp_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir = f'experiments/{exp_name}'
    os.makedirs(exp_dir, exist_ok=True)

    # Models and features saved in experiment directory
    models_dir = f'{exp_dir}/models'
    features_dir = f'{exp_dir}/features'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    print(f"\n✓ Experiment directory: {exp_dir}")
    print(f"  ├─ models/    - Trained models")
    print(f"  ├─ features/  - GMM BoF features")
    print(f"  ├─ val_*.csv  - Validation predictions")
    print(f"  └─ test_*.csv - Kaggle test predictions")

    # File paths
    selected_mfcc = f'{temp_dir}/selected.mfcc.csv'
    gmm_model = f'{models_dir}/gmm.{cluster_num}.model'

    # Check if selected MFCC exists
    if not os.path.exists(selected_mfcc):
        print(f"ERROR: {selected_mfcc} not found")
        print("Please run baseline first: python run_baseline.py")
        return False

    # Step 1: Train GMM
    print("\n" + "=" * 70)
    print("STEP 1: Training GMM")
    print("=" * 70)

    cmd = f"python train_gmm.py {selected_mfcc} {cluster_num} {gmm_model}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("ERROR: GMM training failed")
        return False

    # Step 2: Extract GMM features
    print("\n" + "=" * 70)
    print("STEP 2: Extracting GMM features")
    print("=" * 70)

    soft_flag = "--use_soft" if use_soft else ""
    cmd = f"python get_gmm_bof.py {gmm_model} {cluster_num} data/videos.name.lst --mfcc_path data/mfcc/ --output_path {features_dir}/ {soft_flag}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("ERROR: Feature extraction failed")
        return False

    # Step 3: Train classifiers
    print("\n" + "=" * 70)
    print("STEP 3: Training classifiers")
    print("=" * 70)

    classifiers = ['svm', 'lr', 'mlp']
    train_scripts = {
        'svm': 'train_svm_multiclass.py',
        'lr': 'train_LR.py',
        'mlp': 'train_mlp.py'
    }

    for clf in classifiers:
        model_file = f'{models_dir}/gmm-{cluster_num}-{exp_suffix}.{clf}.model'
        cmd = f"python {train_scripts[clf]} {features_dir}/ {cluster_num} {labels_dir}/train.csv {model_file}"
        print(f"\n  Training {clf.upper()}...")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  ERROR: {clf.upper()} training failed")

    # Step 4: Test
    print("\n" + "=" * 70)
    print("STEP 4: Testing")
    print("=" * 70)

    test_scripts = {
        'svm': 'test_svm_multiclass.py',
        'lr': 'test_LR.py',
        'mlp': 'test_mlp.py'
    }

    # Test on validation
    print("\n  Testing on validation set...")
    for clf in classifiers:
        model_file = f'{models_dir}/gmm-{cluster_num}-{exp_suffix}.{clf}.model'
        output_file = f'{exp_dir}/val_{clf}.csv'
        cmd = f"python {test_scripts[clf]} {model_file} {features_dir}/ {cluster_num} {labels_dir}/val.csv {output_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True)

    # Test on Kaggle test set
    print("  Testing on test set...")
    for clf in classifiers:
        model_file = f'{models_dir}/gmm-{cluster_num}-{exp_suffix}.{clf}.model'
        output_file = f'{exp_dir}/test_{clf}.csv'
        cmd = f"python {test_scripts[clf]} {model_file} {features_dir}/ {cluster_num} {labels_dir}/test_for_student.label {output_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True)

    # Evaluate
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    import pandas as pd

    val_gt = pd.read_csv(f'{labels_dir}/val.csv')
    gt_dict = dict(zip(val_gt['Id'], val_gt['Category']))

    results = {}
    for clf in classifiers:
        pred_file = f'{exp_dir}/val_{clf}.csv'
        if not os.path.exists(pred_file):
            continue

        pred_df = pd.read_csv(pred_file)
        correct = sum(1 for _, row in pred_df.iterrows()
                     if row['Id'] in gt_dict and gt_dict[row['Id']] == row['Category'])
        total = len([1 for _, row in pred_df.iterrows() if row['Id'] in gt_dict])
        accuracy = correct / total if total > 0 else 0
        results[clf] = accuracy

        print(f"{clf.upper():6s}: {accuracy*100:6.2f}% ({correct}/{total})")

    # Try ensemble
    print("\n" + "=" * 70)
    print("ENSEMBLE (Voting)")
    print("=" * 70)

    val_preds = [f'{exp_dir}/val_{clf}.csv' for clf in classifiers]
    ensemble_val = f'{exp_dir}/val_ensemble.csv'

    cmd = f"python ensemble.py {' '.join(val_preds)} --output {ensemble_val} --eval {labels_dir}/val.csv"
    subprocess.run(cmd, shell=True)

    # Create ensemble for test set
    test_preds = [f'{exp_dir}/test_{clf}.csv' for clf in classifiers]
    ensemble_test = f'{exp_dir}/test_ensemble.csv'
    cmd = f"python ensemble.py {' '.join(test_preds)} --output {ensemble_test}"
    subprocess.run(cmd, shell=True)

    print("\n" + "=" * 70)
    print(f"✓ COMPLETE! Results in: {exp_dir}")
    print("=" * 70)

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GMM baseline')
    parser.add_argument('--clusters', type=int, default=50,
                       help='Number of GMM components (default: 50)')
    parser.add_argument('--hard', action='store_true',
                       help='Use hard assignments instead of soft (default: soft)')

    args = parser.parse_args()

    success = run_gmm_pipeline(
        cluster_num=args.clusters,
        use_soft=not args.hard
    )

    sys.exit(0 if success else 1)
