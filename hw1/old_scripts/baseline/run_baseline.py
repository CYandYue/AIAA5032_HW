#!/usr/bin/env python3
"""
Automated baseline runner for HW1
Runs complete pipeline: K-Means -> BoF -> Train/Test 3 classifiers
Records all results systematically
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
import argparse

class BaselineRunner:
    def __init__(self, cluster_num=50, data_dir='/home/YueChang/phd_ws/AIAA5032/data/hkustgz-aiaa-5032-hw-1-spring-2025'):
        self.cluster_num = cluster_num
        self.data_dir = data_dir
        self.work_dir = '/home/YueChang/phd_ws/AIAA5032/homework/hw1'
        self.results = {}
        self.start_time = None

        # Setup shared directories
        self.temp_dir = os.path.join(self.work_dir, 'temp')
        self.data_symlink_dir = os.path.join(self.work_dir, 'data')
        self.labels_dir = os.path.join(self.work_dir, 'labels')

        # Create experiments directory
        experiments_dir = os.path.join(self.work_dir, 'experiments')
        os.makedirs(experiments_dir, exist_ok=True)

        # Setup timestamp and experiment directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f"kmeans_k{cluster_num}_{self.timestamp}"
        self.exp_dir = os.path.join(experiments_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Models and features saved in experiment directory
        self.models_dir = os.path.join(self.exp_dir, 'models')
        self.features_dir = os.path.join(self.exp_dir, 'features')
        self.bof_dir = self.features_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def log(self, message, level='INFO'):
        """Print and log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        # Append to log file
        log_file = os.path.join(self.exp_dir, 'run.log')
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

    def run_command(self, cmd, description):
        """Run shell command and record time"""
        self.log(f"Running: {description}")
        self.log(f"Command: {cmd}", level='DEBUG')

        start = time.time()
        try:
            # Set environment variables to avoid OpenBLAS issues
            env = os.environ.copy()
            env['OPENBLAS_NUM_THREADS'] = '4'
            env['MKL_NUM_THREADS'] = '4'
            env['OMP_NUM_THREADS'] = '4'

            result = subprocess.run(cmd, shell=True, check=True,
                                  capture_output=True, text=True,
                                  cwd=self.work_dir, env=env)
            duration = time.time() - start
            self.log(f"✓ Completed in {duration:.2f}s")

            return {
                'success': True,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.CalledProcessError as e:
            duration = time.time() - start
            self.log(f"✗ Failed after {duration:.2f}s", level='ERROR')
            self.log(f"Error: {e.stderr}", level='ERROR')

            return {
                'success': False,
                'duration': duration,
                'error': str(e),
                'stderr': e.stderr
            }

    def setup_environment(self):
        """Setup directories and symlinks"""
        self.log("=" * 60)
        self.log("STEP 0: Setting up environment")
        self.log("=" * 60)

        # Create shared directory structure (not experiment-specific)
        dirs_to_create = [
            ('temp', 'Temporary files'),
            ('data', 'Data symlinks'),
            ('labels', 'Label files'),
        ]

        for dir_path, description in dirs_to_create:
            full_path = os.path.join(self.work_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            self.log(f"✓ {description:<30} {dir_path}/")

        # Create symlinks to data files
        mfcc_src = os.path.join(self.data_dir, 'mfcc.tgz/mfcc')
        mfcc_dst = os.path.join(self.data_symlink_dir, 'mfcc')

        if not os.path.exists(mfcc_dst):
            os.symlink(mfcc_src, mfcc_dst)
            self.log(f"✓ Symlink: data/mfcc -> original data")

        # Copy label files if needed
        for file in ['trainval.csv', 'test_for_student.label', 'videos.name.lst']:
            src = os.path.join(self.data_dir, file)
            dst_dir = self.labels_dir if file.endswith('.csv') or file.endswith('.label') else self.data_symlink_dir
            dst = os.path.join(dst_dir, file)
            if not os.path.exists(dst):
                subprocess.run(f"cp {src} {dst}", shell=True)
                self.log(f"✓ Copied: {file}")

        self.log("\nDirectory structure:")
        self.log(f"  temp/                   - Temporary files (shared)")
        self.log(f"  data/                   - Data symlinks (shared)")
        self.log(f"  labels/                 - Label files (shared)")
        self.log(f"  experiments/{self.exp_name}/")
        self.log(f"    ├─ models/            - Trained models")
        self.log(f"    ├─ features/          - Extracted features")
        self.log(f"    ├─ val_*.csv          - Validation predictions")
        self.log(f"    ├─ test_*.csv         - Test predictions")
        self.log(f"    └─ SUMMARY.txt        - Results summary")
        self.log("Environment setup complete!\n")

    def step1_select_frames(self):
        """Step 1: Select frames for K-Means"""
        self.log("=" * 60)
        self.log("STEP 1: Selecting frames for K-Means (20% sampling)")
        self.log("=" * 60)

        selected_file = os.path.join(self.temp_dir, 'selected.mfcc.csv')
        cmd = f"python select_frames.py labels/train.csv 0.2 {selected_file} --mfcc_path data/mfcc/"
        result = self.run_command(cmd, "Select frames")
        self.results['step1_select_frames'] = result

        return result['success']

    def step2_train_kmeans(self):
        """Step 2: Train K-Means"""
        self.log("=" * 60)
        self.log(f"STEP 2: Training K-Means ({self.cluster_num} clusters)")
        self.log("=" * 60)

        selected_file = os.path.join(self.temp_dir, 'selected.mfcc.csv')
        model_file = os.path.join(self.models_dir, f"kmeans.{self.cluster_num}.model")
        cmd = f"python train_kmeans.py {selected_file} {self.cluster_num} {model_file}"
        result = self.run_command(cmd, "Train K-Means")
        self.results['step2_train_kmeans'] = result

        return result['success']

    def step3_extract_bof(self):
        """Step 3: Extract Bag-of-Features"""
        self.log("=" * 60)
        self.log("STEP 3: Extracting Bag-of-Features")
        self.log("=" * 60)

        model_file = os.path.join(self.models_dir, f"kmeans.{self.cluster_num}.model")
        cmd = f"python get_bof.py {model_file} {self.cluster_num} data/videos.name.lst --mfcc_path data/mfcc/ --output_path {self.bof_dir}/"
        result = self.run_command(cmd, "Extract BoF features")
        self.results['step3_extract_bof'] = result

        return result['success']

    def step4_train_classifiers(self):
        """Step 4: Train all classifiers"""
        self.log("=" * 60)
        self.log("STEP 4: Training classifiers on train.csv")
        self.log("=" * 60)

        classifiers = {
            'svm': f"python train_svm_multiclass.py {self.bof_dir}/ {self.cluster_num} labels/train.csv {self.models_dir}/mfcc-{self.cluster_num}.svm.model",
            'lr': f"python train_LR.py {self.bof_dir}/ {self.cluster_num} labels/train.csv {self.models_dir}/mfcc-{self.cluster_num}.lr.model",
            'mlp': f"python train_mlp.py {self.bof_dir}/ {self.cluster_num} labels/train.csv {self.models_dir}/mfcc-{self.cluster_num}.mlp.model"
        }

        self.results['step4_train'] = {}
        all_success = True

        for clf_name, cmd in classifiers.items():
            self.log(f"\nTraining {clf_name.upper()}...")
            result = self.run_command(cmd, f"Train {clf_name.upper()}")
            self.results['step4_train'][clf_name] = result
            all_success = all_success and result['success']

        return all_success

    def step5_test_validation(self):
        """Step 5: Test on validation set"""
        self.log("=" * 60)
        self.log("STEP 5: Testing on validation set (val.csv)")
        self.log("=" * 60)

        classifiers = {
            'svm': f"python test_svm_multiclass.py {self.models_dir}/mfcc-{self.cluster_num}.svm.model {self.bof_dir}/ {self.cluster_num} labels/val.csv {self.exp_dir}/val_svm.csv",
            'lr': f"python test_LR.py {self.models_dir}/mfcc-{self.cluster_num}.lr.model {self.bof_dir}/ {self.cluster_num} labels/val.csv {self.exp_dir}/val_lr.csv",
            'mlp': f"python test_mlp.py {self.models_dir}/mfcc-{self.cluster_num}.mlp.model {self.bof_dir}/ {self.cluster_num} labels/val.csv {self.exp_dir}/val_mlp.csv"
        }

        self.results['step5_test_val'] = {}
        all_success = True

        for clf_name, cmd in classifiers.items():
            self.log(f"\nTesting {clf_name.upper()} on validation set...")
            result = self.run_command(cmd, f"Test {clf_name.upper()} (val)")
            self.results['step5_test_val'][clf_name] = result
            all_success = all_success and result['success']

        return all_success

    def step6_test_kaggle(self):
        """Step 6: Test on Kaggle test set"""
        self.log("=" * 60)
        self.log("STEP 6: Testing on Kaggle test set")
        self.log("=" * 60)

        classifiers = {
            'svm': f"python test_svm_multiclass.py {self.models_dir}/mfcc-{self.cluster_num}.svm.model {self.bof_dir}/ {self.cluster_num} labels/test_for_student.label {self.exp_dir}/test_svm.csv",
            'lr': f"python test_LR.py {self.models_dir}/mfcc-{self.cluster_num}.lr.model {self.bof_dir}/ {self.cluster_num} labels/test_for_student.label {self.exp_dir}/test_lr.csv",
            'mlp': f"python test_mlp.py {self.models_dir}/mfcc-{self.cluster_num}.mlp.model {self.bof_dir}/ {self.cluster_num} labels/test_for_student.label {self.exp_dir}/test_mlp.csv"
        }

        self.results['step6_test_kaggle'] = {}
        all_success = True

        for clf_name, cmd in classifiers.items():
            self.log(f"\nTesting {clf_name.upper()} on Kaggle test set...")
            result = self.run_command(cmd, f"Test {clf_name.upper()} (Kaggle)")
            self.results['step6_test_kaggle'][clf_name] = result
            all_success = all_success and result['success']

        return all_success

    def evaluate_validation(self):
        """Evaluate accuracy on validation set"""
        self.log("=" * 60)
        self.log("EVALUATION: Computing validation accuracy")
        self.log("=" * 60)

        import pandas as pd
        import numpy as np

        # Load ground truth
        val_gt_path = os.path.join(self.labels_dir, 'val.csv')
        val_gt = pd.read_csv(val_gt_path)
        gt_dict = dict(zip(val_gt['Id'], val_gt['Category']))

        accuracies = {}

        for clf_name in ['svm', 'lr', 'mlp']:
            pred_file = os.path.join(self.exp_dir, f'val_{clf_name}.csv')

            if not os.path.exists(pred_file):
                self.log(f"Warning: {pred_file} not found", level='WARN')
                continue

            pred_df = pd.read_csv(pred_file)

            # Calculate accuracy
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
            accuracies[clf_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

            self.log(f"{clf_name.upper()}: {accuracy*100:.2f}% ({correct}/{total})")

        self.results['validation_accuracy'] = accuracies

        return accuracies

    def save_results(self):
        """Save all results to JSON"""
        self.log("\n" + "=" * 60)
        self.log("Saving results...")
        self.log("=" * 60)

        # Add metadata
        self.results['metadata'] = {
            'experiment_name': self.exp_name,
            'cluster_num': self.cluster_num,
            'timestamp': self.timestamp,
            'total_duration': time.time() - self.start_time if self.start_time else 0
        }

        # Save to JSON
        json_file = os.path.join(self.exp_dir, 'results.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.log(f"Results saved to: {json_file}")

        # Generate summary
        summary_file = os.path.join(self.exp_dir, 'SUMMARY.txt')
        self.generate_summary(summary_file)
        self.log(f"Summary saved to: {summary_file}")

        # Print summary
        with open(summary_file, 'r') as f:
            print("\n" + f.read())

    def generate_summary(self, output_file):
        """Generate human-readable summary"""
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {self.exp_name}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cluster Number: {self.cluster_num}\n")
            f.write(f"Total Duration: {self.results['metadata']['total_duration']:.2f}s\n\n")

            f.write("-" * 70 + "\n")
            f.write("VALIDATION SET PERFORMANCE\n")
            f.write("-" * 70 + "\n")

            if 'validation_accuracy' in self.results:
                for clf, metrics in sorted(self.results['validation_accuracy'].items()):
                    f.write(f"{clf.upper():10s}: {metrics['accuracy']*100:6.2f}% ")
                    f.write(f"({metrics['correct']}/{metrics['total']})\n")
            else:
                f.write("No validation results available\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Validation predictions:\n")
            f.write(f"  - val_svm.csv\n")
            f.write(f"  - val_lr.csv\n")
            f.write(f"  - val_mlp.csv\n\n")

            f.write(f"Kaggle test predictions (ready for submission):\n")
            f.write(f"  - test_svm.csv\n")
            f.write(f"  - test_lr.csv\n")
            f.write(f"  - test_mlp.csv\n\n")

            f.write("-" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")

            if 'validation_accuracy' in self.results:
                best_clf = max(self.results['validation_accuracy'].items(),
                             key=lambda x: x[1]['accuracy'])
                f.write(f"Best performer: {best_clf[0].upper()} ")
                f.write(f"({best_clf[1]['accuracy']*100:.2f}%)\n")
                f.write(f"Submit: {self.exp_dir}/test_{best_clf[0]}.csv to Kaggle\n\n")

            f.write("=" * 70 + "\n")

    def run_full_pipeline(self):
        """Run complete baseline pipeline"""
        self.start_time = time.time()
        self.log("=" * 70)
        self.log(f"STARTING BASELINE EXPERIMENT: {self.exp_name}")
        self.log("=" * 70 + "\n")

        try:
            # Step 0: Setup
            self.setup_environment()

            # Step 1: Select frames
            if not self.step1_select_frames():
                self.log("Failed at step 1", level='ERROR')
                return False

            # Step 2: Train K-Means
            if not self.step2_train_kmeans():
                self.log("Failed at step 2", level='ERROR')
                return False

            # Step 3: Extract BoF
            if not self.step3_extract_bof():
                self.log("Failed at step 3", level='ERROR')
                return False

            # Step 4: Train classifiers
            if not self.step4_train_classifiers():
                self.log("Failed at step 4", level='ERROR')
                return False

            # Step 5: Test on validation
            if not self.step5_test_validation():
                self.log("Failed at step 5", level='ERROR')
                return False

            # Step 6: Test on Kaggle test set
            if not self.step6_test_kaggle():
                self.log("Failed at step 6", level='ERROR')
                return False

            # Evaluation
            self.evaluate_validation()

            # Save results
            self.save_results()

            self.log("\n" + "=" * 70)
            self.log("✓ EXPERIMENT COMPLETED SUCCESSFULLY!")
            self.log("=" * 70)

            return True

        except Exception as e:
            self.log(f"Unexpected error: {str(e)}", level='ERROR')
            import traceback
            self.log(traceback.format_exc(), level='ERROR')
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline experiments for HW1')
    parser.add_argument('--clusters', type=int, default=50,
                       help='Number of K-Means clusters (default: 50)')
    parser.add_argument('--data_dir', default='/home/YueChang/phd_ws/AIAA5032/data/hkustgz-aiaa-5032-hw-1-spring-2025',
                       help='Path to data directory')

    args = parser.parse_args()

    runner = BaselineRunner(cluster_num=args.clusters, data_dir=args.data_dir)
    success = runner.run_full_pipeline()

    sys.exit(0 if success else 1)
