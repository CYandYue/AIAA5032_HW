#!/usr/bin/env python3
"""
Compare and visualize results from multiple baseline experiments
"""

import os
import json
import pandas as pd
from datetime import datetime
import argparse

class ResultsComparer:
    def __init__(self, results_dir='/home/YueChang/phd_ws/AIAA5032/homework/hw1/experiments'):
        self.results_dir = results_dir

    def load_all_experiments(self):
        """Load all experiment results"""
        experiments = []

        if not os.path.exists(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return experiments

        for exp_name in os.listdir(self.results_dir):
            exp_dir = os.path.join(self.results_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            results_file = os.path.join(exp_dir, 'results.json')
            if not os.path.exists(results_file):
                continue

            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    experiments.append({
                        'name': exp_name,
                        'dir': exp_dir,
                        'data': data
                    })
            except Exception as e:
                print(f"Warning: Failed to load {results_file}: {e}")

        return experiments

    def print_comparison_table(self, experiments):
        """Print comparison table of all experiments"""
        if not experiments:
            print("No experiments found!")
            return

        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON")
        print("=" * 100)

        # Build table data
        table_data = []
        for exp in experiments:
            row = {
                'Experiment': exp['name'],
                'Clusters': exp['data']['metadata']['cluster_num'],
                'Duration(s)': f"{exp['data']['metadata']['total_duration']:.1f}"
            }

            # Add validation accuracies
            if 'validation_accuracy' in exp['data']:
                for clf in ['svm', 'lr', 'mlp']:
                    if clf in exp['data']['validation_accuracy']:
                        acc = exp['data']['validation_accuracy'][clf]['accuracy']
                        row[f'{clf.upper()}_acc'] = f"{acc*100:.2f}%"
                    else:
                        row[f'{clf.upper()}_acc'] = "N/A"

            table_data.append(row)

        # Convert to DataFrame for pretty printing
        df = pd.DataFrame(table_data)

        # Sort by experiment name (which includes timestamp)
        df = df.sort_values('Experiment', ascending=False)

        print(df.to_string(index=False))
        print("\n")

    def print_best_results(self, experiments):
        """Print best performing configurations"""
        print("=" * 100)
        print("BEST RESULTS")
        print("=" * 100)

        all_results = []
        for exp in experiments:
            if 'validation_accuracy' not in exp['data']:
                continue

            for clf, metrics in exp['data']['validation_accuracy'].items():
                all_results.append({
                    'experiment': exp['name'],
                    'classifier': clf.upper(),
                    'clusters': exp['data']['metadata']['cluster_num'],
                    'accuracy': metrics['accuracy'],
                    'correct': metrics['correct'],
                    'total': metrics['total']
                })

        if not all_results:
            print("No validation results found!")
            return

        # Sort by accuracy
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)

        print("\nTop 10 configurations by validation accuracy:\n")
        print(f"{'Rank':<6} {'Classifier':<12} {'Clusters':<10} {'Accuracy':<12} {'Correct/Total':<15} {'Experiment':<30}")
        print("-" * 100)

        for i, result in enumerate(all_results[:10], 1):
            print(f"{i:<6} {result['classifier']:<12} {result['clusters']:<10} "
                  f"{result['accuracy']*100:6.2f}%      "
                  f"{result['correct']}/{result['total']:<10} {result['experiment']:<30}")

        # Print best for each classifier
        print("\n" + "-" * 100)
        print("BEST CONFIGURATION PER CLASSIFIER")
        print("-" * 100 + "\n")

        for clf in ['SVM', 'LR', 'MLP']:
            clf_results = [r for r in all_results if r['classifier'] == clf]
            if clf_results:
                best = clf_results[0]
                print(f"{clf:<6}: {best['accuracy']*100:.2f}% with {best['clusters']} clusters")
                print(f"         Experiment: {best['experiment']}")
                print(f"         File: experiments/{best['experiment']}/test_{clf.lower()}.csv")
                print()

    def generate_comparison_report(self, output_file='comparison_report.txt'):
        """Generate comprehensive comparison report"""
        experiments = self.load_all_experiments()

        if not experiments:
            print("No experiments to compare!")
            return

        output_path = os.path.join(self.results_dir, output_file)

        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"EXPERIMENT COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Total experiments: {len(experiments)}\n\n")

            # Write detailed results
            for exp in sorted(experiments, key=lambda x: x['name'], reverse=True):
                f.write("-" * 100 + "\n")
                f.write(f"Experiment: {exp['name']}\n")
                f.write(f"Clusters: {exp['data']['metadata']['cluster_num']}\n")
                f.write(f"Duration: {exp['data']['metadata']['total_duration']:.2f}s\n\n")

                if 'validation_accuracy' in exp['data']:
                    f.write("Validation Accuracy:\n")
                    for clf, metrics in sorted(exp['data']['validation_accuracy'].items()):
                        f.write(f"  {clf.upper()}: {metrics['accuracy']*100:.2f}% ")
                        f.write(f"({metrics['correct']}/{metrics['total']})\n")
                else:
                    f.write("No validation results available\n")

                f.write("\n")

        print(f"\nDetailed report saved to: {output_path}")

    def show_experiment_details(self, exp_name):
        """Show detailed information for a specific experiment"""
        exp_dir = os.path.join(self.results_dir, exp_name)
        results_file = os.path.join(exp_dir, 'results.json')

        if not os.path.exists(results_file):
            print(f"Experiment not found: {exp_name}")
            return

        with open(results_file, 'r') as f:
            data = json.load(f)

        print("\n" + "=" * 100)
        print(f"DETAILED RESULTS: {exp_name}")
        print("=" * 100 + "\n")

        print(f"Clusters: {data['metadata']['cluster_num']}")
        print(f"Total Duration: {data['metadata']['total_duration']:.2f}s")
        print(f"Timestamp: {data['metadata']['timestamp']}\n")

        # Print step-by-step durations
        print("-" * 100)
        print("STEP DURATIONS")
        print("-" * 100)

        steps = [
            ('step1_select_frames', 'Select Frames'),
            ('step2_train_kmeans', 'Train K-Means'),
            ('step3_extract_bof', 'Extract BoF'),
        ]

        for step_key, step_name in steps:
            if step_key in data and 'duration' in data[step_key]:
                duration = data[step_key]['duration']
                success = "✓" if data[step_key].get('success', False) else "✗"
                print(f"{success} {step_name:<20}: {duration:>8.2f}s")

        # Training times
        if 'step4_train' in data:
            print(f"\nTraining:")
            for clf, result in data['step4_train'].items():
                duration = result.get('duration', 0)
                success = "✓" if result.get('success', False) else "✗"
                print(f"{success}   {clf.upper():<18}: {duration:>8.2f}s")

        # Validation accuracy
        if 'validation_accuracy' in data:
            print("\n" + "-" * 100)
            print("VALIDATION ACCURACY")
            print("-" * 100)
            for clf, metrics in sorted(data['validation_accuracy'].items()):
                print(f"{clf.upper():<6}: {metrics['accuracy']*100:6.2f}% ({metrics['correct']}/{metrics['total']})")

        print("\n" + "=" * 100 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare baseline experiment results')
    parser.add_argument('--results_dir', default='/home/YueChang/phd_ws/AIAA5032/homework/hw1/experiments',
                       help='Path to results directory')
    parser.add_argument('--detail', type=str, help='Show details for specific experiment')
    parser.add_argument('--report', action='store_true', help='Generate comparison report')

    args = parser.parse_args()

    comparer = ResultsComparer(results_dir=args.results_dir)

    if args.detail:
        comparer.show_experiment_details(args.detail)
    elif args.report:
        comparer.generate_comparison_report()
    else:
        # Default: show comparison table and best results
        experiments = comparer.load_all_experiments()
        comparer.print_comparison_table(experiments)
        comparer.print_best_results(experiments)
