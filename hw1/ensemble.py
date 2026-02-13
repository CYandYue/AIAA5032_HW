#!/usr/bin/env python3
"""
Model Ensemble: Combine predictions from multiple classifiers
Supports voting, weighted averaging, and stacking
"""

import pandas as pd
import numpy as np
import argparse
from collections import Counter

def ensemble_voting(pred_files, output_file):
    """Simple majority voting"""
    print(f"Ensemble method: Voting")
    print(f"Combining {len(pred_files)} predictions")

    # Load all predictions
    predictions = []
    for f in pred_files:
        df = pd.read_csv(f)
        predictions.append(df)
        print(f"  Loaded: {f} ({len(df)} samples)")

    # Check all have same video IDs
    video_ids = predictions[0]['Id'].values
    for df in predictions[1:]:
        assert all(df['Id'] == video_ids), "Video IDs mismatch!"

    # Majority voting
    final_predictions = []
    for i in range(len(video_ids)):
        votes = [df.iloc[i]['Category'] for df in predictions]
        # Most common prediction
        majority_vote = Counter(votes).most_common(1)[0][0]
        final_predictions.append(majority_vote)

    # Save result
    result_df = pd.DataFrame({
        'Id': video_ids,
        'Category': final_predictions
    })
    result_df.to_csv(output_file, index=False)
    print(f"✓ Ensemble predictions saved to: {output_file}")

def ensemble_weighted(pred_files, weights, output_file, use_scores=False):
    """Weighted averaging (requires decision scores)"""
    print(f"Ensemble method: Weighted averaging")
    print(f"Weights: {weights}")

    if not use_scores:
        print("Warning: Weighted ensemble works best with decision scores")
        print("Falling back to weighted voting...")

    # For simplicity, implement weighted voting
    predictions = []
    for f in pred_files:
        df = pd.read_csv(f)
        predictions.append(df)

    video_ids = predictions[0]['Id'].values

    final_predictions = []
    for i in range(len(video_ids)):
        votes = [df.iloc[i]['Category'] for df in predictions]
        # Weighted voting
        vote_scores = {}
        for vote, weight in zip(votes, weights):
            vote_scores[vote] = vote_scores.get(vote, 0) + weight

        best_vote = max(vote_scores.items(), key=lambda x: x[1])[0]
        final_predictions.append(best_vote)

    result_df = pd.DataFrame({
        'Id': video_ids,
        'Category': final_predictions
    })
    result_df.to_csv(output_file, index=False)
    print(f"✓ Ensemble predictions saved to: {output_file}")

def evaluate_ensemble(ensemble_file, ground_truth_file):
    """Evaluate ensemble predictions"""
    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(ensemble_file)

    gt_dict = dict(zip(gt_df['Id'], gt_df['Category']))

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
    print(f"\nEnsemble Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble multiple model predictions')
    parser.add_argument('pred_files', nargs='+', help='Prediction CSV files to combine')
    parser.add_argument('--output', required=True, help='Output ensemble prediction file')
    parser.add_argument('--method', default='voting', choices=['voting', 'weighted'],
                       help='Ensemble method (default: voting)')
    parser.add_argument('--weights', type=float, nargs='+',
                       help='Weights for each model (for weighted method)')
    parser.add_argument('--eval', help='Ground truth file for evaluation (optional)')

    args = parser.parse_args()

    # Validate inputs
    if args.method == 'weighted':
        if args.weights is None:
            print("Error: --weights required for weighted ensemble")
            exit(1)
        if len(args.weights) != len(args.pred_files):
            print("Error: Number of weights must match number of prediction files")
            exit(1)

    # Run ensemble
    if args.method == 'voting':
        ensemble_voting(args.pred_files, args.output)
    elif args.method == 'weighted':
        ensemble_weighted(args.pred_files, args.weights, args.output)

    # Evaluate if ground truth provided
    if args.eval:
        evaluate_ensemble(args.output, args.eval)
