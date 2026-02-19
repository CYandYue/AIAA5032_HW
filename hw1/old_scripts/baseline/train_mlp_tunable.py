#!/bin/python
"""
Train MLP with different regularization levels
Use this to find best alpha (L2 penalty)
"""

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--alpha", type=float, default=0.001, help="L2 penalty (default: 0.001)")
parser.add_argument("--hidden", type=str, default="200,100", help="Hidden layer sizes (default: 200,100)")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_layers = tuple(map(int, args.hidden.split(',')))

    # 1. read all features in one array.
    fread = open(args.list_videos, "r")
    feat_list = []
    label_list = []
    df_videos_label = {}
    for line in open(args.list_videos).readlines()[1:]:
        video_id, category = line.strip().split(",")
        df_videos_label[video_id] = category

    for line in fread.readlines()[1:]:
        video_id = line.strip().split(",")[0]
        feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
        if os.path.exists(feat_filepath):
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
            label_list.append(int(df_videos_label[video_id]))

    print("number of samples: %s" % len(feat_list))
    y = np.array(label_list)
    X = np.array(feat_list)

    # Train MLP classifier with custom parameters
    print("Training MLP classifier...")
    print("Configuration:")
    print(f"  - Hidden layers: {hidden_layers}")
    print(f"  - Alpha (L2 penalty): {args.alpha}")
    print(f"  - Max iterations: 2000")
    print(f"  - Early stopping: enabled")
    print(f"  - Validation fraction: 0.15")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.15,  # 稍微多一点验证数据
        n_iter_no_change=25,       # 更耐心一点
        alpha=args.alpha           # 可调的正则化
    )
    model.fit(X, y)

    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    print(f"Final training loss: {model.loss_:.4f}")
    print(f"Best validation score: {model.best_validation_score_:.4f}")
    print(f"Converged in {model.n_iter_} iterations")

    # 检查过拟合
    train_score = model.score(X, y)
    val_score = model.best_validation_score_
    gap = train_score - val_score

    print(f"\nTrain accuracy: {train_score*100:.2f}%")
    print(f"Internal validation: {val_score*100:.2f}%")
    print(f"Gap: {gap*100:.2f}%")

    if gap < 0.05:
        print("✓ No overfitting detected")
    elif gap < 0.15:
        print("⚠ Mild overfitting (acceptable)")
    else:
        print("✗ Significant overfitting! Consider increasing alpha")

    # save trained MLP in output_file
    pickle.dump(model, open(args.output_file, 'wb'))
    print(f'\n✓ MLP model saved to {args.output_file}')
