#!/bin/python
"""
Train MLP without early stopping - let it run full iterations
"""

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python train_mlp_no_early_stop.py feat_dir feat_dim list_videos output_file")
        sys.exit(1)

    feat_dir = sys.argv[1]
    feat_dim = int(sys.argv[2])
    list_videos = sys.argv[3]
    output_file = sys.argv[4]

    # Load features
    fread = open(list_videos, "r")
    feat_list = []
    label_list = []
    df_videos_label = {}

    for line in open(list_videos).readlines()[1:]:
        video_id, category = line.strip().split(",")
        df_videos_label[video_id] = category

    for line in fread.readlines()[1:]:
        video_id = line.strip().split(",")[0]
        feat_filepath = os.path.join(feat_dir, video_id + ".csv")
        if os.path.exists(feat_filepath):
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
            label_list.append(int(df_videos_label[video_id]))

    print(f"Number of samples: {len(feat_list)}")
    y = np.array(label_list)
    X = np.array(feat_list)

    # Train MLP WITHOUT early stopping
    print("\n" + "=" * 60)
    print("Training MLP WITHOUT Early Stopping")
    print("=" * 60)
    print("Configuration:")
    print("  - Hidden layers: (300, 200)")
    print("  - Alpha (L2): 0.005")
    print("  - Max iterations: 1000 (will run ALL iterations)")
    print("  - Early stopping: DISABLED")
    print("  - Learning rate: 0.001")
    print("=" * 60 + "\n")

    model = MLPClassifier(
        hidden_layer_sizes=(300, 200),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,           # 运行完整1000轮
        random_state=42,
        verbose=True,
        early_stopping=False,    # 关闭early stopping
        alpha=0.005,             # 强正则化防止过拟合
        batch_size='auto',
    )

    print("Training will run for FULL 1000 iterations...")
    print("Watch for: loss should keep decreasing\n")

    model.fit(X, y)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final training loss:     {model.loss_:.4f}")
    print(f"Total iterations:        {model.n_iter_}")

    train_acc = model.score(X, y)
    print(f"Train accuracy:          {train_acc*100:.2f}%")

    print("\n⚠️  Note: Without early stopping, check validation manually")
    print("    to detect if overfitting occurred in later iterations.")

    pickle.dump(model, open(output_file, 'wb'))
    print(f"\n✓ Model saved to: {output_file}")
    print("=" * 60)
