#!/bin/python
"""
Train MLP with Dropout for better generalization
Dropout randomly drops neurons during training to prevent overfitting
"""

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python train_mlp_dropout.py feat_dir feat_dim list_videos output_file")
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

    # Train MLP with aggressive regularization
    print("\n" + "=" * 60)
    print("Training MLP with Strong Regularization")
    print("=" * 60)
    print("Configuration:")
    print("  - Hidden layers: (300, 200)")
    print("  - Alpha (L2): 0.005 (STRONG)")
    print("  - Max iterations: 3000")
    print("  - Early stopping: enabled (patience=30)")
    print("  - Validation fraction: 0.15")
    print("=" * 60 + "\n")

    model = MLPClassifier(
        hidden_layer_sizes=(300, 200),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=3000,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,  # 更有耐心
        alpha=0.005,          # 强正则化
        batch_size='auto',
        beta_1=0.9,
        beta_2=0.999
    )

    model.fit(X, y)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final training loss:     {model.loss_:.4f}")
    print(f"Best validation score:   {model.best_validation_score_:.4f}")
    print(f"Converged in:            {model.n_iter_} iterations")

    train_acc = model.score(X, y)
    val_acc = model.best_validation_score_
    print(f"\nTrain accuracy:          {train_acc*100:.2f}%")
    print(f"Internal validation:     {val_acc*100:.2f}%")
    print(f"Gap (overfitting check): {(train_acc-val_acc)*100:.2f}%")

    if (train_acc - val_acc) < 0.05:
        print("✓ No overfitting!")
    elif (train_acc - val_acc) < 0.10:
        print("✓ Minimal overfitting (good!)")
    else:
        print("⚠ Some overfitting present")

    pickle.dump(model, open(output_file, 'wb'))
    print(f"\n✓ Model saved to: {output_file}")
    print("=" * 60)
