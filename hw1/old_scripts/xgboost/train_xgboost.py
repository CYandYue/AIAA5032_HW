#!/usr/bin/env python3
"""
Train XGBoost classifier on Fisher Vector features
XGBoost with built-in early stopping to prevent overfitting
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
import pickle
import argparse
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def load_features(feat_dir, label_file, feat_appendix=".csv"):
    df = pd.read_csv(label_file)
    label_dict = dict(zip(df['Id'], df['Category']))

    X_list, y_list = [], []
    for video_id, label in label_dict.items():
        feat_path = os.path.join(feat_dir, video_id + feat_appendix)
        if os.path.exists(feat_path):
            feat = np.genfromtxt(feat_path, delimiter=";", dtype="float")
            X_list.append(feat)
            y_list.append(label)

    return np.array(X_list), np.array(y_list)

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("list_videos",   help="Train label CSV")
parser.add_argument("val_videos",    help="Val label CSV for early stopping")
parser.add_argument("output_file")
parser.add_argument("--n_estimators",   type=int,   default=1000)
parser.add_argument("--max_depth",      type=int,   default=6)
parser.add_argument("--learning_rate",  type=float, default=0.1)
parser.add_argument("--subsample",      type=float, default=0.8)
parser.add_argument("--colsample",      type=float, default=0.3,
                    help="Fraction of features per tree (important for high-dim Fisher Vector)")
parser.add_argument("--early_stopping", type=int,   default=30)

if __name__ == '__main__':
    args = parser.parse_args()

    print("Loading features...")
    X_train, y_train = load_features(args.feat_dir, args.list_videos)
    X_val,   y_val   = load_features(args.feat_dir, args.val_videos)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Feature dim: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    print(f"  n_estimators:   {args.n_estimators} (with early stopping)")
    print(f"  max_depth:      {args.max_depth}")
    print(f"  learning_rate:  {args.learning_rate}")
    print(f"  subsample:      {args.subsample}  (row sampling)")
    print(f"  colsample:      {args.colsample}  (feature sampling per tree)")
    print(f"  early_stopping: {args.early_stopping} rounds")
    print("=" * 60 + "\n")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        objective='multi:softprob',
        num_class=10,
        eval_metric='mlogloss',
        early_stopping_rounds=args.early_stopping,
        random_state=42,
        n_jobs=4,
        verbosity=1,
        device='cpu',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )

    # Evaluate
    train_acc = model.score(X_train, y_train)
    val_acc   = model.score(X_val,   y_val)
    gap       = train_acc - val_acc

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Best iteration:  {model.best_iteration}")
    print(f"Train accuracy:  {train_acc*100:.2f}%")
    print(f"Val   accuracy:  {val_acc*100:.2f}%")
    print(f"Overfitting gap: {gap*100:.2f}%")

    if gap < 0.05:
        print("✓ No overfitting")
    elif gap < 0.10:
        print("✓ Mild overfitting (good)")
    else:
        print("⚠ Overfitting present")

    pickle.dump(model, open(args.output_file, 'wb'))
    print(f"\n✓ Model saved to: {args.output_file}")
