#!/usr/bin/env python3
"""
Hyperparameter search for classifiers
Tests different configurations and reports best results
"""

import os
# Set environment variables BEFORE importing numpy/sklearn
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import subprocess
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from datetime import datetime
import json

def load_features_and_labels(bof_dir, label_file, cluster_num):
    """Load BoF features and labels"""
    df_labels = pd.read_csv(label_file)
    video_label_dict = dict(zip(df_labels['Id'], df_labels['Category']))

    X_list = []
    y_list = []

    for video_id, label in video_label_dict.items():
        feat_path = os.path.join(bof_dir, f"{video_id}.csv")
        if os.path.exists(feat_path):
            feat = np.genfromtxt(feat_path, delimiter=";", dtype='float')
            X_list.append(feat)
            y_list.append(label)

    return np.array(X_list), np.array(y_list)

def search_svm(X_train, y_train, X_val, y_val):
    """Search SVM hyperparameters"""
    print("\n" + "=" * 70)
    print("SVM HYPERPARAMETER SEARCH")
    print("=" * 70)

    configs = [
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100.0, 'gamma': 'scale'},
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'linear', 'C': 10.0},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config}")
        try:
            model = SVC(**config, random_state=42)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            train_acc = model.score(X_train, y_train)

            results.append({
                'config': config,
                'val_acc': val_acc,
                'train_acc': train_acc
            })

            print(f"  Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Sort by validation accuracy
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    print("\n" + "-" * 70)
    print("TOP 3 CONFIGURATIONS:")
    print("-" * 70)
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. Val: {r['val_acc']*100:.2f}% | Config: {r['config']}")

    return results[0]

def search_lr(X_train, y_train, X_val, y_val):
    """Search Logistic Regression hyperparameters"""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION HYPERPARAMETER SEARCH")
    print("=" * 70)

    configs = [
        {'C': 0.01, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 0.1, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 10.0, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 100.0, 'solver': 'lbfgs', 'max_iter': 1000},
        {'C': 1.0, 'solver': 'saga', 'max_iter': 1000},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config}")
        try:
            model = LogisticRegression(**config, multi_class='ovr', random_state=42)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            train_acc = model.score(X_train, y_train)

            results.append({
                'config': config,
                'val_acc': val_acc,
                'train_acc': train_acc
            })

            print(f"  Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")

    results.sort(key=lambda x: x['val_acc'], reverse=True)

    print("\n" + "-" * 70)
    print("TOP 3 CONFIGURATIONS:")
    print("-" * 70)
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. Val: {r['val_acc']*100:.2f}% | Config: {r['config']}")

    return results[0]

def search_mlp(X_train, y_train, X_val, y_val):
    """Search MLP hyperparameters"""
    print("\n" + "=" * 70)
    print("MLP HYPERPARAMETER SEARCH")
    print("=" * 70)

    configs = [
        {'hidden_layer_sizes': (50,), 'activation': 'relu', 'alpha': 0.0001},
        {'hidden_layer_sizes': (100,), 'activation': 'relu', 'alpha': 0.0001},
        {'hidden_layer_sizes': (200,), 'activation': 'relu', 'alpha': 0.0001},
        {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'alpha': 0.0001},
        {'hidden_layer_sizes': (200, 100), 'activation': 'relu', 'alpha': 0.0001},
        {'hidden_layer_sizes': (100,), 'activation': 'relu', 'alpha': 0.001},
        {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'alpha': 0.0001},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config}")
        try:
            model = MLPClassifier(**config, solver='adam', max_iter=500,
                                 random_state=42, verbose=False)
            model.fit(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            train_acc = model.score(X_train, y_train)

            results.append({
                'config': config,
                'val_acc': val_acc,
                'train_acc': train_acc
            })

            print(f"  Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")

    results.sort(key=lambda x: x['val_acc'], reverse=True)

    print("\n" + "-" * 70)
    print("TOP 3 CONFIGURATIONS:")
    print("-" * 70)
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. Val: {r['val_acc']*100:.2f}% | Config: {r['config']}")

    return results[0]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--bof_dir', default='features/bof',
                       help='Directory with BoF features')
    parser.add_argument('--cluster_num', type=int, default=50,
                       help='Number of clusters')
    parser.add_argument('--classifiers', nargs='+',
                       choices=['svm', 'lr', 'mlp', 'all'],
                       default=['all'],
                       help='Classifiers to tune')

    args = parser.parse_args()

    work_dir = '/home/YueChang/phd_ws/AIAA5032/homework/hw1'
    os.chdir(work_dir)

    # Load data
    print("Loading features...")
    X_train, y_train = load_features_and_labels(
        args.bof_dir, 'labels/train.csv', args.cluster_num
    )
    X_val, y_val = load_features_and_labels(
        args.bof_dir, 'labels/val.csv', args.cluster_num
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Run search
    classifiers = args.classifiers
    if 'all' in classifiers:
        classifiers = ['svm', 'lr', 'mlp']

    best_configs = {}

    if 'svm' in classifiers:
        best_configs['svm'] = search_svm(X_train, y_train, X_val, y_val)

    if 'lr' in classifiers:
        best_configs['lr'] = search_lr(X_train, y_train, X_val, y_val)

    if 'mlp' in classifiers:
        best_configs['mlp'] = search_mlp(X_train, y_train, X_val, y_val)

    # Save results
    output_file = f'experiments/hyperparam_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('experiments', exist_ok=True)

    with open(output_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        serializable_configs = {}
        for clf, result in best_configs.items():
            serializable_configs[clf] = {
                'config': {k: (tuple(v) if isinstance(v, tuple) else v)
                          for k, v in result['config'].items()},
                'val_acc': float(result['val_acc']),
                'train_acc': float(result['train_acc'])
            }
        json.dump(serializable_configs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ Best configurations saved to: {output_file}")
    print("=" * 70)
