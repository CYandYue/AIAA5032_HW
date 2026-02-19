#!/usr/bin/env python3
"""
MLP hyperparameter grid search on Fisher Vector features.
Tries different hidden layer sizes and regularization strengths.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pickle
import argparse
from datetime import datetime
from sklearn.neural_network import MLPClassifier


def load_features(list_file, feat_dir, feat_dim):
    video_ids, labels, feats = [], [], []
    for line in open(list_file).readlines():
        line = line.strip()
        if not line or line.startswith('Id,') or line == 'Id':
            continue
        if ',' in line:
            parts = line.split(',')
            video_id, label = parts[0], int(parts[1])
        else:
            video_id, label = os.path.splitext(line)[0], -1
        feat_path = os.path.join(feat_dir, video_id + '.csv')
        feat = np.genfromtxt(feat_path, delimiter=';', dtype='float') \
               if os.path.exists(feat_path) else np.zeros(feat_dim)
        video_ids.append(video_id)
        labels.append(label)
        feats.append(feat)
    return video_ids, np.array(feats), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_dir')
    parser.add_argument('train_csv')
    parser.add_argument('val_csv')
    parser.add_argument('test_label')
    parser.add_argument('--feat_dim', type=int, default=15600)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/mlp_search_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    log_f = open(os.path.join(out_dir, 'train.log'), 'w')

    def log(msg=''):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f'Output : {out_dir}')
    log(f'\nHyperparameters:')
    for k, v in vars(args).items():
        log(f'  {k}: {v}')

    log('Loading features...')
    tr_ids,  X_tr,  y_tr  = load_features(args.train_csv,  args.feat_dir, args.feat_dim)
    val_ids, X_val, y_val = load_features(args.val_csv,    args.feat_dir, args.feat_dim)
    test_ids, X_test, _   = load_features(args.test_label, args.feat_dir, args.feat_dim)
    log(f'  Train {len(tr_ids)}, Val {len(val_ids)}, Test {len(test_ids)}, Dim {X_tr.shape[1]}')

    # ── Grid ──────────────────────────────────────────────────────
    hidden_configs = [
        (200, 100),        # baseline
        (512, 256),        # wider
        (1024, 512),       # much wider
        (256, 128, 64),    # deeper
        (512, 256, 128),   # wider + deeper
    ]
    alphas = [0.0001, 0.001, 0.01]

    results = []

    total = len(hidden_configs) * len(alphas)
    idx = 0
    for hidden in hidden_configs:
        for alpha in alphas:
            idx += 1
            tag = f'h{"_".join(str(h) for h in hidden)}_a{alpha}'
            log(f'\n[{idx}/{total}] hidden={hidden}  alpha={alpha}')

            mlp = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation='relu',
                solver='adam',
                max_iter=2000,
                random_state=42,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                alpha=alpha,
            )
            mlp.fit(X_tr, y_tr)

            train_acc = mlp.score(X_tr, y_tr)
            val_acc   = mlp.score(X_val, y_val)
            n_iter    = mlp.n_iter_
            log(f'  iters={n_iter:4d}  train={train_acc*100:.2f}%  val={val_acc*100:.2f}%  gap={( train_acc - val_acc)*100:.2f}%')

            results.append({
                'hidden': hidden,
                'alpha': alpha,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'gap': train_acc - val_acc,
                'n_iter': n_iter,
                'model': mlp,
                'tag': tag,
            })

    # ── Summary ───────────────────────────────────────────────────
    results.sort(key=lambda x: -x['val_acc'])
    log(f'\n{"="*70}')
    log(f'  {"Hidden":<22} {"Alpha":>8}  {"Train":>7}  {"Val":>7}  {"Gap":>7}  {"Iters":>6}')
    log(f'{"="*70}')
    for r in results:
        marker = ' ←best' if r == results[0] else ''
        log(f'  {str(r["hidden"]):<22} {r["alpha"]:>8}  '
              f'{r["train_acc"]*100:>6.2f}%  {r["val_acc"]*100:>6.2f}%  '
              f'{r["gap"]*100:>6.2f}%  {r["n_iter"]:>6}{marker}')

    # ── Save best model + test predictions ────────────────────────
    best = results[0]
    log(f'\nBest: hidden={best["hidden"]}, alpha={best["alpha"]}, val={best["val_acc"]*100:.2f}%')

    model_path = os.path.join(out_dir, f'best_mlp_{best["tag"]}.model')
    pickle.dump(best['model'], open(model_path, 'wb'))

    test_preds = best['model'].predict(X_test)
    test_out = os.path.join(out_dir, 'test_best_mlp.csv')
    with open(test_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(test_ids, test_preds):
            f.write(f'{vid},{pred}\n')

    val_preds = best['model'].predict(X_val)
    val_out = os.path.join(out_dir, 'val_best_mlp.csv')
    with open(val_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(val_ids, val_preds):
            f.write(f'{vid},{pred}\n')

    log(f'  Model  → {model_path}')
    log(f'  Val    → {val_out}')
    log(f'  Test   → {test_out}  ← 上传 Kaggle')

    log_f.close()


if __name__ == '__main__':
    main()
