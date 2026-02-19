#!/usr/bin/env python3
"""
PCA + MLP pipeline on Fisher Vector features.
PCA reduces 15600-dim Fisher Vectors before MLP training,
which greatly reduces overfitting.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pickle
import argparse
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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
    parser.add_argument('feat_dir',     help='Fisher Vector feature directory')
    parser.add_argument('train_csv',    help='Train labels (Id,Category)')
    parser.add_argument('val_csv',      help='Val labels (Id,Category)')
    parser.add_argument('test_label',   help='Test video list')
    parser.add_argument('--feat_dim',   type=int,   default=15600)
    parser.add_argument('--n_pca',      type=int,   default=256,
                        help='PCA output dimensions (try 128, 256, 512)')
    parser.add_argument('--hidden',     type=int,   nargs='+', default=[256, 128])
    parser.add_argument('--alpha',      type=float, default=0.001)
    parser.add_argument('--patience',   type=int,   default=20)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/pca{args.n_pca}_mlp_{timestamp}'
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

    log(f'Output dir: {out_dir}')
    log(f'PCA dims  : {args.feat_dim} → {args.n_pca}')
    log(f'MLP arch  : {args.n_pca} → {args.hidden} → 10')

    # ── 1. Load features ──────────────────────────────────────────
    log('\nLoading features...')
    tr_ids,  X_tr,  y_tr  = load_features(args.train_csv,  args.feat_dir, args.feat_dim)
    val_ids, X_val, y_val = load_features(args.val_csv,    args.feat_dir, args.feat_dim)
    test_ids, X_test, _   = load_features(args.test_label, args.feat_dir, args.feat_dim)
    log(f'  Train: {len(tr_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}')
    log(f'  Raw feat dim: {X_tr.shape[1]}')

    # ── 2. PCA ────────────────────────────────────────────────────
    log(f'\nFitting PCA (n_components={args.n_pca})...')
    pca = PCA(n_components=args.n_pca, random_state=42)
    X_tr_pca   = pca.fit_transform(X_tr)
    X_val_pca  = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    log(f'  Explained variance: {explained*100:.1f}%')
    log(f'  Reduced dim: {X_tr.shape[1]} → {X_tr_pca.shape[1]}')

    # ── 3. Train MLP ──────────────────────────────────────────────
    log(f'\nTraining MLP {tuple(args.hidden)}...')
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=args.patience,
        alpha=args.alpha,
    )
    mlp.fit(X_tr_pca, y_tr)

    # ── 4. Evaluate ───────────────────────────────────────────────
    train_acc = mlp.score(X_tr_pca, y_tr)
    val_acc   = mlp.score(X_val_pca, y_val)
    log(f'\n{"="*50}')
    log(f'  Train acc : {train_acc*100:.2f}%')
    log(f'  Val   acc : {val_acc*100:.2f}%')
    log(f'  Gap       : {(train_acc - val_acc)*100:.2f}%')
    log(f'{"="*50}')

    # ── 5. Save models ────────────────────────────────────────────
    pickle.dump(pca, open(os.path.join(out_dir, 'pca.model'), 'wb'))
    pickle.dump(mlp, open(os.path.join(out_dir, 'mlp.model'), 'wb'))
    log(f'  PCA model → {out_dir}/pca.model')
    log(f'  MLP model → {out_dir}/mlp.model')

    # ── 6. Save predictions ───────────────────────────────────────
    val_preds  = mlp.predict(X_val_pca)
    test_preds = mlp.predict(X_test_pca)

    val_out = os.path.join(out_dir, 'val_pca_mlp.csv')
    with open(val_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(val_ids, val_preds):
            f.write(f'{vid},{pred}\n')

    test_out = os.path.join(out_dir, 'test_pca_mlp.csv')
    with open(test_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(test_ids, test_preds):
            f.write(f'{vid},{pred}\n')

    log(f'\n  Val  predictions → {val_out}')
    log(f'  Test predictions → {test_out}  ← 上传 Kaggle')

    log_f.close()


if __name__ == '__main__':
    main()
