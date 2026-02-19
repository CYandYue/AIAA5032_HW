#!/usr/bin/env python3
"""
Bagging with multiple MLPs on Fisher Vector features.
Each MLP is trained on a different random stratified split of trainval data.
Final prediction = average of all models' predicted probabilities (soft voting).
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
from sklearn.model_selection import StratifiedShuffleSplit


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
    parser.add_argument('feat_dir',       help='Directory with .csv feature files')
    parser.add_argument('trainval_csv',   help='Full trainval labels (Id,Category)')
    parser.add_argument('val_csv',        help='Fixed val labels for reporting')
    parser.add_argument('test_label',     help='Test video list')
    parser.add_argument('--n_models',     type=int,   default=10)
    parser.add_argument('--feat_dim',     type=int,   default=15600)
    parser.add_argument('--val_split',    type=float, default=0.15,
                        help='Fraction of trainval held out per split')
    parser.add_argument('--hidden',       type=int,   nargs='+', default=[200, 100])
    parser.add_argument('--alpha',        type=float, default=0.0001)
    parser.add_argument('--patience',     type=int,   default=20)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/bagging_mlp_{args.n_models}_{timestamp}'
    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

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

    # Load data
    log('\nLoading features...')
    tv_ids,  X_tv,  y_tv  = load_features(args.trainval_csv, args.feat_dir, args.feat_dim)
    val_ids, X_val, y_val = load_features(args.val_csv,      args.feat_dir, args.feat_dim)
    test_ids, X_test, _   = load_features(args.test_label,   args.feat_dir, args.feat_dim)
    log(f'  Trainval : {len(tv_ids)} samples')
    log(f'  Fixed val: {len(val_ids)} samples')
    log(f'  Test     : {len(test_ids)} samples')
    log(f'  Feat dim : {X_tv.shape[1]}')

    val_prob_sum  = np.zeros((len(val_ids),  10))
    test_prob_sum = np.zeros((len(test_ids), 10))
    individual_val_accs = []

    sss = StratifiedShuffleSplit(
        n_splits=args.n_models,
        test_size=args.val_split,
        random_state=42
    )

    for i, (tr_idx, _) in enumerate(sss.split(X_tv, y_tv)):
        log(f'\n{"="*55}')
        log(f'  Model {i+1}/{args.n_models}  |  train size: {len(tr_idx)}')
        log(f'{"="*55}')

        X_tr = X_tv[tr_idx]
        y_tr = y_tv[tr_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=i,          # different seed → different weight init
            verbose=False,
            early_stopping=True,     # uses 10% of X_tr internally
            validation_fraction=0.1,
            n_iter_no_change=args.patience,
            alpha=args.alpha,
        )
        mlp.fit(X_tr, y_tr)

        train_acc = mlp.score(X_tr, y_tr)
        val_acc   = mlp.score(X_val, y_val)
        individual_val_accs.append(val_acc)
        log(f'  Train acc: {train_acc*100:.2f}%  |  Fixed val acc: {val_acc*100:.2f}%')

        # Accumulate soft probabilities
        val_prob_sum  += mlp.predict_proba(X_val)
        test_prob_sum += mlp.predict_proba(X_test)

        # Running ensemble accuracy
        ens_acc = np.mean(np.argmax(val_prob_sum, axis=1) == y_val)
        log(f'  Ensemble val ({i+1} models): {ens_acc*100:.2f}%')

        # Save model
        pickle.dump(mlp, open(os.path.join(models_dir, f'mlp_bag_{i+1:02d}.model'), 'wb'))

    # Final results
    log(f'\n{"="*55}')
    log('  FINAL ENSEMBLE RESULTS')
    log(f'{"="*55}')

    val_preds  = np.argmax(val_prob_sum,  axis=1)
    test_preds = np.argmax(test_prob_sum, axis=1)
    final_val_acc = np.mean(val_preds == y_val)

    log(f'  Individual val accs : {[f"{a*100:.2f}%" for a in individual_val_accs]}')
    log(f'  Individual average  : {np.mean(individual_val_accs)*100:.2f}%')
    log(f'  Ensemble val acc    : {final_val_acc*100:.2f}%')

    # Save val predictions
    val_out = os.path.join(out_dir, 'val_bagging_mlp.csv')
    with open(val_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(val_ids, val_preds):
            f.write(f'{vid},{pred}\n')

    # Save test predictions
    test_out = os.path.join(out_dir, 'test_bagging_mlp.csv')
    with open(test_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(test_ids, test_preds):
            f.write(f'{vid},{pred}\n')

    log(f'\n  Val  predictions -> {val_out}')
    log(f'  Test predictions -> {test_out}')

    log_f.close()


if __name__ == '__main__':
    main()
