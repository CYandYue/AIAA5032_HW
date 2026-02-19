#!/usr/bin/env python3
"""
用 Fisher Vector + MLP 生成软概率（softmax），供集成用。

用法：
  python save_fisher_proba.py <fisher_dir> <gmm_dir> \
      [--out_dir experiments/fisher_proba]

需要先有训练好的 Fisher Vector 特征目录（fisher_k*_* 目录）。
"""
import os
import argparse
import numpy as np
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle


def load_fisher(csv_path):
    data = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('video_id'):
                continue
            parts = line.split(',')
            vid = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            data[vid] = vec
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fisher_dir', help='Fisher Vector 特征目录（含 train/val/test .csv）')
    parser.add_argument('--trainval_csv', default='labels/trainval.csv')
    parser.add_argument('--train_csv',    default='labels/train.csv')
    parser.add_argument('--val_csv',      default='labels/val.csv')
    parser.add_argument('--test_label',   default='labels/test_for_student.label')
    parser.add_argument('--out_dir',      default=None)
    parser.add_argument('--n_runs',       type=int, default=10,
                        help='多次 MLP 训练取平均（进一步减少随机性）')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or f'experiments/fisher_proba_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    # ── 加载 Fisher Vector 特征 ──────────────────────────────────
    print(f'Loading Fisher Vectors from {args.fisher_dir}...')
    train_fv = load_fisher(os.path.join(args.fisher_dir, 'train_fisher.csv'))
    val_fv   = load_fisher(os.path.join(args.fisher_dir, 'val_fisher.csv'))
    test_fv  = load_fisher(os.path.join(args.fisher_dir, 'test_fisher.csv'))
    dim = next(iter(train_fv.values())).shape[0]
    print(f'  Fisher Vector dim: {dim}')

    # ── 构建 trainval 数据 ───────────────────────────────────────
    def load_labels(csv_path, fv_dict, skip_missing=True):
        X, y, ids = [], [], []
        for line in open(csv_path):
            line = line.strip()
            if not line or line.startswith('Id') or line.startswith('video_id'):
                continue
            if ',' in line:
                vid, label = line.split(',')[0], int(line.split(',')[1])
            else:
                vid, label = os.path.splitext(line)[0], -1
            if vid not in fv_dict:
                if not skip_missing:
                    X.append(np.zeros(dim, dtype=np.float32))
                    y.append(label)
                    ids.append(vid)
                continue
            X.append(fv_dict[vid])
            y.append(label)
            ids.append(vid)
        return np.array(X), np.array(y), ids

    X_train, y_train, _ = load_labels(args.train_csv,    train_fv)
    X_val,   y_val,   _ = load_labels(args.val_csv,      val_fv)
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    X_test,  y_test, test_ids = load_labels(
        args.test_label, test_fv, skip_missing=False)

    print(f'  Trainval: {len(X_tv)}, Test: {len(X_test)}')

    # ── 归一化 ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tv_s   = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)

    # ── 多次训练 MLP，累加软概率 ─────────────────────────────────
    test_prob_sum = np.zeros((len(X_test), 10))
    val_accs = []

    for run in range(args.n_runs):
        seed = run * 42
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation='relu',
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
        mlp.fit(X_tv_s, y_tv)

        # 用 train+val 原始 val 集验证（这里用 val 部分估计）
        X_val_s = scaler.transform(X_val)
        val_acc = mlp.score(X_val_s, y_val)
        val_accs.append(val_acc)

        test_prob_sum += mlp.predict_proba(X_test_s)
        print(f'  Run {run+1:2d}/{args.n_runs}  val acc: {val_acc*100:.2f}%')

    test_proba_avg = test_prob_sum / args.n_runs
    print(f'\nMean val acc: {np.mean(val_accs)*100:.2f}%  '
          f'Std: {np.std(val_accs)*100:.2f}%')

    # ── 保存软概率 ───────────────────────────────────────────────
    proba_path = os.path.join(out_dir, 'test_fisher_proba.npy')
    ids_path   = os.path.join(out_dir, 'test_ids.txt')
    np.save(proba_path, test_proba_avg)
    with open(ids_path, 'w') as f:
        for vid in test_ids:
            f.write(vid + '\n')

    # 也保存单独的预测结果
    preds = np.argmax(test_proba_avg, axis=1)
    csv_path = os.path.join(out_dir, 'test_fisher_only.csv')
    with open(csv_path, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(test_ids, preds):
            f.write(f'{vid},{pred}\n')

    print(f'\nSoft proba → {proba_path}')
    print(f'IDs        → {ids_path}')
    print(f'CSV (alone)→ {csv_path}')
    print(f'\n接下来用 ensemble_systems.py 把 BiLSTM K-fold 和 Fisher 结果合并')


if __name__ == '__main__':
    main()
