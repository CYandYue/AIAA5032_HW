#!/usr/bin/env python3
"""
集成两个系统的软概率：BiLSTM K-fold + Fisher Vector MLP。

用法：
  python ensemble_systems.py \
      experiments/bilstm_kfold_XXXX/test_kfold_tta.csv \
      experiments/fisher_proba_XXXX/test_fisher_proba.npy \
      experiments/fisher_proba_XXXX/test_ids.txt \
      [--w_bilstm 0.6 --w_fisher 0.4]

原理：
  - BiLSTM K-fold 已经保存了最终 CSV（硬预测）
  - Fisher 保存了软概率 .npy
  - 这里需要 BiLSTM 的软概率；脚本会自动从各 fold 的模型重新提取，
    或者直接从 K-fold 实验目录加载（如果你修改了脚本保存了 .npy）

简化方案：直接加权合并两个系统的 CSV 预测（多数投票）。
如果有 BiLSTM 软概率 .npy 则加权平均，否则退化为硬投票。
"""
import os
import sys
import argparse
import numpy as np


def load_csv_preds(csv_path):
    """加载 CSV 预测，返回 {video_id: category}"""
    preds = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Id'):
                continue
            vid, cat = line.split(',')
            preds[vid] = int(cat)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bilstm_csv',   help='BiLSTM K-fold 预测 CSV')
    parser.add_argument('fisher_npy',   help='Fisher MLP 软概率 .npy (N,10)')
    parser.add_argument('fisher_ids',   help='Fisher 对应的 video_id 列表 .txt')
    parser.add_argument('--bilstm_npy', default=None,
                        help='BiLSTM 软概率 .npy（可选；有则加权平均，无则硬投票）')
    parser.add_argument('--w_bilstm',   type=float, default=0.5,
                        help='BiLSTM 权重（有软概率时生效）')
    parser.add_argument('--w_fisher',   type=float, default=0.5,
                        help='Fisher 权重（有软概率时生效）')
    parser.add_argument('--out',        default=None)
    args = parser.parse_args()

    # ── 加载 Fisher 软概率 ──────────────────────────────────────
    fisher_proba = np.load(args.fisher_npy)       # (N, 10)
    fisher_ids   = [l.strip() for l in open(args.fisher_ids)]
    assert len(fisher_ids) == len(fisher_proba)
    print(f'Fisher proba: {fisher_proba.shape}  ({len(fisher_ids)} samples)')

    # 用 fisher_ids 作为对齐基准
    id_order = fisher_ids

    if args.bilstm_npy and os.path.exists(args.bilstm_npy):
        # ── 软概率加权平均 ───────────────────────────────────────
        bilstm_proba = np.load(args.bilstm_npy)
        assert len(bilstm_proba) == len(id_order), \
            f'BiLSTM proba shape {bilstm_proba.shape} vs Fisher {len(id_order)}'
        w_b = args.w_bilstm
        w_f = args.w_fisher
        total = w_b + w_f
        combined = (w_b * bilstm_proba + w_f * fisher_proba) / total
        preds = np.argmax(combined, axis=1)
        mode = f'soft blend (BiLSTM×{w_b} + Fisher×{w_f})'
    else:
        # ── 硬投票（BiLSTM CSV + Fisher argmax）────────────────
        bilstm_preds = load_csv_preds(args.bilstm_csv)
        fisher_hard  = np.argmax(fisher_proba, axis=1)

        preds = []
        for i, vid in enumerate(id_order):
            b_pred = bilstm_preds.get(vid, -1)
            f_pred = int(fisher_hard[i])
            if b_pred == -1:
                preds.append(f_pred)
            elif b_pred == f_pred:
                preds.append(b_pred)
            else:
                # 不一致时：用 Fisher 软概率最高置信度决策
                # （或简单取 Fisher，因为 Fisher 通常 calibration 较好）
                preds.append(f_pred if fisher_proba[i, f_pred] > 0.6 else b_pred)
        preds = np.array(preds)
        mode = 'hard vote (BiLSTM CSV + Fisher argmax)'

    print(f'Mode: {mode}')
    print(f'Predictions: {len(preds)} samples')

    out_path = args.out or os.path.join(
        os.path.dirname(args.bilstm_csv),
        'test_ensemble.csv'
    )
    with open(out_path, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(id_order, preds):
            f.write(f'{vid},{pred}\n')

    print(f'\nEnsemble predictions → {out_path}  ← 上传 Kaggle')


if __name__ == '__main__':
    main()
