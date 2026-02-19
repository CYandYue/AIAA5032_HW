#!/usr/bin/env python3
"""
BiLSTM K-fold ensemble + TTA + Label Smoothing.

三种正则化叠加，显著减少过拟合：
1. K-fold CV 集成：5个独立模型的 softmax 概率平均
2. 测试时数据增强 (TTA)：每个测试样本增强8次取平均
3. Label smoothing：防止模型过度自信

用法：
  python run_bilstm_kfold.py <mfcc_dir> <trainval_csv> <test_label>
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


# ─── Augmentation & Dataset ─────────────────────────────────────────────────

def augment_mfcc(mfcc, noise_std=0.05,
                 n_time_masks=2, time_mask_ratio=0.1,
                 n_freq_masks=2, freq_mask_ratio=0.15,
                 speed_perturb=True):
    mfcc = mfcc.copy()
    T, F = mfcc.shape

    if speed_perturb and T > 10:
        rate = np.random.uniform(0.8, 1.2)
        new_T = max(4, int(T * rate))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        mfcc = mfcc[indices]
        T = mfcc.shape[0]

    mfcc += np.random.randn(*mfcc.shape).astype(np.float32) * noise_std

    t_mask = int(T * time_mask_ratio)
    for _ in range(n_time_masks):
        if t_mask > 0 and T > t_mask:
            t0 = np.random.randint(0, T - t_mask)
            mfcc[t0:t0 + t_mask, :] = 0.0

    f_mask = int(F * freq_mask_ratio)
    for _ in range(n_freq_masks):
        if f_mask > 0 and F > f_mask:
            f0 = np.random.randint(0, F - f_mask)
            mfcc[:, f0:f0 + f_mask] = 0.0

    return mfcc


class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir, max_len=None, augment=False):
        self.samples, self.labels, self.video_ids = [], [], []
        self.augment = augment

        for line in open(label_file).readlines():
            line = line.strip()
            if not line or line.startswith('Id') or line == 'Id':
                continue
            if ',' in line:
                video_id, label = line.split(',')[0], int(line.split(',')[1])
            else:
                video_id, label = os.path.splitext(line)[0], -1

            mfcc_path = os.path.join(mfcc_dir, video_id + '.mfcc.csv')
            if not os.path.exists(mfcc_path):
                if label == -1:   # 测试集：保留，用零向量占位
                    mfcc = np.zeros((1, 39), dtype='float32')
                else:             # 训练集：跳过
                    continue
            else:
                mfcc = np.genfromtxt(mfcc_path, delimiter=';', dtype='float32')
            if mfcc.ndim == 1:
                mfcc = mfcc.reshape(1, -1)

            mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)
            if max_len:
                mfcc = mfcc[:max_len]

            self.samples.append(mfcc)
            self.labels.append(label)
            self.video_ids.append(video_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mfcc = self.samples[idx]
        if self.augment:
            mfcc = augment_mfcc(mfcc)
        return mfcc, self.labels[idx], self.video_ids[idx]


def collate_fn(batch):
    mfccs, labels, video_ids = zip(*batch)
    lengths = [m.shape[0] for m in mfccs]
    max_len = max(lengths)
    feat_dim = mfccs[0].shape[1]
    padded = torch.zeros(len(mfccs), max_len, feat_dim)
    for i, m in enumerate(mfccs):
        padded[i, :m.shape[0]] = torch.FloatTensor(m)
    return padded, torch.LongTensor(labels), torch.LongTensor(lengths), list(video_ids)


# ─── Model ──────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.w(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=256, num_layers=2,
                 num_classes=10, dropout=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = AttentionPooling(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths):
        B, T, F = x.shape
        x = self.input_bn(x.reshape(-1, F)).reshape(B, T, F)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        pooled = self.attn(out, lengths)
        return self.head(pooled)


# ─── Train / Eval ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y, lengths, _ in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def predict_tta(model, dataset_aug, device, n_tta=8, batch_size=64):
    """
    TTA: 对 dataset_aug（augment=True）进行 n_tta 次随机增强推断，
    返回平均 softmax 概率矩阵 (N, num_classes)。
    """
    model.eval()
    n = len(dataset_aug)
    num_classes = 10
    prob_sum = np.zeros((n, num_classes), dtype=np.float64)

    loader = DataLoader(dataset_aug, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=2)

    for _ in range(n_tta):
        probs_list = []
        with torch.no_grad():
            for x, _, lengths, _ in loader:
                x, lengths = x.to(device), lengths.to(device)
                logits = model(x, lengths)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        prob_sum += np.concatenate(probs_list, axis=0)

    return prob_sum / n_tta


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mfcc_dir',       help='MFCC 特征目录')
    parser.add_argument('trainval_csv',   help='Trainval label 文件 (Id,Category)')
    parser.add_argument('test_label',     help='测试集 label 文件')
    parser.add_argument('--hidden_dim',      type=int,   default=256)
    parser.add_argument('--num_layers',      type=int,   default=2)
    parser.add_argument('--dropout',         type=float, default=0.5)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--epochs',          type=int,   default=150)
    parser.add_argument('--patience',        type=int,   default=20)
    parser.add_argument('--max_len',         type=int,   default=None)
    parser.add_argument('--n_folds',         type=int,   default=5,
                        help='K-fold 折数')
    parser.add_argument('--n_tta',           type=int,   default=8,
                        help='TTA 次数（推断时每个样本的增强次数）')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing 系数 (0=关闭, 推荐0.1)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/bilstm_kfold_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    log_f = open(os.path.join(out_dir, 'train.log'), 'w')

    def log(msg=''):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f'Output : {out_dir}')
    log(f'Device : {device}')
    log(f'\nHyperparameters:')
    for k, v in vars(args).items():
        log(f'  {k}: {v}')

    # ── 加载数据 ──────────────────────────────────────────────────
    log('\nLoading data...')
    # 同样的 trainval 数据：一份带增强（训练用），一份不带（验证用）
    tv_aug   = MFCCDataset(args.trainval_csv, args.mfcc_dir, args.max_len, augment=True)
    tv_noaug = MFCCDataset(args.trainval_csv, args.mfcc_dir, args.max_len, augment=False)
    # 测试集：带增强（TTA 用）
    test_aug = MFCCDataset(args.test_label,   args.mfcc_dir, args.max_len, augment=True)

    log(f'  Trainval: {len(tv_aug)}, Test: {len(test_aug)}')

    labels      = np.array(tv_noaug.labels)
    test_ids    = test_aug.video_ids

    # ── K-fold ────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    test_prob_sum = np.zeros((len(test_aug), 10), dtype=np.float64)
    fold_val_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(range(len(labels)), labels)):
        log(f'\n{"="*60}')
        log(f'Fold {fold+1}/{args.n_folds}  '
            f'(train={len(tr_idx)}, val={len(va_idx)})')
        log(f'{"="*60}')

        train_loader = DataLoader(
            Subset(tv_aug,   tr_idx),
            batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2
        )
        val_loader = DataLoader(
            Subset(tv_noaug, va_idx),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2
        )

        model = BiLSTMClassifier(
            input_dim=39,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val_acc, patience_cnt, best_epoch = 0.0, 0, 0
        model_path = os.path.join(out_dir, f'fold{fold+1}_best.pt')

        log(f'{"Epoch":>5}  {"TrainLoss":>9}  {"TrainAcc":>8}  '
            f'{"ValLoss":>7}  {"ValAcc":>7}')
        log('-' * 50)

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, True)
            va_loss, va_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, False)
            scheduler.step()

            mark = ''
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch
                patience_cnt = 0
                torch.save(model.state_dict(), model_path)
                mark = ' ✓'
            else:
                patience_cnt += 1

            log(f'{epoch:>5}  {tr_loss:>9.4f}  {tr_acc*100:>7.2f}%  '
                f'{va_loss:>7.4f}  {va_acc*100:>7.2f}%{mark}')

            if patience_cnt >= args.patience:
                log(f'\nEarly stopping (best val {best_val_acc*100:.2f}% '
                    f'@ epoch {best_epoch})')
                break

        fold_val_accs.append(best_val_acc)
        log(f'\nFold {fold+1} best val: {best_val_acc*100:.2f}%')

        # ── TTA 推断测试集 ────────────────────────────────────────
        log(f'Running TTA (n={args.n_tta}) on test set...')
        model.load_state_dict(torch.load(model_path, map_location=device))
        fold_test_probs = predict_tta(
            model, test_aug, device, args.n_tta, args.batch_size)
        test_prob_sum += fold_test_probs
        log(f'Fold {fold+1} TTA done.')

    # ── 最终集成 ──────────────────────────────────────────────────
    log(f'\n{"="*60}')
    log(f'Fold val accs : {[f"{a*100:.2f}%" for a in fold_val_accs]}')
    log(f'Mean val acc  : {np.mean(fold_val_accs)*100:.2f}%')
    log(f'Std val acc   : {np.std(fold_val_accs)*100:.2f}%')
    log(f'Total ensemble: {args.n_folds} folds × {args.n_tta} TTA '
        f'= {args.n_folds * args.n_tta} predictions per sample')
    log(f'{"="*60}')

    test_preds = np.argmax(test_prob_sum, axis=1)
    test_out = os.path.join(out_dir, 'test_kfold_tta.csv')
    with open(test_out, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(test_ids, test_preds):
            f.write(f'{vid},{pred}\n')

    log(f'\nTest predictions ({len(test_preds)} samples) → {test_out}  ← 上传 Kaggle')
    log_f.close()


if __name__ == '__main__':
    main()
