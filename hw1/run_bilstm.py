#!/usr/bin/env python3
"""
BiLSTM + Attention Pooling for audio classification.
Trained directly on raw variable-length MFCC sequences,
bypassing GMM/Fisher Vector entirely.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse
import pickle


# ─── Dataset ────────────────────────────────────────────────────────────────

def augment_mfcc(mfcc, noise_std=0.05,
                 n_time_masks=2, time_mask_ratio=0.1,
                 n_freq_masks=2, freq_mask_ratio=0.15,
                 speed_perturb=True, crop_len=None):
    """
    Strong augmentation for MFCC sequences:
    1. Speed perturbation  – subsample/oversample frames (±20%)
    2. Random crop         – take a random window of fixed length
    3. Gaussian noise      – small additive noise
    4. Multiple time masks – mask N random time blocks
    5. Multiple freq masks – mask N random frequency bands
    """
    mfcc = mfcc.copy()
    T, F = mfcc.shape

    # 1. Speed perturbation: randomly stretch or compress time axis
    if speed_perturb and T > 10:
        rate = np.random.uniform(0.8, 1.2)
        new_T = max(4, int(T * rate))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        mfcc = mfcc[indices]
        T = mfcc.shape[0]

    # 2. Random crop to fixed length (if specified)
    if crop_len is not None and T > crop_len:
        t0 = np.random.randint(0, T - crop_len)
        mfcc = mfcc[t0:t0 + crop_len]
        T = crop_len

    # 3. Gaussian noise
    mfcc += np.random.randn(*mfcc.shape).astype(np.float32) * noise_std

    # 4. Multiple time masks
    t_mask = int(T * time_mask_ratio)
    for _ in range(n_time_masks):
        if t_mask > 0 and T > t_mask:
            t0 = np.random.randint(0, T - t_mask)
            mfcc[t0:t0 + t_mask, :] = 0.0

    # 5. Multiple frequency masks
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
                if label == -1:  # 测试集：必须保留，用零向量
                    mfcc = np.zeros((1, 39), dtype='float32')
                else:            # 训练集：跳过
                    continue

            else:
                mfcc = np.genfromtxt(mfcc_path, delimiter=';', dtype='float32')
            if mfcc.ndim == 1:
                mfcc = mfcc.reshape(1, -1)

            # Per-video CMVN: zero mean, unit variance per MFCC dimension
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
        # x: (B, T, D)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.w(x).squeeze(-1)                    # (B, T)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)                   # (B, D)


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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mfcc_dir')
    parser.add_argument('train_csv')
    parser.add_argument('val_csv')
    parser.add_argument('test_label')
    parser.add_argument('--hidden_dim',  type=int,   default=256)
    parser.add_argument('--num_layers',  type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--max_len',     type=int,   default=None,
                        help='Truncate MFCC to this many frames (None = no limit)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/bilstm_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    # ── Logger: write to file + stdout simultaneously ──
    log_path = os.path.join(out_dir, 'train.log')
    log_f = open(log_path, 'w')

    def log(msg=''):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f'Output : {out_dir}')
    log(f'Device : {device}')
    log(f'Command: python run_bilstm.py ' + ' '.join(
        f'--{k} {v}' for k, v in vars(args).items()))
    log(f'\nHyperparameters:')
    for k, v in vars(args).items():
        log(f'  {k}: {v}')

    # ── Data ──
    log('\nLoading data...')
    train_ds = MFCCDataset(args.train_csv,  args.mfcc_dir, args.max_len, augment=True)
    val_ds   = MFCCDataset(args.val_csv,    args.mfcc_dir, args.max_len, augment=False)
    test_ds  = MFCCDataset(args.test_label, args.mfcc_dir, args.max_len, augment=False)
    log(f'  Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Model ──
    model = BiLSTMClassifier(
        input_dim=39,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'\nModel params: {n_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ──
    best_val_acc, patience_cnt, best_epoch = 0.0, 0, 0
    log(f'\n{"Epoch":>5}  {"TrainLoss":>9}  {"TrainAcc":>8}  {"ValLoss":>7}  {"ValAcc":>7}')
    log('-' * 50)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        mark = ''
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            mark = ' ✓'
        else:
            patience_cnt += 1

        log(f'{epoch:>5}  {tr_loss:>9.4f}  {tr_acc*100:>7.2f}%  '
            f'{va_loss:>7.4f}  {va_acc*100:>7.2f}%{mark}')

        if patience_cnt >= args.patience:
            log(f'\nEarly stopping at epoch {epoch} (best val {best_val_acc*100:.2f}% @ epoch {best_epoch})')
            break

    # ── Test predictions with best model ──
    log(f'\nLoading best model (epoch {best_epoch}, val {best_val_acc*100:.2f}%)...')
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
    model.eval()

    for loader, label_file, out_name in [
        (val_loader,  args.val_csv,    'val_bilstm.csv'),
        (test_loader, args.test_label, 'test_bilstm.csv'),
    ]:
        all_ids, all_preds = [], []
        with torch.no_grad():
            for x, _, lengths, video_ids in loader:
                x, lengths = x.to(device), lengths.to(device)
                preds = model(x, lengths).argmax(1).cpu().tolist()
                all_ids.extend(video_ids)
                all_preds.extend(preds)

        out_path = os.path.join(out_dir, out_name)
        with open(out_path, 'w') as f:
            f.write('Id,Category\n')
            for vid, pred in zip(all_ids, all_preds):
                f.write(f'{vid},{pred}\n')
        log(f'  {out_name} → {out_path}')

    log(f'\n{"="*50}')
    log(f'  Best val acc: {best_val_acc*100:.2f}% (epoch {best_epoch})')
    log(f'  Test predictions → {out_dir}/test_bilstm.csv  ← 上传 Kaggle')
    log(f'{"="*50}')
    log_f.close()


if __name__ == '__main__':
    main()
