#!/usr/bin/env python3
"""
1D CNN with residual blocks for audio classification.
Treats MFCC (T, 39) as a 1D signal with 39 channels.
Global Average Pooling handles variable-length sequences naturally.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse


# ─── Dataset (same as run_bilstm.py) ────────────────────────────────────────

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
                if label == -1:  # 测试集：必须保留，用零向量
                    mfcc = np.zeros((1, 39), dtype='float32')
                else:            # 训练集：跳过
                    continue
            else:
                mfcc = np.genfromtxt(mfcc_path, delimiter=';', dtype='float32')
            if mfcc.ndim == 1:
                mfcc = mfcc.reshape(1, -1)

            # Per-video CMVN
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

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim=39, num_classes=10, dropout=0.5):
        super().__init__()
        # Stem: wide receptive field to capture local patterns
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Residual blocks with increasing channels and stride-2 downsampling
        self.layer1 = ResBlock1D(64,  128, stride=2)   # T → T/2
        self.layer2 = ResBlock1D(128, 256, stride=2)   # T/2 → T/4
        self.layer3 = ResBlock1D(256, 256, stride=1)   # T/4 → T/4
        self.layer4 = ResBlock1D(256, 512, stride=2)   # T/4 → T/8

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.total_stride = 8  # product of all strides

    def masked_gap(self, x, lengths):
        """Global Average Pooling over valid (non-padded) positions."""
        B, C, T = x.shape
        # Scale original lengths to match output time dimension
        scaled = (lengths.float() / self.total_stride).long().clamp(min=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < scaled.unsqueeze(1)
        mask = mask.unsqueeze(1).float()          # (B, 1, T)
        return (x * mask).sum(-1) / mask.sum(-1)  # (B, C)

    def forward(self, x, lengths):
        # x: (B, T, 39) → (B, 39, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.masked_gap(x, lengths)  # (B, 512)
        x = self.dropout(x)
        return self.head(x)


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
    parser.add_argument('--dropout',    type=float, default=0.5)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--patience',   type=int,   default=20)
    parser.add_argument('--max_len',    type=int,   default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'experiments/cnn1d_{timestamp}'
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

    log(f'Device: {device}')

    log('\nLoading data...')
    train_ds  = MFCCDataset(args.train_csv,  args.mfcc_dir, args.max_len, augment=True)
    val_ds    = MFCCDataset(args.val_csv,    args.mfcc_dir, args.max_len, augment=False)
    test_ds   = MFCCDataset(args.test_label, args.mfcc_dir, args.max_len, augment=False)
    log(f'  Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CNN1DClassifier(input_dim=39, num_classes=10, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'\nModel params: {n_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc, patience_cnt, best_epoch = 0.0, 0, 0
    log(f'\n{"Epoch":>5}  {"TrainLoss":>9}  {"TrainAcc":>8}  {"ValLoss":>7}  {"ValAcc":>7}')
    log('-' * 50)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, False)
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
            log(f'\nEarly stopping (best val {best_val_acc*100:.2f}% @ epoch {best_epoch})')
            break

    log(f'\nLoading best model (epoch {best_epoch})...')
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
    model.eval()

    for loader, out_name in [(val_loader, 'val_cnn1d.csv'), (test_loader, 'test_cnn1d.csv')]:
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
    log(f'  Best val acc : {best_val_acc*100:.2f}%  (epoch {best_epoch})')
    log(f'  Test output  → {out_dir}/test_cnn1d.csv  ← 上传 Kaggle')
    log(f'{"="*50}')

    log_f.close()


if __name__ == '__main__':
    main()
