#!/usr/bin/env python3
"""
用已保存的 BiLSTM 模型对测试集生成预测，不需要重新训练。
用法：
  python infer_bilstm.py <model_pt> <mfcc_dir> <test_label> <out_csv> [--hidden_dim 256] [--num_layers 2]
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Dataset ────────────────────────────────────────────────────────────────

class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir, max_len=None):
        self.samples, self.labels, self.video_ids = [], [], []

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
                if label == -1:  # 测试集：保留，用零向量占位
                    mfcc = np.zeros((1, 39), dtype='float32')
                else:
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
        return self.samples[idx], self.labels[idx], self.video_ids[idx]


def collate_fn(batch):
    mfccs, labels, video_ids = zip(*batch)
    lengths = [m.shape[0] for m in mfccs]
    max_len = max(lengths)
    feat_dim = mfccs[0].shape[1]
    padded = torch.zeros(len(mfccs), max_len, feat_dim)
    for i, m in enumerate(mfccs):
        padded[i, :m.shape[0]] = torch.FloatTensor(m)
    return padded, torch.LongTensor(labels), torch.LongTensor(lengths), list(video_ids)


# ─── Model（与 run_bilstm.py 完全一致）────────────────────────────────────────

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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_pt',    help='已保存的 best_model.pt 路径')
    parser.add_argument('mfcc_dir',    help='MFCC 特征目录')
    parser.add_argument('label_file',  help='测试集 label 文件（test_for_student.label）')
    parser.add_argument('out_csv',     help='输出预测 CSV 路径')
    parser.add_argument('--hidden_dim',  type=int, default=256)
    parser.add_argument('--num_layers',  type=int, default=2)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--max_len',     type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Model  : {args.model_pt}')
    print(f'Output : {args.out_csv}')

    ds = MFCCDataset(args.label_file, args.mfcc_dir, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f'Samples: {len(ds)}')

    model = BiLSTMClassifier(
        input_dim=39,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_pt, map_location=device))
    model.eval()

    all_ids, all_preds = [], []
    with torch.no_grad():
        for x, _, lengths, video_ids in loader:
            x, lengths = x.to(device), lengths.to(device)
            preds = model(x, lengths).argmax(1).cpu().tolist()
            all_ids.extend(video_ids)
            all_preds.extend(preds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(all_ids, all_preds):
            f.write(f'{vid},{pred}\n')

    print(f'Done. {len(all_preds)} predictions → {args.out_csv}')


if __name__ == '__main__':
    main()
