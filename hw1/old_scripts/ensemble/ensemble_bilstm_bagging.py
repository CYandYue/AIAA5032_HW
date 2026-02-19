#!/usr/bin/env python3
"""
集成 BiLSTM + Bagging MLP 的软概率。

用法（在 hw1/ 目录下）：
  python ensemble_bilstm_bagging.py \
    --bilstm_dir   experiments/bilstm_20260219_010641 \
    --bagging_dir  experiments/bagging_mlp_10_20260218_173046 \
    --fisher_dir   experiments/fisher_k200_20260216_173903/features \
    --mfcc_dir     /home/YueChang/phd_ws/AIAA5032/data/hkustgz-aiaa-5032-hw-1-spring-2025/mfcc.tgz/mfcc/ \
    --test_label   labels/test_for_student.label \
    --w_bilstm 0.5 --w_bagging 0.5
"""
import os, argparse, pickle, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ─── BiLSTM model definition ────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.w(x).squeeze(-1).masked_fill(~mask, float('-inf'))
        return (x * F.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=256, num_layers=2,
                 num_classes=10, dropout=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        self.attn = AttentionPooling(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x, lengths):
        B, T, F_ = x.shape
        x = self.input_bn(x.reshape(-1, F_)).reshape(B, T, F_)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(self.attn(out, lengths))


# ─── Dataset for BiLSTM ─────────────────────────────────────────────────────

class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir):
        self.samples, self.labels, self.video_ids = [], [], []
        for line in open(label_file):
            line = line.strip()
            if not line or line.startswith('Id') or line == 'Id':
                continue
            if ',' in line:
                vid, label = line.split(',')[0], int(line.split(',')[1])
            else:
                vid, label = os.path.splitext(line)[0], -1
            path = os.path.join(mfcc_dir, vid + '.mfcc.csv')
            if not os.path.exists(path):
                mfcc = np.zeros((1, 39), dtype='float32') if label == -1 else None
                if mfcc is None: continue
            else:
                mfcc = np.genfromtxt(path, delimiter=';', dtype='float32')
            if mfcc.ndim == 1: mfcc = mfcc.reshape(1, -1)
            mfcc = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-8)
            self.samples.append(mfcc)
            self.labels.append(label)
            self.video_ids.append(vid)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i], self.labels[i], self.video_ids[i]


def collate_fn(batch):
    mfccs, labels, ids = zip(*batch)
    lengths = [m.shape[0] for m in mfccs]
    T, F_ = max(lengths), mfccs[0].shape[1]
    padded = torch.zeros(len(mfccs), T, F_)
    for i, m in enumerate(mfccs):
        padded[i, :m.shape[0]] = torch.FloatTensor(m)
    return padded, torch.LongTensor(labels), torch.LongTensor(lengths), list(ids)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def bilstm_proba(model_pt, mfcc_dir, test_label, device, hidden_dim=256):
    ds = MFCCDataset(test_label, mfcc_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)
    model = BiLSTMClassifier(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()
    probs, ids = [], []
    with torch.no_grad():
        for x, _, lengths, vids in loader:
            x, lengths = x.to(device), lengths.to(device)
            probs.append(F.softmax(model(x, lengths), dim=1).cpu().numpy())
            ids.extend(vids)
    return ids, np.concatenate(probs, axis=0)


def bagging_proba(models_dir, fisher_dir, test_label, feat_dim=15600):
    # 加载测试集 Fisher Vector
    ids, feats = [], []
    for line in open(test_label):
        line = line.strip()
        if not line or line.startswith('Id') or line == 'Id':
            continue
        vid = os.path.splitext(line)[0] if ',' not in line else line.split(',')[0]
        path = os.path.join(fisher_dir, vid + '.csv')
        vec = np.genfromtxt(path, delimiter=';', dtype='float64') \
              if os.path.exists(path) else np.zeros(feat_dim)
        ids.append(vid)
        feats.append(vec)
    X = np.array(feats)

    # 累加所有 MLP 的 predict_proba
    model_files = sorted(
        f for f in os.listdir(models_dir) if f.endswith('.model'))
    print(f'  Found {len(model_files)} bagging MLP models')
    prob_sum = np.zeros((len(X), 10))
    for mf in model_files:
        clf = pickle.load(open(os.path.join(models_dir, mf), 'rb'))
        prob_sum += clf.predict_proba(X)
    return ids, prob_sum / len(model_files)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bilstm_dir',  required=True)
    parser.add_argument('--bagging_dir', required=True)
    parser.add_argument('--fisher_dir',  required=True)
    parser.add_argument('--mfcc_dir',    required=True)
    parser.add_argument('--test_label',  default='labels/test_for_student.label')
    parser.add_argument('--w_bilstm',    type=float, default=0.5)
    parser.add_argument('--w_bagging',   type=float, default=0.5)
    parser.add_argument('--hidden_dim',  type=int,   default=256)
    parser.add_argument('--out',         default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── BiLSTM 软概率 ──────────────────────────────────────────
    print('\n[1/2] BiLSTM inference...')
    bilstm_pt = os.path.join(args.bilstm_dir, 'best_model.pt')
    b_ids, b_proba = bilstm_proba(
        bilstm_pt, args.mfcc_dir, args.test_label, device, args.hidden_dim)
    print(f'  BiLSTM: {len(b_ids)} samples')

    # ── Bagging MLP 软概率 ─────────────────────────────────────
    print('\n[2/2] Bagging MLP inference...')
    models_dir = os.path.join(args.bagging_dir, 'models')
    m_ids, m_proba = bagging_proba(
        models_dir, args.fisher_dir, args.test_label)
    print(f'  Bagging MLP: {len(m_ids)} samples')

    # ── 对齐顺序（以 BiLSTM 的 id 顺序为基准）─────────────────
    m_id2idx = {vid: i for i, vid in enumerate(m_ids)}
    aligned_m_proba = np.zeros_like(b_proba)
    missing = 0
    for i, vid in enumerate(b_ids):
        if vid in m_id2idx:
            aligned_m_proba[i] = m_proba[m_id2idx[vid]]
        else:
            aligned_m_proba[i] = b_proba[i]  # fallback
            missing += 1
    if missing:
        print(f'  Warning: {missing} samples missing in Bagging MLP, using BiLSTM proba')

    # ── 加权平均 ───────────────────────────────────────────────
    total = args.w_bilstm + args.w_bagging
    combined = (args.w_bilstm * b_proba + args.w_bagging * aligned_m_proba) / total
    preds = np.argmax(combined, axis=1)

    # ── 保存 ──────────────────────────────────────────────────
    out_path = args.out or os.path.join(
        args.bilstm_dir,
        f'test_ensemble_b{args.w_bilstm}_m{args.w_bagging}.csv')
    with open(out_path, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(b_ids, preds):
            f.write(f'{vid},{pred}\n')

    print(f'\nEnsemble ({args.w_bilstm:.1f}×BiLSTM + {args.w_bagging:.1f}×Bagging) '
          f'→ {out_path}')
    print(f'Total samples: {len(preds)}  ← 上传 Kaggle')


if __name__ == '__main__':
    main()
