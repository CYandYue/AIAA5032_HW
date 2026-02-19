#!/usr/bin/env python3
"""
Test XGBoost classifier on Fisher Vector features
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python test_xgboost.py model_file feat_dir feat_dim list_videos output_file")
        sys.exit(1)

    model_file  = sys.argv[1]
    feat_dir    = sys.argv[2]
    feat_dim    = int(sys.argv[3])
    list_videos = sys.argv[4]
    output_file = sys.argv[5]

    # Load model
    model = pickle.load(open(model_file, 'rb'))
    print(f"Loaded model from: {model_file}")

    # Load features
    video_ids = []
    feat_list = []

    for line in open(list_videos).readlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Skip header
        if line_stripped.startswith('Id,') or line_stripped == 'Id':
            continue
        # Handle both CSV (Id,Category) and plain text (Id) formats
        if ',' in line_stripped:
            video_id = line_stripped.split(',')[0]
        else:
            video_id = os.path.splitext(line_stripped)[0]

        feat_path = os.path.join(feat_dir, video_id + '.csv')
        if os.path.exists(feat_path):
            feat = np.genfromtxt(feat_path, delimiter=';', dtype='float')
            feat_list.append(feat)
        else:
            feat_list.append(np.zeros(feat_dim))

        video_ids.append(video_id)

    X = np.array(feat_list)
    print(f"Loaded {len(video_ids)} videos, feature shape: {X.shape}")

    # Predict
    predictions = model.predict(X)

    # Write output
    with open(output_file, 'w') as f:
        f.write('Id,Category\n')
        for vid, pred in zip(video_ids, predictions):
            f.write(f'{vid},{int(pred)}\n')

    print(f"Predictions saved to: {output_file}")
