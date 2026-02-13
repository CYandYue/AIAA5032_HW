#!/bin/python

import argparse
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys
import numpy as np

# Apply the MLP model to the testing videos;
# Output prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    line_stripped = line.strip()

    # Skip empty lines
    if not line_stripped:
      continue

    # Skip header line if present
    if line_stripped.startswith('Id,') or line_stripped == 'Id':
      continue

    # Handle both CSV format (Id,Category) and plain text (Id only)
    if ',' in line_stripped:
      # CSV format: extract only the video ID (first column)
      video_id = line_stripped.split(',')[0]
    else:
      # Plain text format: entire line is the video ID
      video_id = os.path.splitext(line_stripped)[0]

    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
