#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # Train MLP classifier
  print("Training MLP classifier...")
  print("Configuration:")
  print("  - Hidden layers: (200, 100)")
  print("  - Max iterations: 2000")
  print("  - Early stopping: enabled")
  print("  - Validation fraction: 0.1")

  model = MLPClassifier(
      hidden_layer_sizes=(200, 100),
      activation='relu',
      solver='adam',
      max_iter=2000,
      random_state=42,
      verbose=True,
      early_stopping=True,
      validation_fraction=0.1,
      n_iter_no_change=20,
      alpha=0.0001
  )
  model.fit(X, y)
  print("MLP training completed!")
  print(f"Final training loss: {model.loss_:.4f}")
  print(f"Best validation score: {model.best_validation_score_:.4f}")
  print(f"Converged in {model.n_iter_} iterations")

  # save trained MLP in output_file
  pickle.dump(model, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
