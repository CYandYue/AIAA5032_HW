#!/bin/python
"""
Train Gaussian Mixture Model (GMM) for feature extraction
GMM uses soft clustering which typically works better than K-Means for audio features
"""
import os
# Set environment variables BEFORE importing numpy/sklearn
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("mfcc_csv_file", help="CSV file with selected MFCC features")
parser.add_argument("cluster_num", type=int, help="Number of GMM components")
parser.add_argument("output_file", help="Output model file path")
parser.add_argument("--covariance_type", default="diag",
                   choices=['full', 'tied', 'diag', 'spherical'],
                   help="Covariance type (default: diag)")

if __name__ == '__main__':
    args = parser.parse_args()

    # 1. load all mfcc features in one array
    selection = pd.read_csv(args.mfcc_csv_file, sep=';', dtype='float')

    print(f"Loaded {len(selection)} MFCC feature vectors")
    print(f"Training GMM with {args.cluster_num} components...")
    print(f"Covariance type: {args.covariance_type}")

    # 2. Train GMM
    gmm = GaussianMixture(
        n_components=args.cluster_num,
        covariance_type=args.covariance_type,
        random_state=42,
        verbose=2,
        max_iter=100,
        n_init=3
    )
    gmm.fit(selection)

    print(f"GMM training completed!")
    print(f"Converged: {gmm.converged_}")
    print(f"Log-likelihood: {gmm.lower_bound_:.2f}")

    # 3. Save trained model
    pickle.dump(gmm, open(args.output_file, "wb"))
    print(f"GMM model saved to {args.output_file}")
