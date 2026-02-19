#!/bin/python
"""
Extract Fisher Vector features from MFCC using trained GMM
Fisher Vector encodes the gradient of log-likelihood w.r.t GMM parameters
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pickle
import argparse
from tqdm import tqdm

def compute_fisher_vector(mfcc, gmm):
    """
    Compute Fisher Vector for a video's MFCC features

    Args:
        mfcc: (num_frames, d) MFCC features
        gmm: trained GaussianMixture model

    Returns:
        fisher_vector: (2*K*d,) flattened Fisher Vector
    """
    # Get GMM parameters
    K = gmm.n_components  # number of components
    d = gmm.means_.shape[1]  # feature dimension

    # Compute posterior probabilities (soft assignments)
    # gamma: (num_frames, K)
    gamma = gmm.predict_proba(mfcc)

    # Get GMM parameters
    means = gmm.means_  # (K, d)

    # Handle different covariance types
    if gmm.covariance_type == 'diag':
        covars = gmm.covariances_  # (K, d)
    elif gmm.covariance_type == 'full':
        covars = np.array([np.diag(gmm.covariances_[k]) for k in range(K)])  # (K, d)
    elif gmm.covariance_type == 'tied':
        covars = np.tile(np.diag(gmm.covariances_), (K, 1))  # (K, d)
    elif gmm.covariance_type == 'spherical':
        covars = np.tile(gmm.covariances_[:, np.newaxis], (1, d))  # (K, d)

    weights = gmm.weights_  # (K,)

    # Compute Fisher Vector components
    # For each Gaussian, compute gradient w.r.t. mean and variance

    # Initialize Fisher Vector
    fv_mu = np.zeros((K, d))
    fv_sigma = np.zeros((K, d))

    # Compute for each component
    for k in range(K):
        # Gradient w.r.t. mean (mu)
        # Sum over all frames: gamma(k,t) * (x_t - mu_k) / sigma_k
        diff = mfcc - means[k]  # (num_frames, d)
        weighted_diff = gamma[:, k:k+1] * diff / np.sqrt(covars[k])  # (num_frames, d)
        fv_mu[k] = np.sum(weighted_diff, axis=0) / (np.sqrt(weights[k]) * len(mfcc))

        # Gradient w.r.t. variance (sigma)
        # Sum over all frames: gamma(k,t) * [(x_t - mu_k)^2 / sigma_k^2 - 1]
        normalized_diff_sq = (diff ** 2) / covars[k] - 1  # (num_frames, d)
        weighted_diff_sq = gamma[:, k:k+1] * normalized_diff_sq / np.sqrt(2)  # (num_frames, d)
        fv_sigma[k] = np.sum(weighted_diff_sq, axis=0) / (np.sqrt(weights[k]) * len(mfcc))

    # Concatenate mean and variance gradients
    fisher_vector = np.concatenate([fv_mu.flatten(), fv_sigma.flatten()])

    # L2 normalization (power normalization + L2 norm)
    # Power normalization: sign(x) * |x|^alpha
    alpha = 0.5
    fisher_vector = np.sign(fisher_vector) * np.abs(fisher_vector) ** alpha

    # L2 normalization
    norm = np.linalg.norm(fisher_vector)
    if norm > 0:
        fisher_vector = fisher_vector / norm

    return fisher_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gmm_model", help="Path to trained GMM model")
    parser.add_argument("file_list", help="List of video IDs")
    parser.add_argument("--mfcc_path", default="mfcc", help="Directory with MFCC files")
    parser.add_argument("--output_path", default="fisher", help="Output directory")

    args = parser.parse_args()

    # Load GMM model
    print("Loading GMM model...")
    gmm = pickle.load(open(args.gmm_model, "rb"))

    K = gmm.n_components
    d = gmm.means_.shape[1]
    fv_dim = 2 * K * d

    print(f"GMM: {K} components, {d} dimensions")
    print(f"Fisher Vector dimension: {fv_dim}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Process each video
    print("Extracting Fisher Vectors...")
    fread = open(args.file_list, "r")

    processed = 0
    skipped = 0

    for line in tqdm(fread.readlines()):
        video_id = line.strip()
        mfcc_path = os.path.join(args.mfcc_path, video_id + ".mfcc.csv")
        fv_path = os.path.join(args.output_path, video_id + ".csv")

        if not os.path.exists(mfcc_path):
            skipped += 1
            continue

        # Load MFCC
        mfcc = np.genfromtxt(mfcc_path, delimiter=";")

        # Handle edge cases
        if len(mfcc.shape) == 1:
            mfcc = mfcc.reshape(1, -1)

        if mfcc.shape[0] == 0:
            skipped += 1
            continue

        # Compute Fisher Vector
        fisher_vector = compute_fisher_vector(mfcc, gmm)

        # Save
        np.savetxt(fv_path, fisher_vector, delimiter=";")
        processed += 1

    print(f"\n✓ Fisher Vector extraction complete!")
    print(f"  Processed: {processed} videos")
    print(f"  Skipped: {skipped} videos (no MFCC)")
    print(f"  Feature dimension: {fv_dim}")
