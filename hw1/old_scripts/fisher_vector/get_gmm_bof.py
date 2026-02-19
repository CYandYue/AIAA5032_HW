#!/bin/python
"""
Generate GMM-based Bag-of-Features for videos
Uses soft assignments (posterior probabilities) instead of hard clustering
"""
import numpy
import os
import pickle
import sys
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("gmm_model", help="Path to trained GMM model")
parser.add_argument("cluster_num", type=int, help="Number of GMM components")
parser.add_argument("file_list", help="List of video IDs")
parser.add_argument("--mfcc_path", default="mfcc", help="Directory with MFCC files")
parser.add_argument("--output_path", default="bof", help="Output directory for features")
parser.add_argument("--use_soft", action="store_true",
                   help="Use soft assignments (posterior probabilities) instead of hard clustering")

if __name__ == '__main__':
    args = parser.parse_args()

    # 1. load the gmm model
    gmm = pickle.load(open(args.gmm_model, "rb"))
    print(f"Loaded GMM model with {gmm.n_components} components")
    print(f"Using {'soft' if args.use_soft else 'hard'} assignments")

    # 2. iterate over each video
    start = time.time()
    fread = open(args.file_list, "r")
    for line in tqdm(fread.readlines()):
        mfcc_path = os.path.join(args.mfcc_path, line.strip() + ".mfcc.csv")
        bof_path = os.path.join(args.output_path, line.strip() + ".csv")

        if not os.path.exists(mfcc_path):
            continue

        # Load MFCC features (num_frames, d)
        array = numpy.genfromtxt(mfcc_path, delimiter=";")

        if args.use_soft:
            # Soft assignment: use posterior probabilities
            # Returns (num_frames, n_components) - probability of each frame belonging to each component
            posteriors = gmm.predict_proba(array)
            # Average over all frames to get video-level representation
            list_freq = numpy.mean(posteriors, axis=0)
        else:
            # Hard assignment: similar to K-Means
            predictions = gmm.predict(array)
            # Count frequency of each component
            list_freq = numpy.zeros(args.cluster_num)
            for pred in predictions:
                list_freq[pred] += 1
            # Normalize by number of frames
            list_freq = list_freq / len(predictions)

        numpy.savetxt(bof_path, list_freq)

    end = time.time()
    print("GMM features generated successfully!")
    print("Time for computation: ", (end - start))
