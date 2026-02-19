#!/bin/python
import os
# Set environment variables BEFORE importing numpy/sklearn
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("mfcc_csv_file -- path to the mfcc csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    mfcc_csv_file = sys.argv[1]; 
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    # 1. load all mfcc features in one array
    selection = pd.read_csv(mfcc_csv_file, sep=';', dtype='float')

    # TA: perform kmeans clustering here. get a model file variable kmeans
    print(f"Loaded {len(selection)} MFCC feature vectors")
    print(f"Training K-Means with {cluster_num} clusters...")

    kmeans = KMeans(n_clusters=cluster_num, random_state=42, verbose=1, max_iter=300)
    kmeans.fit(selection)

    print(f"K-Means training completed. Inertia: {kmeans.inertia_:.2f}")

    # 2. Save trained model
    pickle.dump(kmeans, open(output_file, 'wb'))

    print("K-means trained successfully!")
