from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import time
import pickle as pk
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import math
from collections import Counter
from tensorflow.keras.datasets import mnist
import pickle as pk

#python mnist\dram\gmra_inverse.py mnist\results\mnist_covertree.json --wavelettree_path mnist\results\wavelettree_mnist.pkl

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree
from pysrc.utils.utils import *
from pysrc.utils.inverse import *


def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    parser.add_argument("--wavelettree_path", type=str, default=None, required=False, help="path to pk file for the wavelet tree (optional)")
    args = parser.parse_args()


    print("loading data")
    start_time = time.time()
    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    if args.wavelettree_path is not None:
        print(f"loading wavelet tree from {args.wavelettree_path}")
        start_time = time.time()
        wavelet_tree = pk.load(open(args.wavelettree_path, 'rb'))
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time-start_time))
    else:
        if not os.path.exists(args.covertree_path):
            raise ValueError("ERROR: covertree json file does not exist at [%s]"
                             % args.covertree_path)
        print("loading covertree from [%s]" % args.covertree_path)
        start_time = time.time()
        cover_tree: CoverTree = CoverTree(args.covertree_path)
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time-start_time))

        print("constructing dyadic tree")
        start_time = time.time()
        dyadic_tree = DyadicTree(cover_tree)
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time-start_time))

        print("constructing wavelet tree")
        start_time = time.time()
        wavelet_tree = WaveletTree(dyadic_tree, X, 0, X.shape[-1], inverse=True)
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time-start_time))
        print("took script {0:.4f} seconds to run".format(end_time-init_time))

        pk.dump(wavelet_tree, open("mnist/results/wavelettree_mnist.pkl", 'wb'))
        print("Saved wavelet tree to pickle file")

    print("Reconstructing X")
    start_time = time.time()
    projections = invert(wavelet_tree, X)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))
    # np.save('mnist/results/mnist_inverse.npy', projections)

    _, num_pts, num_scales = projections.shape
    count = 0
    #count of well reconstructed pts
    for pt in range(num_pts):
        for scale in range(num_scales):
            embedding = projections[:,pt, scale]
            #check its L2 norm
            if np.linalg.norm(embedding, 2) > .000001:
                count +=1
                pic = np.reshape(embedding,(28,28))
                plt.imshow(pic)
                plt.show()
    print(count)
    print("Finished saving results")

if __name__ == "__main__":
    main()