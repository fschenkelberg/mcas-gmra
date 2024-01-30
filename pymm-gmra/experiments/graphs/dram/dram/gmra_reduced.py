# Node2Vec
# python ./graphs/dram/gmra.py ./graphs/results/theia_test_256.json --data_file /thayerfs/home/f006dg0/theia_test_256.txt --input_dim 256
# SYSTEM IMPORTS
from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import time
import pickle as pk

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_
print(sys.path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree
from pysrc.utils.utils import *

# This is similar to the original gmra, but instead of going to the max depth in the tree with the same dim, 
# to get embeddings we traverse to the dim that is represented by the most nodes that is not the max or 0.
# Note that nodes may not be at the same level to get the same dim.
# Also, not all pts will be represented. We will use all zeros in this case.
# This method is preferable to use if GMRA is giving max dim embeddings that are the same as the original data

def create_filename(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base_filename)

    # Create the new filename
    filename = f"{root}_reduced_dram.txt"

    return filename

def main():
	init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    parser.add_argument("--input_dim", type=int,
                        help="dimension of the input data")
    args = parser.parse_args()

    print("loading data")
    start_time = time.time()
    X = np.loadtxt(args.data_file, skiprows=1, usecols=tuple(range(1,args.input_dim+1)))
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

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
    wavelet_tree = WaveletTree(dyadic_tree, X, 0, X.shape[-1])
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))
    print("took script {0:.4f} seconds to run".format(end_time-init_time))

    best_dims = get_dim_dists(wavelet_tree)[::-1]
    print("Dim counts:", best_dims)
    best_dim=args.input_dim
    for dim,count in best_dims:
        if dim==0 or dim==args.input_dim:
            pass
        else:
            best_dim=dim
            break
    print("Selected dimension:", best_dim)
    embeddings = get_embeddings_at_dim(X,wavelet_tree,best_dim)
    print("Extracted embeddings, writing out to file!")


    # Output the embeddings to a text file
    output_dir = "./graphs/results"
    os.path.join(output_dir, create_filename(args.data_file))

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in embeddings:
            file.write(" ".join(map(str, embedding)) + "\n")

if __name__ == "__main__":
    main()