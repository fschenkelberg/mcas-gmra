# GraphSage
# python ./graphs/dram/gmra.py ./graphs/results/theia_test_256.json --data_file /thayerfs/home/f006dg0/theia_test_256.txt
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
import uuid

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_
print(sys.path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree

# The low-dimensional features for each point at an arbitrary scale (i.e. 0) are stored inside the wavelet nodes themselves. 
# When you want to extract the features for points in the dataset at a known scale, you need to traverse the tree to the 
# depth for that scale (i.e. 0) and then inspect all the nodes at that scale to get the features.

# Note: When you're at the correct depth, a single point only will be contained within
# one of the nodes at that level, you need to traverse down to that depth and then go through each node one at a time. 
# If you want to get more than a single point you just have to aggregate across nodes. 
# Be warned: the dimensionality at one node at depth d might be different than another node also at depth d!
def get_nodes_at_depth(node, depth):
    #Return a list of all nodes at depth in tree
    #subroutine to best_depth, start by passing in root node and depth you want
    #TODO: what happens when depth > max depth of node (return [])
    if depth == 0:
        return [node]
    result = []
    for child in node.children:
        result+=get_nodes_at_depth(child,depth-1)
    return result

def best_depth(tree):
    #Find the list of WaveletNodes that exist at the deepest level where all nodes have the same dimension
    #TODO: What happens if node.basis is empty, does this work still?
    depth_counter = 1
    #root will satisfy best depth parameters since all nodes are present in root
    best_nodes = [tree]
    best_dim = tree.basis.shape[1]
    while True:
        nodes = get_nodes_at_depth(tree,depth_counter)

        #check if this set is "good" - all nodes have the same dimension
        dims = list({node.basis.shape[1] for node in nodes})
        # print("dims", dims)
        # print("depth counter", depth_counter)

        num_dims = len(dims)

        #if num_dims is 1, then all nodes have the same dimension (good). need to make sure its not 0
        if num_dims == 1 and not dims[0] == 0:
            best_nodes = nodes
            best_dim = dims[0]
        else:
            #if this is a bad depth, the previous depth was BEST. return those nodes
            # print(depth_counter)
            return best_nodes, best_dim
        depth_counter += 1

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def normalize_embeddings(embeddings):
    normalized_embeddings = np.apply_along_axis(normalize_vector, axis=1, arr=embeddings)
    return normalized_embeddings

def get_embeddings(tree, X):
    #returns the embeddings matrix and the idxs map
    nodes, dim = best_depth(tree.root)

    #aggregate nodes along depth
    #need basis, idxs, and sigmas
    basis = np.vstack([node.basis for node in nodes])
    idxs = np.hstack([node.idxs for node in nodes])
    sigmas = np.hstack([node.sigmas[:-1] for node in nodes])

    #TODO check dimensions of basis, idxs, sigmas
    #expect: basis nxd where n is 1000 (num nodes) and d is the best dim
    #idxs is a column vector of length 1000 (node idxs corresponding to the elements in the basis)
    #sigmas is a column vector of length 1000 (scaling factors for each basis vector)
    try:
        embeddings = np.multiply(basis, sigmas.reshape((basis.shape[0],1)))
    except:
        embeddings = basis
    # print(embeddings.shape)

    # Normalize the embeddings
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # we need to reorder embeddings based on sigmas 
    reordered_embs = np.zeros((X.shape[0],normalized_embeddings.shape[1]))
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        reordered_embs[new_idx] = normalized_embeddings[idx]
    return reordered_embs

def create_filename(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base_filename)

    # Create the new filename with the ".json" extension
    filename = f"{root}_dram.txt"

    return filename

def create_filename_2(data_file):
    base_filename = os.path.basename(data_file)
    root, _ = os.path.splitext(base_filename)
    filename = f"{root}_test_256.txt"
    return filename

"""
# Read Data
def read_data(file_path):
    # Initialize an empty dictionary to store the data
    data_dict = {}

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into entries using space as the delimiter
            entries = line.strip().split(' ')
            
            # Extract id from the line
            id = str(entries[0])

            # Skip the first entry (identifier) and convert subsequent entries to float
            float_values = [float(entry) for entry in entries[1:]]

            # Check if the set of float_values already exists in the dictionary
            if id not in data_dict or set(float_values) != set(data_dict[id]):
                # Add the id and corresponding values to the dictionary
                data_dict[id] = float_values

    # Filter out any values that are not of the same length [Expected: 256]
    data_dict = {k: v for k, v in data_dict.items() if len(v) == 256}

    # Convert the dictionary values to a PyTorch tensor
    values = pt.tensor(list(data_dict.values()))

    # Converted to a list to make it subscribable
    keys = list(data_dict.keys())

    return keys, values
"""

# Read Data from Pickle File
def read_data_pickle(file_path):
    # Open the pickle file and read its contents
    with open(file_path, 'rb') as file:
        # Load data from pickle
        # data = np.array(pk.load(file))
        data = pk.load(file).detach().numpy()

    # Filter out any values that are not of the same length [Expected: 256]
    data = [values for values in data if len(values) == 256]

    # Generate UUIDs for each set of float_values
    ids = [uuid.uuid4() for _ in range(len(data))]

    # Convert the list of float_values to a PyTorch tensor
    values = pt.tensor(data)

    # Output the ids and values to a text file
    output_dir = "./graphs/results"
    # CHANGE OUTPUT FILE NAME!!
    output_path = os.path.join(output_dir, create_filename_2(file_path))

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for i in range(len(ids)):
            file.write(f"{ids[i]} {' '.join(map(str, data[i]))}\n")

    return ids, values

def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    start_time = time.time()
    ids, X = read_data_pickle(args.data_file)
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

    print("Extracting low-dimensional embeddings")
    start_time = time.time()
    embeddings = get_embeddings(wavelet_tree, X)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time - start_time))

    # Output the embeddings to a text file
    output_dir = "./graphs/results"
    # CHANGE OUTPUT FILE NAME!!
    output_path = os.path.join(output_dir, create_filename(args.data_file))

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for i, embedding in enumerate(embeddings):
            # Write the original ID followed by the embedding values
            file.write(f"{ids[i]} {' '.join(map(str, embedding))}\n")

    print("Low-dimensional embeddings saved to {0}".format(output_path))

if __name__ == "__main__":
    main()
