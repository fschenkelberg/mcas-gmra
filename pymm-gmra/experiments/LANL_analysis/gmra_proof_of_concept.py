# SYSTEM IMPORTS
import math
import numpy as np
import torch
from scipy.spatial.distance import pdist
import argparse
import os
from tqdm import tqdm
import sys
import time

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_
print(sys.path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree

def read_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            id = str(entries[0])
            float_values = [float(entry) for entry in entries[1:]]
            if id not in data_dict or set(float_values) != set(data_dict[id]):
                data_dict[id] = float_values
    data_dict = {k: v for k, v in data_dict.items() if len(v) == 256}
    values = torch.tensor(list(data_dict.values()))
    keys = list(data_dict.keys())

    return keys, values

def calculate_max_scale(dataset, num_samples=10000):
    if isinstance(dataset, torch.Tensor):
        dataset = dataset.numpy()
    if not isinstance(dataset, np.ndarray):
        raise ValueError("Invalid type for dataset. Must be a PyTorch tensor or a NumPy array.")
    if len(dataset) > num_samples:
        dataset = dataset[np.random.choice(len(dataset), num_samples, replace=False)]
    distances = pdist(dataset, 'sqeuclidean')
    if len(distances) == 0:
        raise ValueError("No distances computed from the dataset.")
    max_distance = np.max(distances)
    if not (np.isfinite(max_distance) and max_distance > 0):
        raise ValueError("Invalid value for max_distance. Must be a positive numerical value.")
    log_value = math.log2(max_distance)
    if not np.isfinite(log_value):
        raise ValueError("Invalid value for log_value.")
    max_scale = math.ceil(log_value)
    if not np.isfinite(max_scale):
        raise ValueError("Invalid value for max_scale.")
    return max_scale

def calculate_block_size(max_scale):
    num_samples = 2**(2 * max_scale)
    # 2 ** (max_scale - 1)
    return num_samples

def get_nodes_at_depth(node, depth):
    if depth == 0:
        return [node]
    result = []
    for child in node.children:
        result+=get_nodes_at_depth(child,depth-1)
    return result

def best_depth(tree):
    depth_counter = 1
    best_nodes = [tree]
    best_dim = tree.basis.shape[1]
    while True:
        nodes = get_nodes_at_depth(tree,depth_counter)
        dims = list({node.basis.shape[1] for node in nodes})
        num_dims = len(dims)
        if num_dims == 1 and not dims[0] == 0:
            best_nodes = nodes
            best_dim = dims[0]
        else:
            return best_nodes, best_dim
        depth_counter += 1

def get_embeddings(tree, X):
    nodes, dim = best_depth(tree.root)
    basis = np.vstack([node.basis for node in nodes])
    idxs = np.hstack([node.idxs for node in nodes])
    sigmas = np.hstack([node.sigmas[:-1] for node in nodes])
    try:
        embeddings = np.multiply(basis, sigmas.reshape((basis.shape[0],1)))
    except:
        embeddings = basis
    reordered_embs = np.zeros((X.shape[0],embeddings.shape[1]))
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        reordered_embs[new_idx] = embeddings[idx]
    return reordered_embs

def create_filename(data_file):
    base_filename = os.path.basename(data_file)
    root, _ = os.path.splitext(base_filename)
    filename = f"{root}_dram.txt"
    return filename

def create_json(data_file):
    base_filename = os.path.basename(data_file)
    root, _ = os.path.splitext(base_filename)
    json_filename = f"{root}.json"
    return json_filename

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    start_time = time.time()
    ids, X_pt = read_data(args.data_file)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    max_scale = calculate_max_scale(X_pt)
    block_size = calculate_block_size(max_scale)

    print("max_scale: " + max_scale)
    print("block_size: " + block_size)
    
    '''
    print("Processing data in blocks...")
    num_blocks = math.ceil(len(X_pt) / block_size)
    for i in range(num_blocks):
        print("Block {i}")
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, len(X_pt))
        blocks = X_pt[start_idx:end_idx]
        # Process the blocks here
        cover_tree = CoverTree(max_scale=8)

        for pt_idx in tqdm(list(range(X_pt.shape[0])),
                        desc="building covertree"):
            cover_tree.insert_pt(pt_idx, X_pt)

        if(args.validate):
            print("validating covertree...this may take a while")
            assert(cover_tree.validate(X_pt))

        filename = create_json(args.data_file)
        filepath = os.path.join(args.data_dir, filename)

        print("serializing covertree to [%s]" % filepath)
        cover_tree.save(filepath)

        print("loading data")
        start_time = time.time()
        # ids, X = read_data_2(args.data_file)
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
        wavelet_tree = WaveletTree(dyadic_tree, X_pt, 0, X_pt.shape[-1])
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time-start_time))
        # print("took script {0:.4f} seconds to run".format(end_time-init_time))

        print("Extracting low-dimensional embeddings")
        start_time = time.time()
        embeddings = get_embeddings(wavelet_tree, X_pt)
        end_time = time.time()
        print("done. took {0:.4f} seconds".format(end_time - start_time))

        # Output the embeddings to a text file
        output_dir = "./graphs/results"

        output_path = os.path.join(output_dir, create_filename(args.data_file))

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open the file in 'w' mode (write mode)
        with open(output_path, 'a') as file: # Creats file if it does not exist, or appends to end if it does exist.
            for i, embedding in enumerate(embeddings):
                # Write the original ID followed by the embedding values
                file.write(f"{ids[i]} {' '.join(map(str, embedding))}\n")

        print("Low-dimensional embeddings saved to {0}".format(output_path))

        print("done")
        '''

if __name__ == "__main__":
    main()
