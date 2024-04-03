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

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
    return 2 ** max_scale

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
    filename = f"{root}_dram"
    return filename

def create_json(data_file):
    base_filename = os.path.basename(data_file)
    root, _ = os.path.splitext(base_filename)
    json_filename = f"{root}.json"
    return json_filename

def email(success, output_file=None, error=None):
    # Email configurations
    sender_email = "felicia.schenkelberg.th@dartmouth.edu"
    receiver_email = "felicia.schenkelberg.th@dartmouth.edu"

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email

    if success:
        subject = "GMRA File Generation Complete"
        message = f"Low-dimensional embeddings saved to {output_file}"
    else:
        subject = "Error with GMRA File Generation"
        message = f"Partial low-dimensional embeddings saved to {output_file}"

    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    with smtplib.SMTP('smtp.dartmouth.edu', 25) as server:
        server.sendmail(sender_email, receiver_email, msg.as_string())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    print("loading data")
    start_time = time.time()
    ids, X_pt = read_data(args.data_file)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    max_scale = calculate_max_scale(X_pt)
    block_size = calculate_block_size(max_scale)

    print("max_scale: " + str(max_scale))
    print("block_size: " + str(block_size))
    print("Processing data in blocks...")

    num_blocks = math.ceil(len(X_pt) / block_size)

    # Load existing CoverTree or create a new one if it doesn't exist
    filename = create_json(args.data_file)
    covertree_path = os.path.join(os.path.dirname(args.data_file), filename)

    if os.path.exists(covertree_path):
        # print("loading covertree from [%s]" % covertree_path)
        cover_tree: CoverTree = CoverTree(covertree_path)
    else:
        # print("creating new covertree")
        cover_tree = CoverTree(max_scale=max_scale)
    
    try:
        for i in range(num_blocks):
            block = X_pt[block_size * i: block_size * (i + 1)]

            # Insert the current block as a batch
            cover_tree.insert(block, is_batch=True)

            # Save the updated covertree after each batch
            print("Serializing covertree to [%s]" % covertree_path)

            cover_tree.save(covertree_path)

            if not os.path.exists(covertree_path):
                raise ValueError("ERROR: covertree json file does not exist at [%s]"
                            % covertree_path)

            print("loading covertree from [%s]" % covertree_path)
            start_time = time.time()
            cover_tree: CoverTree = CoverTree(covertree_path)
            end_time = time.time()
            print("done. took {0:.4f} seconds".format(end_time-start_time))

            print("constructing dyadic tree")
            start_time = time.time()
            dyadic_tree = DyadicTree(cover_tree)
            end_time = time.time()
            print("done. took {0:.4f} seconds".format(end_time-start_time))

            print("constructing wavelet tree")
            start_time = time.time()
            wavelet_tree = WaveletTree(dyadic_tree, block, 0, block.shape[-1])
            end_time = time.time()
            print("done. took {0:.4f} seconds".format(end_time-start_time))
            

            print("Extracting low-dimensional embeddings")
            start_time = time.time()
            embeddings = get_embeddings(wavelet_tree, block)
            end_time = time.time()
            print("done. took {0:.4f} seconds".format(end_time - start_time))

            # Output the embeddings to a text file
            output_dir = "./LANL/results"
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, create_filename(args.data_file) + f"_{i}.txt")

            # Open the file in 'w' mode (write mode)
            with open(output_path, 'w') as file:
                for embedding in embeddings:
                    file.write(" ".join(map(str, embedding)) + "\n")

            # Check if the end of the list has been reached
            if block_size * (i + 1) >= len(X_pt):
                break

        email(True, output_path, None)
    
    except Exception as e:
        email(False, None, e)

if __name__ == "__main__":
    main()