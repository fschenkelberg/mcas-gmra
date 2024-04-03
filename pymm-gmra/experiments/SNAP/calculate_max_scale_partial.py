import argparse
import numpy as np
import torch
from scipy.spatial.distance import pdist
import math
import os
import torch as pt

# Use: python /scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/graphs/calculate_max_scale_partial.py --data_file /thayerfs/home/f006dg0/theia_test_256.txt

def filename(data_file):
    base_filename = os.path.basename(data_file)
    root, _ = os.path.splitext(base_filename)
    
    return root

# Function to calculate max scale
# max scale = ceil(log_2(max(||x_i, x_j||_2^2)))
def calculate_max_scale(dataset, num_samples=10000):
    if isinstance(dataset, torch.Tensor):
        dataset = dataset.numpy()

    if not isinstance(dataset, np.ndarray):
        raise ValueError("Invalid type for dataset. Must be a PyTorch tensor or a NumPy array.")
    
    # if len(dataset) > num_samples:
    #     dataset = dataset[np.random.choice(len(dataset), num_samples, replace=False)]

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

def read_data(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            float_values = [float(entry) for entry in entries[1:]]
            data_list.append(float_values)
    data_list = [data for data in data_list if len(data) == 256]
    tensor_data = pt.tensor(data_list)

    return tensor_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    X_pt = read_data(args.data_file)
    print("done")

    max_scale = calculate_max_scale(X_pt)
    print("Max Scale:", max_scale)

if __name__ == "__main__":
    main()
