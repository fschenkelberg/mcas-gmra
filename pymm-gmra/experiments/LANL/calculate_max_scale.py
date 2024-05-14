import argparse
import numpy as np
import torch
from scipy.spatial.distance import pdist
import math
# import pickle as pk
import os
import torch as pt

# Use: python /scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/graphs/calculate_max_scale_partial.py --data_file /thayerfs/home/f006dg0/theia_test_256.txt

def filename(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base_filename)

    return root

# Function to calculate max scale
# max scale = ceil(log_2(max(||x_i, x_j||_2^2)))
# Modified due to error: numpy.core._exceptions._ArrayMemoryError: Unable to allocate 5.34 TiB for an array with shape (733807368426,) and data type float64
# def calculate_max_scale(dataset):
def calculate_max_scale(dataset, num_samples=10000):
     # Check if the input is a PyTorch tensor
    if isinstance(dataset, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        dataset = dataset.numpy()

    # Check if the input is a NumPy array
    if not isinstance(dataset, np.ndarray):
        raise ValueError("Invalid type for dataset. Must be a PyTorch tensor or a NumPy array.")
    
    # Modified: Added this section to reduce num_points for max_scale calculation
    # Randomly sample num_samples points from the dataset
    if len(dataset) > num_samples:
        dataset = dataset[np.random.choice(len(dataset), num_samples, replace=False)]

    # Computes the squared Euclidean distance between the vectors.
    distances = pdist(dataset, 'sqeuclidean')

    # Check if the distances array is not empty
    if len(distances) == 0:
        raise ValueError("No distances computed from the dataset.")

    # Return the maximum of an array or maximum along an axis.
    max_distance = np.max(distances)

    # Check if max_distance is a valid, positive numerical value
    if not (np.isfinite(max_distance) and max_distance > 0):
        raise ValueError("Invalid value for max_distance. Must be a positive numerical value.")

    # returns the base 2 logarithm of a number.
    log_value = math.log2(max_distance)

    # Check if log_value is a valid numerical value
    if not np.isfinite(log_value):
        raise ValueError("Invalid value for log_value.")

    # rounds up and returns the smallest integer greater than or equal to a given number.
    max_scale = math.ceil(log_value)

    # Check if max_scale is a valid numerical value
    if not np.isfinite(max_scale):
        raise ValueError("Invalid value for max_scale.")

    return max_scale

import torch as pt

def read_data(file_path):
    # Initialize an empty list to store the data
    data_list = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into entries using space as the delimiter
            entries = line.strip().split(' ')
            
            # Skip the first entry (identifier) and convert subsequent entries to float
            float_values = [float(entry) for entry in entries[1:]]
            data_list.append(float_values)

    # Added to filter out any values that are not of the same length [Expected: 256]
    # data_list = [data for data in data_list if len(data) == 256]
    # Convert the list of lists to a PyTorch tensor
    tensor_data = pt.tensor(data_list)

    return tensor_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    # X = pk.load(open(args.data_file, "rb"))
    # X_pt = X[:int(.1*len(X))].detach()
    # X_pt = X.detach()
    X_pt = read_data(args.data_file)
    print("done")

    max_scale = calculate_max_scale(X_pt)
    print("Max Scale:", max_scale)

    # Save max_scale to a file
    # output_file = f"max_scale_{filename(args.data_file)}"
    # with open(output_file, "w") as file:
    #     file.write(str(max_scale))

    # print(f"Max scale saved to {output_file}")

if __name__ == "__main__":
    main()