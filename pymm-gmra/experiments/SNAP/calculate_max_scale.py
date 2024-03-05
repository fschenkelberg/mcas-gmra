import argparse
import numpy as np
import torch
from scipy.spatial.distance import pdist
import math
import pickle as pk
import os

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
def calculate_max_scale(dataset):
     # Check if the input is a PyTorch tensor
    if isinstance(dataset, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        dataset = dataset.numpy()

    # Check if the input is a NumPy array
    if not isinstance(dataset, np.ndarray):
        raise ValueError("Invalid type for dataset. Must be a PyTorch tensor or a NumPy array.")
    
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    X = pk.load(open(args.data_file, "rb"))
    # X_pt = X[:int(.1*len(X))].detach()
    X_pt = X.detach()
    print("done")

    max_scale = calculate_max_scale(X_pt)
    print("Max Scale:", max_scale)

    # Save max_scale to a file
    output_file = f"max_scale_{filename(args.data_file)}"
    with open(output_file, "w") as file:
        file.write(str(max_scale))

    print(f"Max scale saved to {output_file}")

if __name__ == "__main__":
    main()
