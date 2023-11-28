import argparse
import numpy as np
import torch
from scipy.spatial.distance import pdist
import math

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

# Read Data
def read_data(file_path):
    # Initialize an empty list to store the data
    data_list = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into three values using a comma as the delimiter
            values = line.strip().split(' ')
            
            # Convert each value to a float and append to the list
            # Modified for the graphs
            # float_values = [float(val) for val in values]
            float_values = [float(val) for val in values[1:]]
            data_list.append(float_values)

    # Added to filter out any values that are not of the same length [Expected: 32]
    data_list = [data for data in data_list if len(data) == 32]
    # Convert the list of lists to a PyTorch tensor
    tensor_data = torch.tensor(data_list)

    return tensor_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    X_pt = read_data(args.data_path)
    print("done")

    max_scale = calculate_max_scale(X_pt)
    print("Max Scale:", max_scale)

    # Save max_scale to a file
    output_file = f"max_scale_{args.data_path.split('/')[-1]}"
    with open(output_file, "w") as file:
        file.write(str(max_scale))

    print(f"Max scale saved to {output_file}")

if __name__ == "__main__":
    main()
