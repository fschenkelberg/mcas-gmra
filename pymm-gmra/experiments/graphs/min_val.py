import argparse
import pickle as pk
import torch
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,
                        help="path to the data file")
    args = parser.parse_args()

    print("loading data")
    X = pk.load(open(args.data_file, "rb"))
    X_pt = X.detach()
    print("done")

    # Convert the PyTorch tensor to a NumPy array
    X_np = X_pt.numpy()

    # Find the minimum value in the dataset
    min_value = X_np.min()

    print("The smallest value in the dataset is:", min_value)

    # Generate a random epsilon a few orders of magnitude smaller than the smallest value
    epsilon = np.random.uniform(low=1e-10, high=1e-7) * min_value

    print("Generated epsilon:", epsilon)

if __name__ == "__main__":
    main()
