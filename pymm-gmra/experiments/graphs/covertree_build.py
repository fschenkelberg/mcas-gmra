# SYSTEM IMPORTS
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import pickle as pk

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree

def create_pkl(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base_filename)

    # Create the new filename with the ".json" extension
    pkl_filename = f"{root}.pkl"

    return pkl_filename

def create_json(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base_filename)

    # Create the new filename with the ".json" extension
    json_filename = f"{root}.json"

    return json_filename

# Read Data
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
    data_list = [data for data in data_list if len(data) == 128]
    # Convert the list of lists to a PyTorch tensor
    tensor_data = pt.tensor(data_list)

    return tensor_data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory to where covertree will serialize itself to")
    parser.add_argument("--validate", action="store_true",
                        help="if enabled, perform an expensive tree validate operation")
    parser.add_argument("--data_file", type=str, 
                        help="path to the data file")
    parser.add_argument("--max_scale", type=int, 
                        help="calculated max_scale for the given data file")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X_pt = read_data(args.data_file)
    print("done")

    cover_tree = CoverTree(max_scale=args.max_scale)

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

if __name__ == "__main__":
    main()
