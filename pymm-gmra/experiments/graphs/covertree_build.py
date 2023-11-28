# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree

def create_json(data_file):
    # Extract the base filename from the provided path
    base_filename = os.path.basename(data_file)

    # Create the new filename with the ".json" extension
    json_filename = f"{base_filename}.json"

    return json_filename

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory to where covertree will serialize itself to")
    parser.add_argument("--validate", action="store_true",
                        help="if enabled, perform an expensive tree validate operation")
    parser.add_argument("--data_file", type=str, 
                        help="path to the data file")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    X_pt = pt.from_numpy(X)
    print("done")

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

if __name__ == "__main__":
    main()
