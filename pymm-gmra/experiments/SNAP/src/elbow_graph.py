import argparse
import os
import matplotlib.pyplot as plt
import torch as pt
from sklearn.cluster import KMeans
import numpy as np

# Read Data
def read_data_and_count(file_path):
    max_shared_elements = 0
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            count = len(entries) - 1
            if count > max_shared_elements:
                max_shared_elements = count
            float_values = [float(entry) for entry in entries[1:]]
            data_list.append(float_values)

    filtered_data_list = [data for data in data_list if len(data) == max_shared_elements]
    tensor_data = pt.tensor(filtered_data_list)

    return tensor_data

def calculate_inertia(X, max_clusters):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias

def filename(data_file):
    # Extract the base filename from the provided path
    base = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base)

    # Create the new filename with the ".json" extension
    filename = f"{root}_elbow.png"

    return filename

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to where results will save to")
    parser.add_argument("--data_file", type=str, help="path to the data file")
    parser.add_argument("--max_clusters", type=int, default=10, help="maximum number of clusters for elbow graph")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X_pt = read_data_and_count(args.data_file)
    print("done")

    inertias = calculate_inertia(X_pt.numpy(), args.max_clusters)

    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.xticks(np.arange(1, args.max_clusters + 1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(args.data_dir, filename(args.data_file)))
    plt.show()

    # print("saving elbow graph")

if __name__ == "__main__":
    main()
