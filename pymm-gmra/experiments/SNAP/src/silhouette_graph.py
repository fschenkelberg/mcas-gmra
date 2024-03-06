import argparse
import os
import matplotlib.pyplot as plt
import torch as pt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def calculate_silhouette(X, max_clusters):
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def filename(data_file):
    # Extract the base filename from the provided path
    base = os.path.basename(data_file)

    # Remove the extension from the base filename
    root, _ = os.path.splitext(base)

    # Create the new filename with the ".png" extension
    filename = f"{root}_silhouette.png"

    return filename

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="directory to where results will save to")
    parser.add_argument("--data_file", type=str, help="path to the data file")
    parser.add_argument("--max_clusters", type=int, default=10, help="maximum number of clusters for silhouette graph")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X_pt = read_data_and_count(args.data_file)
    print("done")

    silhouette_scores = calculate_silhouette(X_pt.numpy(), args.max_clusters)

    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, args.max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.xticks(np.arange(2, args.max_clusters + 1, 1))
    plt.grid(True)
    plt.savefig(os.path.join(args.data_dir, filename(args.data_file)))
    plt.show()

    # print("saving silhouette graph")

if __name__ == "__main__":
    main()
