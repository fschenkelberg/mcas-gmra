import torch as pt
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch, BisectingKMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from stellargraph import datasets

def read_data(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            float_values = [float(entry) for entry in entries]
            data_list.append(float_values)
    tensor_data = pt.tensor(data_list)

    return tensor_data

# Original
node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/attri2vec_embeddings_32.txt")

# Load the CiteSeer dataset to get ground truth labels
dataset = datasets.CiteSeer()
G, subjects = dataset.load(largest_connected_component_only=True)
ground_truth_labels = np.array([subjects[target] for target in subjects.index])

print(len(np.unique(subjects)))

# Define clustering algorithms
clustering_algorithms = [
    KMeans(n_clusters=len(np.unique(subjects)), random_state=0),
    AffinityPropagation(),
    MeanShift(),
    SpectralClustering(n_clusters=len(np.unique(subjects)), random_state=0),
    AgglomerativeClustering(n_clusters=len(np.unique(subjects))),
    DBSCAN(),
    OPTICS(),
    Birch(n_clusters=len(np.unique(subjects))),
    BisectingKMeans(n_clusters=len(np.unique(subjects)), random_state=0),
]

# Apply clustering algorithms and evaluate metrics
for algorithm in clustering_algorithms:
    algorithm.fit(node_embeddings)
    predicted_labels = algorithm.labels_
    
    # Convert ground truth labels to numeric if they are strings
    ground_truth_labels_numeric = np.array([int(label) for label in ground_truth_labels])
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels_numeric, predicted_labels)
    
    # Calculate precision, recall, and F1-score
    precision, recall, fscore, _ = precision_recall_fscore_support(ground_truth_labels_numeric, predicted_labels, average='weighted')
    
    # Print metrics
    print(f"Algorithm: {algorithm.__class__.__name__}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {fscore}")
