import stellargraph as sg
from stellargraph.datasets import Cora
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation, MeanShift, DBSCAN, OPTICS, Birch, DBSCAN, OPTICS
import torch as pt
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, adjusted_mutual_info_score, fowlkes_mallows_score

# Load the Cora dataset
dataset = Cora()
G, node_subjects = dataset.load()

rw = BiasedRandomWalk(G)

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)

str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)

# Retrieve node embeddings and corresponding subjects
node_ids = model.wv.index_to_key  # list of node IDs
node_embeddings = model.wv.vectors

# def read_data(file_path):
#     data_list = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             entries = line.strip().split(' ')
#             float_values = [float(entry) for entry in entries]
#             data_list.append(float_values)
#     tensor_data = pt.tensor(data_list)

#     return tensor_data

# # Load node embeddings
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/attri2vec_embeddings_dram.txt")
# Reduced
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/attri2vec_embeddings_dram.txt")

# Split the labeled nodes into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(
    node_embeddings, np.array(node_subjects), test_size=0.2, random_state=42
)

# Clustering algorithms to try
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=7),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=7),
    "SpectralClustering": SpectralClustering(n_clusters=7, affinity='nearest_neighbors', random_state=0),
    "AffinityPropagation": AffinityPropagation(damping=0.9, preference=-200),
    "MeanShift": MeanShift(bandwidth=0.1),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "OPTICS": OPTICS(min_samples=5),
    "Birch": Birch(n_clusters=7),
}

for name, algorithm in clustering_algorithms.items():
    try:
        # Evaluation
        y_pred = algorithm.fit_predict(X_train)

        # Example evaluation using silhouette score
        score = accuracy_score(y_test, y_pred)
        silhouette = silhouette_score(y_test, y_pred)
        db_index = davies_bouldin_score(y_test, y_pred)
        # ari = adjusted_rand_score(y_train, y_pred)
        # nmi = normalized_mutual_info_score(y_train, y_pred)
        # homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_train, y_pred)
        # ami = adjusted_mutual_info_score(y_train, y_pred)
        # fmi = fowlkes_mallows_score(y_train, y_pred)
        
        # Display Results
        print(f'{name} Accuracy: {score:.4f}')

    except ValueError as e:
        print(f"\n{name} - Error occurred:", e)
