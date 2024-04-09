import sys

import stellargraph as sg
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras

from IPython.display import display, HTML

# Load the Cora dataset
dataset = sg.datasets.Cora()
display(HTML(dataset.description))
graphs, graph_labels = dataset.load()

# Convert the single graph into a list
graph_list = [graphs]

# Calculate the summary DataFrame for the graphs
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graph_list],
    columns=["nodes", "edges"],
)

summary.describe().round(1)

generator = sg.mapper.PaddedGraphGenerator(graph_list)

gc_model = sg.layer.GCNSupervisedGraphClassification(
    [64, 32], ["relu", "relu"], generator, pool_all_layers=True
)

inp1, out1 = gc_model.in_out_tensors()
inp2, out2 = gc_model.in_out_tensors()

vec_distance = tf.norm(out1 - out2, axis=1)

pair_model = keras.Model(inp1 + inp2, vec_distance)
embedding_model = keras.Model(inp1, out1)

def graph_distance(graph1, graph2):
    spec1 = nx.laplacian_spectrum(graph1.to_networkx(feature_attr=None))
    spec2 = nx.laplacian_spectrum(graph2.to_networkx(feature_attr=None))
    k = min(len(spec1), len(spec2))
    return np.linalg.norm(spec1[:k] - spec2[:k])

graph_idx = np.random.RandomState(0).randint(len(graph_list), size=(100, 2))

targets = [graph_distance(graph_list[left], graph_list[right]) for left, right in graph_idx]

train_gen = generator.flow(graph_idx, batch_size=10, targets=targets)

pair_model.compile(keras.optimizers.Adam(1e-2), loss="mse")

embeddings = embedding_model.predict(generator.flow(graph_list))

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

train_labels, test_labels = model_selection.train_test_split(
    graph_labels, train_size=0.1, test_size=None, stratify=graph_labels
)

test_embeddings = embeddings[test_labels.index - 1]
train_embeddings = embeddings[train_labels.index - 1]

lr = LogisticRegression(multi_class="auto", solver="lbfgs")
lr.fit(train_embeddings, train_labels)

y_pred = lr.predict(test_embeddings)
gcn_acc = (y_pred == test_labels).mean()
print(f"Test classification accuracy: {gcn_acc}")

pd.crosstab(test_labels, y_pred, rownames=["true"], colnames=["predicted"])


from sklearn.manifold import TSNE

tsne = TSNE(2)
two_d = tsne.fit_transform(embeddings)

from matplotlib import pyplot as plt

plt.scatter(two_d[:, 0], two_d[:, 1], c=graph_labels.cat.codes, cmap="jet", alpha=0.4)

# Add title and labels if needed
plt.title("Scatter Plot of Two-Dimensional Data")
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")

# Save the plot to a file
plt.savefig("/scratch/f006dg0/mcas-gmra/experiments/results/gcn_unsupervised_graph_embeddings.png")
