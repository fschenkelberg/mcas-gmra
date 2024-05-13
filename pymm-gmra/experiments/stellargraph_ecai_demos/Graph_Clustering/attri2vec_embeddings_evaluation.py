import numpy as np
import os
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from stellargraph import datasets

"""## Dataset

For clarity we ignore isolated nodes and subgraphs and use only the largest connected component.

(See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)
"""

dataset = datasets.CiteSeer()
G, subjects = dataset.load(largest_connected_component_only=True)


"""## Train attri2vec on Citeseer

Specify the other optional parameter values: root nodes, the number of walks to take per node, the length of each walk.
"""

nodes = list(G.nodes())
number_of_walks = 4
length = 5

"""Create the UnsupervisedSampler instance with the relevant parameters passed to it."""

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

"""Set the batch size and the number of epochs."""

batch_size = 50
epochs = 4

"""Define an attri2vec training generator, which generates a batch of (feature of target node, index of context node, label of node pair) pairs per iteration."""

generator = Attri2VecLinkGenerator(G, batch_size)
train_gen = generator.flow(unsupervised_samples)

"""Building the model: a 1-hidden-layer node representation ('input embedding') of the `target` node and the parameter vector ('output embedding') for predicting the existence of `context node` for each `(target context)` pair, with a link classification layer performed on the dot product of the 'input embedding' of the `target` node and the 'output embedding' of the `context` node.

Attri2Vec part of the model, with a 128-dimension hidden layer, no bias term and no normalization. (Normalization can be set to 'l2').
"""

dimention = 256

layer_sizes = [dimention]
attri2vec = Attri2Vec(
    layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
)

# Build the model and expose input and output sockets of attri2vec, for node pair inputs:
x_inp, x_out = attri2vec.in_out_tensors()

"""Use the link_classification function to generate the prediction, with the `ip` edge embedding generation method and the `sigmoid` activation, which actually performs the dot product of the 'input embedding' of the target node and the 'output embedding' of the context node followed by a sigmoid activation."""

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)

"""Stack the Attri2Vec encoder and prediction layer into a Keras model, and specify the loss."""

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

"""Train the model."""

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)

"""##  Visualise Node Embeddings

Build the node based model for predicting node representations from node content attributes with the learned parameters. Below a Keras model is constructed, with `x_inp[0]` as input and `x_out[0]` as output. Note that this model's weights are the same as those of the corresponding node encoder in the previously trained node pair classifier.
"""

x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

"""Get the node embeddings by applying the learned mapping function to node content features."""

node_gen = Attri2VecNodeGenerator(G, batch_size).flow(subjects.index)
node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)

node_gen = Attri2VecNodeGenerator(G, batch_size).flow(subjects.index)
node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)

# Output the embeddings to a text file
output_dir = "/scratch/f006dg0/stellargraph_ecai_demos"
output_path = os.path.join(output_dir, f"attri2vec_embeddings_{dimention}.txt")

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the file in 'w' mode (write mode)
with open(output_path, 'w') as file:
    for embedding in node_embeddings:
        file.write(" ".join(map(str, embedding)) + "\n")

"""Transform the embeddings to 2d space for visualisation."""

transform = TSNE  # PCA

trans = transform(n_components=2)
node_embeddings_2d = trans.fit_transform(node_embeddings)

# draw the embedding points, coloring them by the target label (paper subject)
alpha = 0.7
label_map = {l: i for i, l in enumerate(np.unique(subjects))}
node_colours = [label_map[target] for target in subjects]

plt.figure(figsize=(7, 7))
plt.axes().set(aspect="equal")
plt.scatter(
    node_embeddings_2d[:, 0],
    node_embeddings_2d[:, 1],
    c=node_colours,
    cmap="jet",
    alpha=alpha,
)
plt.title("{} visualization of node embeddings".format(transform.__name__))
plt.savefig(os.path.join(output_dir, f"attri2vec_embeddings_{dimention}_tsne_visualization.png"))
