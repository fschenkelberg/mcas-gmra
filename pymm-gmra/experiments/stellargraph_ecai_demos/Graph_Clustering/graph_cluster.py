import pandas as pd
import numpy as np

from networkx.readwrite import json_graph

from sklearn.preprocessing import StandardScaler

import stellargraph as sg
from stellargraph.mapper import ClusterNodeGenerator
from stellargraph.layer import GCN
from stellargraph import globalvar

from tensorflow.keras import backend as K

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
from IPython.display import display, HTML
import matplotlib.pyplot as plt
# %matplotlib inline

"""## Loading the dataset

This notebook demonstrates node classification using the *Cluster-GCN* algorithm using one of two citation networks, `Cora` and `Pubmed`.
"""

display(HTML(datasets.Cora().description))
display(HTML(datasets.PubMedDiabetes().description))

"""(See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)"""

dataset = "cora"  # can also select 'pubmed'

if dataset == "cora":
    G, labels = datasets.Cora().load()
elif dataset == "pubmed":
    G, labels = datasets.PubMedDiabetes().load()

print(G.info())

set(labels)

"""### Splitting the data

We aim to train a graph-ML model that will predict the **subject** or **label** (depending on the dataset) attribute on the nodes.

For machine learning we want to take a subset of the nodes for training, and use the rest for validation and testing. We'll use scikit-learn again to do this.

The number of labeled nodes we use for training depends on the dataset. We use 140 labeled nodes for `Cora` and 60 for `Pubmed` training. The validation and test sets have the same sizes for both datasets. We use 500 nodes for validation and the rest for testing.
"""

if dataset == "cora":
    train_size = 140
elif dataset == "pubmed":
    train_size = 60

train_labels, test_labels = model_selection.train_test_split(
    labels, train_size=train_size, test_size=None, stratify=labels
)
val_labels, test_labels = model_selection.train_test_split(
    test_labels, train_size=500, test_size=None, stratify=test_labels
)

"""Note using stratified sampling gives the following counts:"""

from collections import Counter

Counter(train_labels)

"""The training set has class imbalance that might need to be compensated, e.g., via using a weighted cross-entropy loss in model training, with class weights inversely proportional to class support. However, we will ignore the class imbalance in this example, for simplicity.

### Converting to numeric arrays

For our categorical target, we will use one-hot vectors that will be fed into a soft-max Keras layer during training.
"""

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_labels)
val_targets = target_encoding.transform(val_labels)
test_targets = target_encoding.transform(test_labels)

"""Next, we prepare a Pandas DataFrame holding the node attributes we want to use to predict the subject. These are the feature vectors that the Keras model will use as input. `Cora` contains attributes 'w_x' that correspond to words found in that publication. If a word occurs more than once in a publication the relevant attribute will be set to one, otherwise it will be zero. `Pubmed` has similar feature vectors associated with each node but the values are [tf-idf.](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## Train using cluster GCN

### Graph Clustering

*Cluster-GCN* requires that a graph is clustered into `k` non-overlapping subgraphs. These subgraphs are used as batches to train a *GCN* model.

Any graph clustering method can be used, including random clustering that is the default clustering method in `StellarGraph`.

However, the choice of clustering algorithm can have a large impact on performance. In the *Cluster-GCN* paper, [1], it is suggested that the *METIS* algorithm is used as it produces subgraphs that are well connected with few intra-graph edges.

This demo uses random clustering by default.

#### METIS

In order to use *METIS*, you must download and install the official implementation from [here](http://glaros.dtc.umn.edu/gkhome/views/metis). Also, you must install the Python `metis` library by following the instructions [here.](https://metis.readthedocs.io/en/latest/)
"""

number_of_clusters = 10  # the number of clusters/subgraphs
clusters_per_batch = 2  # combine two cluster per batch
random_clusters = True  # Set to False if you want to use METIS for clustering

node_ids = np.array(G.nodes())

if random_clusters:
    # We don't have to specify the cluster because the CluserNodeGenerator will take
    # care of the random clustering for us.
    clusters = number_of_clusters
else:
    # We are going to use the METIS clustering algorith,
    print("Graph clustering using the METIS algorithm.")

    import metis

    lil_adj = G.to_adjacency_matrix().tolil()
    adjlist = [tuple(neighbours) for neighbours in lil_adj.rows]

    edgecuts, parts = metis.part_graph(adjlist, number_of_clusters)
    parts = np.array(parts)
    clusters = []
    cluster_ids = np.unique(parts)
    for cluster_id in cluster_ids:
        mask = np.where(parts == cluster_id)
        clusters.append(node_ids[mask])

"""Next we create the `ClusterNodeGenerator` object ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.mapper.ClusterNodeGenerator)) that will give us access to a generator suitable for model training, evaluation, and prediction via the Keras API.

We specify the number of clusters and the number of clusters to combine per batch, **q**.
"""

generator = ClusterNodeGenerator(G, clusters=clusters, q=clusters_per_batch, lam=0.1)

"""Now we can specify our machine learning model, we need a few more parameters for this:

 * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GCN layers with 32-dimensional hidden node features at each layer.
 * `activations` is a list of activations applied to each layer's output
 * `dropout=0.5` specifies a 50% dropout at each layer.

We create the `GCN` model ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.layer.GCN)) for training with *Cluster-GCN* as follows:
"""

cluster_gcn = GCN(
    layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.5
)

"""To create a Keras model we now expose the input and output tensors of the *Cluster-GCN* model for node prediction, via the `GCN.in_out_tensors` method ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.layer.GCN.in_out_tensors)):"""

x_inp, x_out = cluster_gcn.in_out_tensors()

x_inp

x_out

"""We are also going to add a final layer dense layer with softmax output activation. This layers performs classification so we set the number of units to equal the number of classes."""

predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

predictions

"""Finally, we build the TensorFlow model and compile it specifying the loss function, optimiser, and metrics to monitor."""

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

"""### Train the model

We are now ready to train the `GCN` model using *Cluster-GCN*, keeping track of its loss and accuracy on the training set, and its generalisation performance on a validation set.

We need two generators, one for training and one for validation data. We can create such generators by calling the `flow` method of the `ClusterNodeGenerator` object we created earlier and specifying the node IDs and corresponding ground truth target values for each of the two datasets.
"""

train_gen = generator.flow(train_labels.index, train_targets, name="train")
val_gen = generator.flow(val_labels.index, val_targets, name="val")

"""Finally, we are ready to train our `GCN` model by calling the `fit` method of our TensorFlow Keras model."""

history = model.fit(
    train_gen, validation_data=val_gen, epochs=20, verbose=1, shuffle=False
)

"""Plot the training history:"""

sg.utils.plot_history(history)

"""Evaluate the best model on the test set.

Note that *Cluster-GCN* performance can be very poor if using random graph clustering. Using *METIS* instead of random graph clustering produces considerably better results.
"""

test_gen = generator.flow(test_labels.index, test_targets)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

"""### Making predictions with the model

For predictions to work correctly, we need to remove the extra batch dimensions necessary for the implementation of *Cluster-GCN* to work. We can easily achieve this by adding a layer after the dense predictions layer to remove this extra dimension.
"""

predictions_flat = layers.Lambda(lambda x: K.squeeze(x, 0))(predictions)

# Notice that we have removed the first dimension
predictions, predictions_flat

"""Now let's get the predictions for all nodes.

We need to create a new model using the same as before input Tensor and our new **predictions_flat** Tensor as the output. We are going to re-use the trained model weights.
"""

model_predict = Model(inputs=x_inp, outputs=predictions_flat)

all_nodes = list(G.nodes())
all_gen = generator.flow(all_nodes, name="all_gen")
all_predictions = model_predict.predict(all_gen)

all_predictions.shape

"""These predictions will be the output of the softmax layer, so to get final categories we'll use the `inverse_transform` method of our target attribute specification to turn these values back to the original categories."""

node_predictions = target_encoding.inverse_transform(all_predictions)

"""Let's have a look at a few predictions after training the model:"""

len(all_gen.node_order)

results = pd.Series(node_predictions, index=all_gen.node_order)
df = pd.DataFrame({"Predicted": results, "True": labels})
df.head(10)

"""## Node embeddings

Evaluate node embeddings as activations of the output of the last graph convolution layer in the `GCN` layer stack and visualise them, coloring nodes by their true subject label. We expect to see nice clusters of papers in the node embedding space, with papers of the same subject belonging to the same cluster.

To calculate the node embeddings rather than the class predictions, we create a new model with the same inputs as we used previously `x_inp` but now the output is the embeddings `x_out` rather than the predicted class. Additionally note that the weights trained previously are kept in the new model.

Note that the embeddings from the `GCN` model have a batch dimension of 1 so we `squeeze` this to get a matrix of $N_{nodes} \times N_{emb}$.
"""

x_out_flat = layers.Lambda(lambda x: K.squeeze(x, 0))(x_out)
embedding_model = Model(inputs=x_inp, outputs=x_out_flat)

emb = embedding_model.predict(all_gen, verbose=1)
emb.shape

"""Project the embeddings to 2d using either TSNE or PCA transform, and visualise, coloring nodes by their true subject label"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

"""**Prediction Node Order**

The predictions are not returned in the same order as the input nodes given. The generator object internally maintains the order of predictions. These are stored in the object's member variable `node_order`. We use `node_order` to re-index the `node_data` DataFrame such that the prediction order in `y` corresponds to that of node embeddings in `X`.
"""

X = emb
y = np.argmax(
    target_encoding.transform(labels.reindex(index=all_gen.node_order)), axis=1,
)

if X.shape[1] > 2:
    transform = TSNE  # or use PCA for speed

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=all_gen.node_order)
    emb_transformed["label"] = y
else:
    emb_transformed = pd.DataFrame(X, index=list(G.nodes()))
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = y

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GCN embeddings for cora dataset".format(transform.__name__)
)
plt.show()
