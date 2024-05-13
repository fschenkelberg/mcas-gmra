from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from stellargraph import datasets
from stellargraph.utils import plot_history

from tensorflow.keras import Model, regularizers
import tensorflow as tf

import os

tf.random.set_seed(1234)

dataset = datasets.Cora()
G, subjects = dataset.load()

generator = AdjacencyPowerGenerator(G, num_powers=10)

dimensions = [32,64,128,256]

for dimension in dimensions:
    wys = WatchYourStep(
        generator,
        num_walks=80,
        embedding_dimension=dimension,
        attention_regularizer=regularizers.l2(0.5),
    )
    x_in, x_out = wys.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))

    epochs = 100

    batch_size = 10
    train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)

    history = model.fit(
        train_gen, epochs=epochs, verbose=1, steps_per_epoch=int(len(G.nodes()) // batch_size)
    )

    plot_history(history)

    embeddings = wys.embeddings()
    print(embeddings.shape)

    # Output the embeddings to a text file
    output_dir = "./Classification/results"
    output_path = os.path.join(output_dir, f"watch_your_step_embeddings_{dimension}.txt")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in embeddings:
            file.write(" ".join(map(str, embedding)) + "\n")

"""
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

nodelist = list(G.nodes())

labels = subjects.loc[nodelist]
target_encoding = OneHotEncoder(sparse=False)
label_vectors = target_encoding.fit_transform(labels.values.reshape(-1, 1))

transform = TSNE

trans = transform(n_components=2)
emb_transformed = pd.DataFrame(trans.fit_transform(embeddings), index=nodelist)

emb_transformed["label"] = np.argmax(label_vectors, 1)

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
    "{} visualization of Watch Your Step embeddings for cora dataset".format(
        transform.__name__
    )
)
plt.show()

## Classification
# Here, we predict the class of a node by performing a weighted average of the training labels, with the weights determined by the similarity of that node's embedding with the training node embeddings.
# choose a random set of training nodes by permuting the labels and taking the first 300.
shuffled_idx = np.random.permutation(label_vectors.shape[0])
train_node_idx = shuffled_idx[:300]
test_node_idx = shuffled_idx[300:]

training_labels = label_vectors.copy()
training_labels[test_node_idx] = 0

d = embeddings.shape[1] // 2

predictions = np.dot(
    np.exp(np.dot(embeddings[:, :d], embeddings[:, d:].transpose())), training_labels
)

np.mean(
    np.argmax(predictions[test_node_idx], 1) == np.argmax(label_vectors[test_node_idx], 1)
)
"""