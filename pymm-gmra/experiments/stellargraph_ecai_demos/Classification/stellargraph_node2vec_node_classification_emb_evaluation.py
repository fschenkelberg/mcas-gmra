from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np
from stellargraph.data import BiasedRandomWalk
from stellargraph import datasets
from gensim.models import Word2Vec

dataset = datasets.Cora()
G, node_subjects = dataset.load(largest_connected_component_only=True)

rw = BiasedRandomWalk(G)

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))

str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)

node_ids = model.wv.index_to_key  # list of node IDs
node_embeddings = (
    model.wv.vectors
)
node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

def read_data(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split(' ')
            float_values = [float(entry) for entry in entries]
            data_list.append(float_values)
    # data_list = [data for data in data_list if len(data) == 128]
    numpy_data = np.array(data_list)

    return numpy_data

# Original
node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/Learned_Embeddings/stellargraph_node2vec_node_classification.txt")
# Reduced
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/stellargraph_node2vec_node_classification_dram.txt")

# Apply t-SNE transformation on node embeddings
# tsne = TSNE(n_components=2)
# node_embeddings_2d = tsne.fit_transform(node_embeddings)

# draw the points
# alpha = 0.7
# label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
# node_colours = [label_map[target] for target in node_targets]

# plt.figure(figsize=(10, 8))
# plt.scatter(
#     node_embeddings_2d[:, 0],
#     node_embeddings_2d[:, 1],
#     c=node_colours,
#     cmap="jet",
#     alpha=alpha,
# )

# X will hold the 128-dimensional input features
X = node_embeddings
# y holds the corresponding target values
y = np.array(node_targets)

### Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)

print(
    "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )
)

### Classifier Training
clf = LogisticRegressionCV(
    Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
)
clf.fit(X_train, y_train)

# Predict the hold out test set.

y_pred = clf.predict(X_test)

# Original: 0.7067501117568172
# Reduced: 
print(accuracy_score(y_test, y_pred))
