# verify that we're using the correct version of StellarGraph for this notebook
import numpy as np
import torch as pt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from stellargraph import datasets
from IPython.display import display, HTML

dataset = datasets.CiteSeer()
G, subjects = dataset.load(largest_connected_component_only=True)

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
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/attri2vec_citeseer_node_classification_32.txt")
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/attri2vec_citeseer_node_classification_64.txt")
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/attri2vec_citeseer_node_classification_128.txt")
# node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/attri2vec_citeseer_node_classification_256.txt")

# Reduced
node_embeddings = read_data("/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/attri2vec_citeseer_node_classification_example_dram.txt")

############################################################################################################
# The Node Classificaion Task
X = node_embeddings
y = np.array(subjects)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=None)

print(
    "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )
)

### Classifier Training
clf = LogisticRegressionCV(
    Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1000
)

clf.fit(X_train, y_train)

#Predict the hold-out test set.
y_pred = clf.predict(X_test)

# Original: 0.7553317535545023
# Reduced: 0.754739336492891
print(accuracy_score(y_test, y_pred))