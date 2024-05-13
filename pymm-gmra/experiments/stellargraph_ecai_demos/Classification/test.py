from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np
from stellargraph.data import BiasedRandomWalk
from stellargraph import datasets
from gensim.models import Word2Vec

# Define the list of datasets
dataset_names = [
    # "Cora",
    # "CiteSeer",
    # "PubMedDiabetes",
    # "BlogCatalog3",
    # "MovieLens",
    # "AIFB",
    # "MUTAG",
    # "PROTEINS",
    # "WN18",
    "WN18RR",
    "FB15k",
    "FB15k_237",
    "IAEnronEmployees",
]

# Loop through each dataset
for dataset_name in dataset_names:
    print(f"Processing {dataset_name} dataset...")
    
    # Load dataset
    dataset = getattr(datasets, dataset_name)()
    G, node_subjects = dataset.load()

    # Biased random walk
    rw = BiasedRandomWalk(G)
    walks = rw.run(
        nodes=list(G.nodes()),
        length=100,
        n=10,
        p=0.5,
        q=2.0,
    )

    # Word2Vec model
    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)

    # Node embeddings and targets
    node_ids = model.wv.index_to_key
    node_embeddings = model.wv.vectors
    
    # Handle node IDs as strings
    node_id_mapping = {str(node_id): node_id for node_id in node_ids}
    node_targets = node_subjects[[node_id_mapping[node_id] for node_id in node_ids]]

    # Data splitting
    X = node_embeddings
    y = np.array(node_targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)

    # Classifier training
    clf = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300)
    clf.fit(X_train, y_train)

    # Prediction and accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {dataset_name}: {accuracy}")
