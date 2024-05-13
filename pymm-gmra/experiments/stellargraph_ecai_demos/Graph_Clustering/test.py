from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stellargraph.datasets import Cora
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import numpy as np

# Assuming you have already loaded the dataset and trained the Word2Vec model
dataset = Cora()
G, subjects = dataset.load()

rw = BiasedRandomWalk(G)
walks = rw.run(
    nodes=list(G.nodes()),
    length=100,
    n=10,
    p=0.5,
    q=2.0,
)

str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, vector_size=128, window=5, min_count=0, sg=1, workers=2, epochs=1)

node_ids = model.wv.index_to_key
node_embeddings = model.wv.vectors
# node_targets = node_subjects.loc[[int(node_id) for node_id in node_ids]]

X_train, X_test, y_train, y_test = train_test_split(
    node_embeddings, np.array(subjects), test_size=0.2, random_state=42
)

### Kmeans Training
clf = KMeans(
    n_clusters=16
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import silhouette_score

# Assuming you have already trained your KMeans model and made predictions as in your code

# Calculate the silhouette score for the clustering
silhouette_avg = silhouette_score(X_test, y_pred)
print("Silhouette Score:", silhouette_avg)
