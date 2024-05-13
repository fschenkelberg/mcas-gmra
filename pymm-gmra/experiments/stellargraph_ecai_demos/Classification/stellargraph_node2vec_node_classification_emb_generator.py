import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import numpy as np
from stellargraph.data import BiasedRandomWalk
from stellargraph import datasets
from gensim.models import Word2Vec

dataset = datasets.CiteSeer()
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

dimentions = [32,64,128,256]

for dim in dimentions:

    model = Word2Vec(str_walks, vector_size=dim, window=5, min_count=0, sg=1, workers=2, epochs=1)

    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )
    # node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

    # Output the embeddings to a text file
    output_dir = "/scratch/f006dg0/stellargraph_ecai_demos"
    filename = f"metapath2vec_embeddings_{dim}.txt"
    output_path = os.path.join(output_dir, filename)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        # Iterate over each node vector in model.wv.vectors
        for node_vector in model.wv.vectors:
            # Convert the node vector to a string with elements separated by spaces
            vector_str = " ".join(map(str, node_vector))
            # Write the node vector string to the file, followed by a newline character
            file.write(vector_str + "\n")
