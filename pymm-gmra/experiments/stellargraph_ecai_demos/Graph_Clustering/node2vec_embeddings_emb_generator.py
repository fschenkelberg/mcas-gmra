import os
from stellargraph import datasets
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

dataset = datasets.Cora()
display(HTML(dataset.description))
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

dimentions = [32,64,128,256]

for dim in dimentions:
    model = Word2Vec(str_walks, vector_size=dim, window=5, min_count=0, sg=1, workers=2, epochs=1)

    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = node_subjects.loc[[int(node_id) for node_id in node_ids]]

    """Transform the embeddings to 2d space for visualisation"""

    # Output the embeddings to a text file
    output_dir = "/scratch/f006dg0/stellargraph_ecai_demos"
    output_path = os.path.join(output_dir, f"node2vec_embeddings_{dim}.txt")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in node_embeddings:
            file.write(" ".join(map(str, embedding)) + "\n")

"""
transform = TSNE  # PCA

trans = transform(n_components=2)
node_embeddings_2d = trans.fit_transform(node_embeddings)

# draw the embedding points, coloring them by the target label (paper subject)
alpha = 0.7
label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
node_colours = [label_map[target] for target in node_targets]

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
plt.show()
"""