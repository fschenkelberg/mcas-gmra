# This script is the generation of the
# 32, 64, 128, and 256 dim using attri2vec
# On the stellargraph dataset.

import os
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification
from tensorflow import keras
from stellargraph import datasets

dataset = datasets.CiteSeer()
G, subjects = dataset.load(largest_connected_component_only=True)

nodes = list(G.nodes())
number_of_walks = 4
length = 5

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 4

generator = Attri2VecLinkGenerator(G, batch_size)

layer_sizes = [32, 64, 128, 256]

for layer in layer_sizes:
    attri2vec = Attri2Vec(
        layer_sizes=[layer], generator=generator, bias=False, normalize=None
    )

    x_inp, x_out = attri2vec.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    history = model.fit(
        generator.flow(unsupervised_samples),
        epochs=epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=1,
        shuffle=True,
    )

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_gen = Attri2VecNodeGenerator(G, batch_size).flow(G.nodes())
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)

    output_dir = "/scratch/f006dg0/stellargraph_ecai_demos"
    output_path = os.path.join(output_dir, f"attri2vec_citeseer_node_classification_{layer}.txt")

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in node_embeddings:
            file.write(" ".join(map(str, embedding)) + "\n")
