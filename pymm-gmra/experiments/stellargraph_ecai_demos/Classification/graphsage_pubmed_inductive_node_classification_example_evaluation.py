import numpy as np
import pandas as pd
import stellargraph as sg

from collections import Counter
from IPython.display import display, HTML

from sklearn import preprocessing, model_selection
from sklearn.metrics import accuracy_score

from stellargraph import datasets
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator

from tensorflow.keras import layers, optimizers, losses, Model

dataset = datasets.PubMedDiabetes()
display(HTML(dataset.description))
graph_full, labels = dataset.load()

seed = 123

labels_sampled = labels.sample(frac=0.8, replace=False, random_state=101)
graph_sampled = graph_full.subgraph(labels_sampled.index)

train_labels, test_labels = model_selection.train_test_split(
    labels_sampled,
    train_size=0.05,
    test_size=None,
    stratify=labels_sampled,
    random_state=42,
)

val_labels, test_labels = model_selection.train_test_split(
    test_labels, train_size=0.2, test_size=None, stratify=test_labels, random_state=100,
)

Counter(train_labels)
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_labels)
val_targets = target_encoding.transform(val_labels)
test_targets = target_encoding.transform(test_labels)

batch_size = 50
num_samples = [10, 10]

generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)
train_gen = generator.flow(train_labels.index, train_targets, shuffle=True)

graphsage_model = GraphSAGE(
    layer_sizes=[128, 128], generator=generator, bias=True, dropout=0.5,
)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
prediction.shape

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

val_gen = generator.flow(val_labels.index, val_targets)

history = model.fit(
    train_gen, epochs=15, validation_data=val_gen, verbose=0, shuffle=False
)

sg.utils.plot_history(history)

test_gen = generator.flow(test_labels.index, test_targets)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

generator = GraphSAGENodeGenerator(graph_full, batch_size, num_samples)

hold_out_nodes = labels.index.difference(labels_sampled.index)
labels_hold_out = labels[hold_out_nodes]
hold_out_targets = target_encoding.transform(labels_hold_out)
hold_out_gen = generator.flow(hold_out_nodes, hold_out_targets)
hold_out_predictions = model.predict(hold_out_gen)
hold_out_predictions = target_encoding.inverse_transform(hold_out_predictions)

results = pd.Series(hold_out_predictions, index=hold_out_nodes)
df = pd.DataFrame({"Predicted": results, "True": labels_hold_out})
df.head(10)

hold_out_metrics = model.evaluate(hold_out_gen)
print("\nHold Out Set Metrics:")
for name, val in zip(model.metrics_names, hold_out_metrics):
    print("\t{}: {:0.4f}".format(name, val))

embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(hold_out_gen)

# test_predictions = model.predict(test_gen)
hold_out_predictions = model.predict(hold_out_gen)

# test_labels_argmax = np.argmax(test_targets, axis=1)
hold_out_labels_argmax = np.argmax(hold_out_targets, axis=1)

# test_accuracy = accuracy_score(test_labels_argmax, np.argmax(test_predictions, axis=1))
hold_out_accuracy = accuracy_score(hold_out_labels_argmax, np.argmax(hold_out_predictions, axis=1))

print(hold_out_accuracy)
