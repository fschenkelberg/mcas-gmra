import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from sklearn.metrics import accuracy_score
import keras

# Load the Cora dataset
dataset = sg.datasets.Cora()
G, _ = dataset.load(subject_as_feature=True)

# Split the graph into training and test sets
edge_splitter_test = EdgeSplitter(G)

G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

edge_splitter_train = EdgeSplitter(G_test)

G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define batch size and number of epochs
batch_size = 20
epochs = 20
num_samples = [20, 10]

# Create the GraphSAGELinkGenerator for training and testing data
train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

# Define the GraphSAGE model for edge classification
layer_sizes = [20, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

x_inp, x_out = graphsage.in_out_tensors()

# Perform link classification (edge classification)
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)

# Create the Keras model
model = keras.models.Model(inputs=x_inp, outputs=prediction)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["accuracy"],
)

# Evaluate the initial (untrained) model
init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Train the model
history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)
test_embeddings = model.predict(test_flow)

# Get the predicted labels for the test data
predicted_labels = (test_embeddings > 0.5).astype(int)

# Calculate the accuracy score
accuracy = accuracy_score(edge_labels_test, predicted_labels)

# Print the accuracy score
print("\nAccuracy Score:", accuracy)
