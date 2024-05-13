import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import time
import warnings

import os

# Suppress the specific UserWarning about collinear variables
warnings.filterwarnings("ignore", message="Variables are collinear", category=UserWarning)

N2V_FNAME = "auth_10mil_4_256.txt"
GMRA_FNAME = "auth_10mil_4__n2v_256_gmra.txt"
# Additive GMRA Testing
AGMRA_FNAME = "auth_10mil_4_256_dram.txt"
Y_FNAME = "redteam.txt"

def get_y(order):
	df = pd.read_csv(Y_FNAME, delimiter=",", names=["ts", "srcUser", "src", "dst"])
	y =[]
	for host in order["TGT"]:
		if host in df["src"].values or host in df["dst"].values:
			y.append(1)
		else:
			y.append(0)
	return np.array(y)

def load_lanl_data():
    n2v = np.loadtxt(N2V_FNAME, skiprows=1, usecols=tuple(range(1,257)))
    gmra = np.loadtxt(GMRA_FNAME)
    agmra = np.loadtxt(AGMRA_FNAME, usecols=tuple(range(1,5)))
    order = pd.read_csv(N2V_FNAME, delimiter=" ", skiprows=1, usecols=[0], names= ["TGT"])
    y = get_y(order)
    print(y.sum())
    return n2v, gmra, agmra, y

def reset_classifiers():
	classifiers = [
	KNeighborsClassifier(3),
	DecisionTreeClassifier(max_depth=5, random_state=42),
	MLPClassifier(alpha=1, max_iter=1000, random_state=42),
	AdaBoostClassifier(algorithm="SAMME", random_state=42),
	GaussianNB(),
	QuadraticDiscriminantAnalysis(),
	]
	return classifiers

def save_confusion_matrix(cm, data_idx, classifier_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['Normal', 'Malicious'])
    plt.yticks([0, 1], ['Normal', 'Malicious'])
    plt.tight_layout()

    # Save the plot as an image
    if not os.path.exists("confusion_matrices"):
        os.makedirs("confusion_matrices")
    plt.savefig(f"confusion_matrices/{classifier_name}_{data_idx}_confusion_matrix.png")
    plt.close()

def main():
    n2v, gmra, agmra, y = load_lanl_data()
    print(n2v.shape, gmra.shape, agmra.shape, y.shape)
    train_test_split_idx = int(len(n2v)*.8)
    train_data = [n2v[:train_test_split_idx], gmra[:train_test_split_idx]]
    test_data = [n2v[train_test_split_idx:], gmra[train_test_split_idx:]]
    y_train = y[:train_test_split_idx]
    y_test = y[train_test_split_idx:]

    print("Training labels:", y_train)
    print("Testing labels:", y_test)

    names = [
        "Nearest Neighbors",
        "Decision Tree",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]
    classifiers = reset_classifiers()

    results = [[] for i in range(len(train_data))]
    times = []

    for data_idx in range(len(train_data)):
        for (classifier, name) in zip(classifiers, names):
            start_time = time.time()
            classifier.fit(train_data[data_idx], y_train)
            end_time = time.time()
            time_taken = end_time - start_time
            times.append(time_taken)
            score = classifier.score(test_data[data_idx], y_test)
            results[data_idx].append(score)
            print(score, data_idx, name, "Time taken:", time_taken)

            # Compute confusion matrix
            y_pred = classifier.predict(test_data[data_idx])

            print(f"{name} Classification Report:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)

            # Save confusion matrix as image
            save_confusion_matrix(cm, data_idx, name)

        # reset classifiers after running them on each dataset
        classifiers = reset_classifiers()

    # plot results
    df = pd.DataFrame({'N2V Embeddings': results[0],
                    'GMRA Embeddings': results[1],
                    'AGMRA Embeddings': results[2]}, index=names)
    ax = df.plot.bar(rot=70)
    plt.title("Classification of Malicious Source Hosts Performance on LANL Data")
    plt.ylabel("Classification Accuracy")
    plt.subplots_adjust(bottom=0.15)

    # Save the plot as an image
    if not os.path.exists("classification_results"):
        os.makedirs("classification_results")
    plt.savefig("classification_results/classification_results.png")

    # Close the plot to prevent displaying
    plt.close()

if __name__ == "__main__":
    main()