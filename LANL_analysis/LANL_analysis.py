import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn.svm import SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score

import time
import warnings

# Suppress the specific UserWarning about collinear variables
warnings.filterwarnings("ignore", message="Variables are collinear", category=UserWarning)

N2V_FNAME = "auth_10mil_4_256.txt"
GMRA_FNAME = "auth_10mil_4__n2v_256_gmra.txt"
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
	order = pd.read_csv(N2V_FNAME, delimiter=" ", skiprows=1, usecols=[0], names= ["TGT"])
	y = get_y(order)
	print(y.sum())
	return n2v, gmra, y

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

import os

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
    n2v, gmra, y = load_lanl_data()
    print(n2v.shape, gmra.shape, y.shape)
    train_test_split_idx = int(len(n2v)*.8)
    train_data = [n2v[:train_test_split_idx], gmra[:train_test_split_idx]]
    test_data = [n2v[train_test_split_idx:], gmra[train_test_split_idx:]]
    y_train = y[:train_test_split_idx]
    y_test = y[train_test_split_idx:]

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
            print(name, data_idx, cm)

            # Save confusion matrix as image
            save_confusion_matrix(cm, data_idx, name)

        # reset classifiers after running them on each dataset
        classifiers = reset_classifiers()

    # plot results
    # df = pd.DataFrame({'N2V Embeddings': results[0],
    #                    'GMRA Embeddings': results[1]}, index=names)
    # ax = df.plot.bar(rot=70)
    # plt.title("Classifaction of Malicious Source Hosts Performance on LANL Data")
    # plt.ylabel("Classification Accuracy")
    # plt.subplots_adjust(bottom=0.15)
    # plt.show()

def reset_detectors(outliers_fraction):
    anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]
    return anomaly_algorithms


def anomaly_detection():

    n2v, gmra, y = load_lanl_data()
    print(n2v.shape, gmra.shape, y.shape)
    train_data = [n2v, gmra]
    y_train = y

    outliers_fraction = sum(y)/len(y)
    detectors = reset_detectors(outliers_fraction)

    results = [[] for i in range(len(train_data))]

    for data_idx in range(len(train_data)):
        for (name, detector) in detectors:
            X = train_data[data_idx]
            if name == "Local Outlier Factor":
                y_pred = (detector.fit_predict(X)-1)*-0.5
            else:
                y_pred = (detector.fit(X).predict(X)-1)*-0.5

            score = accuracy_score(y_train, y_pred)
            results[data_idx].append(score)

            cm = confusion_matrix(y_train, y_pred)
            print(name, data_idx, cm)

            print(score, data_idx, name)
        #reset classifiers after running them on each dataset
        detectors = reset_detectors(outliers_fraction)

    #plot results
    df = pd.DataFrame({'N2V Embeddings': results[0],
                        'GMRA Embeddings': results[1]}, index=[x[0] for x in detectors])
    ax = df.plot.bar(rot=70)
    plt.title("Anomaly Detection of Malicious Source Hosts Performance on LANL Data")
    plt.ylabel("Detection Accuracy")
    plt.subplots_adjust(bottom=0.15)
    plt.show()

if __name__ == "__main__":
    anomaly_detection()