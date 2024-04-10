import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

RESULTS_FOLDER = "/scratch/f006dg0/mcas-gmra/pymm-gmra/experiments/LANL/results"
N2V_FNAME = "auth_10mil_4_256.txt"
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

def load_lanl_data(N2V_FNAME, GMRA_FNAME):
	n2v = np.loadtxt(N2V_FNAME, skiprows=1, usecols=tuple(range(1,257)))
	gmra = np.loadtxt(GMRA_FNAME)
	n2v = n2v[:len(gmra)]
	order = pd.read_csv(N2V_FNAME, delimiter=" ", skiprows=1, usecols=[0], names= ["TGT"])
	y = get_y(order)
	y = y[:len(gmra)]
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

def main():
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith(".txt"):
            GMRA_FNAME = os.path.join(RESULTS_FOLDER, filename)
            n2v, gmra, y = load_lanl_data(N2V_FNAME, GMRA_FNAME)
            # print(n2v.shape, gmra.shape, y.shape)
            train_test_split_idx = int(len(n2v) * .8)
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
            results = [[] for _ in range(len(train_data))]

            for data_idx in range(len(train_data)):
                for (classifier, name) in zip(classifiers, names):
                    classifier.fit(train_data[data_idx], y_train)
                    score = classifier.score(test_data[data_idx], y_test)
                    results[data_idx].append(score)
                    print(score, data_idx, name)
                classifiers = reset_classifiers()

            df = pd.DataFrame({'N2V Embeddings': results[0],
                               'GMRA Embeddings': results[1]}, index=names)
            ax = df.plot.bar(rot=70)
            plt.title("Classifaction of Malicious Source Hosts Performance on LANL Data")
            plt.ylabel("Classification Accuracy")
            plt.subplots_adjust(bottom=0.15)
            # plt.savefig(f"plots/{GMRA_FNAME}_plt.png")
            plt.savefig('/scratch/f006dg0/mcas-gmra/LANL_analysis/plots/' + f"{os.path.basename(GMRA_FNAME)}_plt.png")

            plt.close()

if __name__ == "__main__":
    main()