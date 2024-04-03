import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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
	# SVC(kernel="linear", C=0.025, random_state=42),
	# SVC(gamma=2, C=1, random_state=42),
	DecisionTreeClassifier(max_depth=5, random_state=42),
	# RandomForestClassifier(
	# 	max_depth=5, n_estimators=10, max_features=1, random_state=42
	# ),
	MLPClassifier(alpha=1, max_iter=1000, random_state=42),
	AdaBoostClassifier(algorithm="SAMME", random_state=42),
	GaussianNB(),
	QuadraticDiscriminantAnalysis(),
	]
	return classifiers

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

	for data_idx in range(len(train_data)):
		for (classifier,name) in zip(classifiers,names):
			classifier.fit(train_data[data_idx], y_train)
			score = classifier.score(test_data[data_idx], y_test)
			results[data_idx].append(score)
			print(score, data_idx, name)
		#reset classifiers after running them on each dataset
		classifiers = reset_classifiers()

	#plot results
	df = pd.DataFrame({'N2V Embeddings': results[0],
	                    'GMRA Embeddings': results[1]}, index=names)
	ax = df.plot.bar(rot=70)
	plt.title("Classifaction of Malicious Source Hosts Performance on LANL Data")
	plt.ylabel("Classification Accuracy")
	plt.subplots_adjust(bottom=0.15)
	plt.show()

if __name__ == "__main__":
	main()
