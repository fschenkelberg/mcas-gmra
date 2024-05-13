import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Generating synthetic data with outliers
np.random.seed(42)
X, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=42)
outliers = np.random.uniform(low=-10, high=10, size=(10, 2))
X = np.vstack([X, outliers])

# Define models
models = {
    "Isolation Forest": IsolationForest(contamination=0.1),
}

# Fit and predict for each model
plt.figure(figsize=(15, 12))
plot_num = 1
for model_name, model in models.items():
    plt.subplot(2, 2, plot_num)
    model.fit(X)
    y_pred = model.predict(X)
    
    plt.title(model_name)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    
    plot_num += 1

plt.tight_layout()

# Save the plot to the specified location
plt.savefig('/scratch/f006dg0/stellargraph_ecai_demos/Anomaly_Detection_TODO/IsolationForest_plot.png')