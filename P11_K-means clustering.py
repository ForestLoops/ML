import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Load and standardize data
X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# KMeans clustering
k = 3
model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
clusters = model.labels_
inertia = model.inertia_
silhouette = silhouette_score(X_scaled, clusters)

# Evaluation
print(f"K-Means Results:\nClusters: {k}\nInertia: {inertia:.2f}\nSilhouette Score: {silhouette:.2f}")
print(f"ARI: {adjusted_rand_score(y, clusters):.2f}")
print(f"NMI: {normalized_mutual_info_score(y, clusters):.2f}")
print("Cluster distribution:")
for c, n in zip(*np.unique(clusters, return_counts=True)):
    print(f"Cluster {c}: {n} samples")

# PCA for 2D visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)
centroids_pca = PCA(n_components=2).fit(X_scaled).transform(model.cluster_centers_)

plt.figure(figsize=(8, 5))
for i in range(k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('K-Means Clustering (PCA Reduced)')
plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.legend(); plt.grid(True); plt.show()

# Elbow method
inertias = [KMeans(n_clusters=i, random_state=42, n_init=10).fit(X_scaled).inertia_ for i in range(1, 11)]
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters'); plt.ylabel('Inertia'); plt.grid(True); plt.show()