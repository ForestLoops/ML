import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and scale data
X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# Hierarchical clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = model.fit_predict(X_scaled)

# Evaluation
print("Hierarchical Clustering Results:")
print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y, clusters):.3f}")
print(f"Normalized Mutual Info: {normalized_mutual_info_score(y, clusters):.3f}")

# Cluster distribution
for label in np.unique(clusters):
    print(f"Cluster {label}: {(clusters == label).sum()} samples")

# Visualization: PCA Scatter Plot
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(8, 5))
for i in range(3):
    plt.scatter(*X_pca[clusters == i].T, label=f'Cluster {i}')
plt.title('Hierarchical Clustering (PCA Projection)')
plt.xlabel('PC 1'); plt.ylabel('PC 2')
plt.legend(); plt.grid(True); plt.show()

# Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage(X_scaled, method='ward'), orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Samples'); plt.ylabel('Distance')
plt.show()