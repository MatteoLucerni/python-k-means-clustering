import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme()

X, _ = make_blobs(n_samples=1500, centers=4, cluster_std=1.8)

plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

Y = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="viridis")
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200)
plt.show()

ssd = {}

for k in range(1, 10):
    kmeans = KMeans(init="k-means++", n_clusters=k)
    kmeans.fit(X)
    ssd[k] = kmeans.inertia_

plt.plot(list(ssd.keys()), list(ssd.values()), marker="o")
plt.xlabel("N clusters")
plt.ylabel("SSD")
plt.show()

# k = 4
