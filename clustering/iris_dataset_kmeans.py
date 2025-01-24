import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X  = iris.data
y = iris.target
plt.scatter(X[:, 0], X[:, 1], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# print(iris.DESCR)

# # create a kmeans modelq
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(X)
# # get the cluster centers
# centers = kmeans.cluster_centers_
# labels = kmeans.labels_

# plot the clusters
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
# plt.show()
