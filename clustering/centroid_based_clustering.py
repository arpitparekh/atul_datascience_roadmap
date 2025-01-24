# centroid based clustering
# k-means
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Generate synthetic data

X,y = make_blobs(n_samples=300, centers=3,cluster_std=5 ,random_state=42)

# now create a clusters from random data
model = KMeans(n_clusters=4,random_state=42)
model.fit(X)

predicted =  model.labels_
print(predicted)

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],color="red")
plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=predicted,cmap='plasma')
# plot the centrod also
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color="black")


plt.show()
