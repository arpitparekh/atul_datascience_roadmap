path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/clustering/photos_no_class"

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from PIL import Image

# pip install pillow

def readImages(path):
  counter = 0
  list = []

  for root, dirs, files in os.walk(path):
    for file in files:
      if file.endswith(".jpg"):
        image = Image.open(os.path.join(root, file)).convert('RGB')

        image = image.resize((100, 100),Image.Resampling.NEAREST)

        image = np.array(image)/255.0  # for [0,1]
        image = image.flatten()

        list.append(image)

        counter = counter + 1
        print(80-counter)

  return np.array(list)

array =  readImages(path)
# Create a clustering model
clustering = AgglomerativeClustering()
clustering.fit(array)
# Get the cluster labels for each data point
labels = clustering.labels_


def display_cluster_images(imageArray, labels, num_images_per_cluster=4):
    """
    Display multiple images from each cluster

    Parameters:
    - imageArray: Numpy array of flattened images
    - labels: Cluster labels
    - num_images_per_cluster: Number of images to display from each cluster
    """
    unique_clusters = np.unique(labels) # [0, 1, 2, 3, 4]
    num_clusters = len(unique_clusters)  # 5

    plt.figure(figsize=(15, 3*num_clusters))  # manaually

    for cluster in unique_clusters:  # 0 ..4  # 1
        # Find indices of images in this cluster
        cluster_indices = np.where(labels == cluster)[0] # 25 images in cluster 0

        # Select first few images from this cluster
        selected_images = cluster_indices[:num_images_per_cluster]  # 4 images

        for i, img_index in enumerate(selected_images):
            plt.subplot(num_clusters, num_images_per_cluster,
                        cluster*num_images_per_cluster + i + 1)

          # (7,4,5

            plt.imshow(imageArray[img_index].reshape(100, 100, 3))
            plt.title(f"Cluster {cluster}, Image {i+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function
display_cluster_images(array, labels)
