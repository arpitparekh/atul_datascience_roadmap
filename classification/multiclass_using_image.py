folder_path = "/home/arpit-parekh/Downloads/bascom_projects/atul_datascience_roadmap/weather_pictures"

# pip install opencv-python

import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# os.walk()


def read_images(folder_path):
  image_list = []    # list to store all the images
  label_list = []    # list to store all the labels
  counter = 0

  for root,dirs,files in os.walk(folder_path):
    for file in files:
      if file.endswith('.jpg'):
        image_path = os.path.join(root,file)
        imageArray = cv2.imread(image_path)
        print(image_path)
        print(1120-counter)
        counter += 1
        imageArray = cv2.resize(imageArray, (100,100)).flatten()
        image_list.append(imageArray)
        label_list.append(root.split('/')[-1])


  return image_list, label_list

image_list, label_list = read_images(folder_path)
print(image_list)
print(label_list)

X = np.array(image_list)
y = np.array(label_list)

print(X.shape)
print(y.shape)

current_time = time.time()
print(current_time)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(time.time() - current_time)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# use image to check

imageArray =  cv2.imread("/home/arpit-parekh/Downloads/raindrops-illuminated-by-street-lamp-260nw-2274651743.jpg")
imageArray = cv2.resize(imageArray, (100,100)).flatten()
print(clf.predict(np.array([imageArray])))

