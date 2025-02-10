# classification using deep learning
# binary classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# create a model
print(cancer.DESCR)


model = keras.Sequential([

  keras.layers.Dense(30,activation='relu',input_shape=(30,)),  # value<0 -> 0 else value
  keras.layers.Dense(1,activation='sigmoid')

])

# compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100)

# evaluate the model
loss,accuracy = model.evaluate(X_test,y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# plot the data
plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.show()
