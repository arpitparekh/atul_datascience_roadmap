# classification using deep learning
# binary classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
Y = iris.target
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# create a model
print(iris.DESCR)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

model = keras.Sequential([

  keras.layers.Dense(4,activation='relu',input_shape=(4,)),  # value<0 -> 0 else value
  keras.layers.Dense(3,activation='softmax')

])

# compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=500)

# evaluate the model
loss,accuracy = model.evaluate(X_test,y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# plot the data
plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.show()
