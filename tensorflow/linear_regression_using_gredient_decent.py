# linear regression using gradient decent
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
Y = np.array([2,4,6,8,10,12,14,16,18,20],dtype=float) + np.random.randn(10)

# y = mx + c

# m and c (gradient decent)
M = tf.Variable(0.0)
C = tf.Variable(0.0)

def linear_regression():
  Y_pred = M*X + C
  return Y_pred

def loss():
  Y_pred = linear_regression()
  mse = tf.reduce_mean(tf.square(Y - Y_pred))
  return mse

model = tf._optimizers.SGD(learning_rate=0.001)

epoch = 1000

loss_array = []

for i in range(epoch):
  with tf.GradientTape() as tape:
    mse = loss()
    gradients = tape.gradient(mse,[M,C])

    """
    d(mse)/d(M) = 2*X*(Y_pred - Y)
    d(mse)/d(C) = 2*(Y_pred - Y)q
    """

    model.apply_gradients(zip(gradients,[M,C]))

    print(f"M: {M.numpy()}, C: {C.numpy()}, MSE: {mse.numpy()}")
    loss_array.append(mse.numpy())

# plot the data
pred = linear_regression()
plt.scatter(X,Y)
plt.plot(X,pred,color='red')
plt.show()

# plot the loss
plt.plot(loss_array)
plt.show()


