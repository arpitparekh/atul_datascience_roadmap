import tensorflow as tf

X = tf.constant(12)
Y = tf.constant(13)

# X = tf.constant([12,3,4,5])
# X = tf.constant([[12,3,4,5],[6,7,8,9]])

Z = tf.add(X,Y)
print(Z.numpy())

# linear regression // prediction
# tensorflow is used for optimization

# y = mx + c

# m and c (gradient decent)

# mse (error function)
