from time import time
import tensorflow as tf
import numpy as np
from tfdiffeq import odeint_adjoint as odeint

class Lambda(tf.keras.Model):

    def __init__(self):
        super(Lambda, self).__init__()
        self.A = tf.constant([[-0.01, -0.49, -0.046, -0.001, 0., 0., 0., 0.],
                              [0.11, 0.0003, 1.14, 0.043, 0., 0., 0., 0.],
                              [-0.11, 0.0003, -1.14, 0.957, 0., 0., 0., 0.],
                              [0.1, 0.0, -15.34, -3.00, 0., 0., 0., 0.],
                              [0., 0., 0., 0., -0.87, 6.47, -0.41, 0.],
                              [0., 0., 0., 0., -1, -0.38, 0, 0.07],
                              [0., 0., 0., 0., 0.91, -18.8, -0.65, 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0.]], dtype=tf.float64)

    def call(self, t, y):
        return tf.matmul(tf.cast(self.A, y.dtype), tf.expand_dims(y, -1))[..., 0]

shape1 = (64, 8)
shape2 = (64, 8)
scale = 0.5213
y0 = tf.random.uniform(shape=shape1, dtype=tf.float64)
y1 = tf.random.uniform(shape=shape1, dtype=tf.float32)
y2 = tf.random.uniform(shape=shape2, dtype=tf.float64)
y3 = tf.random.uniform(shape=shape2, dtype=tf.float64)


t0 = time()
for i in range(10000):
    z = tf.cast(y0, tf.float32)
t1 = time()
for i in range(10000):
    z = tf.cast(y1, tf.float64)
t2 = time()
for i in range(10000):
    z = tf.cast(y0, tf.float16)
t3 = time()
for i in range(10000):
    z = tf.cast(y1, tf.float32)
t4 = time()
print(t1-t0, t2-t1, t3-t2, t4-t3)
