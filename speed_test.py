from time import time
import tensorflow as tf
import numpy as np
from tfdiffeq import odeint_adjoint as odeint

class Lambda(tf.keras.Model):

    def __init__(self):
        super(Lambda, self).__init__()
        self.A = tf.Variable([[-0.01, -0.49, -0.046, -0.001],
                              [0.11, 0.0003, 1.14, 0.043],
                              [-0.11, 0.0003, -1.14, 0.957],
                              [0.1, 0.0, -15.34, -3.00]], trainable=True)

    def call(self, t, y):
        x = tf.matmul(tf.cast(self.A, y.dtype), tf.expand_dims(y, -1))[..., 0]
        return x
class Lambdafunc(tf.keras.Model):

    def __init__(self):
        super(Lambdafunc, self).__init__()
        self.A = tf.Variable([[-0.01, -0.49, -0.046, -0.001],
                              [0.11, 0.0003, 1.14, 0.043],
                              [-0.11, 0.0003, -1.14, 0.957],
                              [0.1, 0.0, -15.34, -3.00]], trainable=True)
    @tf.function
    def call(self, t, y):
        return tf.matmul(tf.cast(self.A, y.dtype), tf.expand_dims(y, -1))[..., 0]

shape = (4,)
y0 = tf.random.uniform(shape=shape, dtype=tf.float32)
y1 = tf.random.uniform(shape=shape, dtype=tf.float32)
y2 = tf.random.uniform(shape=shape, dtype=tf.float32)
y3 = tf.random.uniform(shape=shape, dtype=tf.float32)
t = tf.linspace(0., 1., 10)


model1 = Lambda()
model2 = Lambdafunc()
t0 = time()
for i in range(10):
    with tf.GradientTape() as tape:
        d = odeint(model1, y0, t)
        loss = tf.reduce_sum(d[-1])
        print('FWD:', time()-t0)
    grads = tape.gradient(loss, model1.trainable_variables)
    print('BWD:', time()-t0)
# model.fit(y0, y1)

t1 = time()
for i in range(10):
    with tf.GradientTape() as tape:
        d = odeint(model2, y0, t)
        loss = tf.reduce_sum(d[-1])
        print('FWD:', time()-t1)
    grads = tape.gradient(loss, model2.trainable_variables)
    print('BWD:', time()-t1)
t2 = time()
# for i in range(1, 1000):
#     t = func(0., y0)
#     t = func(0., y1)

t3 = time()
# for i in range(1, 1000):
#     t = model_func(0., z0)
#     t = model_func(0., z1)
t4 = time()
    # k = tf.constant(x)
#     for k_, f_ in zip(k, y):
#         k_.append(f_)
print(t1-t0, t2-t1, t3-t2, t4-t3)
