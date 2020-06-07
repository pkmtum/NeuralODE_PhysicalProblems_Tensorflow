import tensorflow as tf
import torch
from time import time

shape = (128, 6)
y0 = tf.random.uniform(shape=shape, dtype=tf.float32)
y1 = tf.random.uniform(shape=shape, dtype=tf.float32)
y2 = tf.random.uniform(shape=shape, dtype=tf.float32)
y3 = tf.random.uniform(shape=shape, dtype=tf.float32)
# x = [y, y1]
x = [y0, y1, y2, y3]
scale = tf.constant(.2, tf.float32)
t0 = time()
for i in range(1, 10000):
    t = tf.math.add_n(x)
t1 = time()
for i in range(1, 10000):
    t = tf.math.accumulate_n(x)
t2 = time()
# t_vec1 = tf.ones([x.shape[0], 1], dtype=tf.float32)
# for i in range(1, 10000):
#     if t_vec1.shape[0] != x.shape[0]:
#         t_vec1 = tf.ones([x.shape[0], 1], dtype=tf.float32)
#     t_vec = tf.multiply(t, t_vec1)
t3 = time()
# for i in range(1, 10000):
#     t_vec = tf.constant(1., shape=[x.shape[0], 1], dtype=tf.float32)
#     t_vec = tf.multiply(t, t_vec)
t4 = time()
    # k = tf.constant(x)
#     for k_, f_ in zip(k, y):
#         k_.append(f_)
# k = tuple(([y0],))

# print(tf.reduce_mean(z))
print(t1-t0, t2-t1, t3-t2, t4-t3)
