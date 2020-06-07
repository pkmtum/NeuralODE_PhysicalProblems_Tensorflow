import tensorflow as tf
from tfdiffeq import odeint_adjoint as odeint
import numpy as np
import time

t = tf.linspace(0., 10., 1001)

class Lambda(tf.keras.Model):
    def call(self, t, y):
        return tf.stack([y[:,1], -9.81*tf.math.sin(y[:,0])], axis=-1)
train_y = np.load('experiments/datasets/single_pendulum_x_train.npy').astype(np.float32)
x_val = tf.convert_to_tensor(np.load('experiments/datasets/single_pendulum_x_val.npy').astype(np.float32).reshape(-1, 2))
y_val = tf.convert_to_tensor(np.load('experiments/datasets/single_pendulum_y_val.npy').astype(np.float32).reshape(-1, 2))

data_size = 1001
batch_time = 2
batch_size = 32

def get_batch():
    # pick random data series
    n = np.random.choice(
        np.arange(train_y.shape[0],
                  dtype=np.int64), batch_size,
        replace=False)
    # pick random starting time
    s = np.random.choice(
        np.arange(data_size - batch_time,
                  dtype=np.int64), batch_size,
        replace=False)

    temp_y = train_y# val_y.numpy()
    batch_y0 = tf.convert_to_tensor(temp_y[n, s])  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = tf.stack([temp_y[n, s + i] for i in range(batch_time)], axis=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def gen_batch():
    while True:
        # pick random data series
        n = np.random.choice(
            np.arange(train_y.shape[0],
                      dtype=np.int64), batch_size,
            replace=False)
        # pick random starting time
        s = np.random.choice(
            np.arange(data_size - batch_time,
                      dtype=np.int64), batch_size,
            replace=False)

        temp_y = train_y# val_y.numpy()
        batch_y0 = tf.convert_to_tensor(temp_y[n, s])  # (M, D)
        batch_t = t[:batch_time]  # (T)
        batch_y = tf.stack([temp_y[n, s + i] for i in range(batch_time)], axis=0)  # (T, M, D)
        yield batch_y0, batch_t, batch_y

ds_series = tf.data.Dataset.from_generator(
    gen_batch,
    output_types=(tf.float32, tf.float32, tf.float32),
    output_shapes=((batch_size, 2), (batch_time,), (batch_time, batch_size, 2)))

ds_gen = iter(ds_series.prefetch(tf.data.experimental.AUTOTUNE))
t0 = time.time()

for i in range(1000):
    x,y,z = get_batch()
    time.sleep(0.002)
t1 = time.time()
for i in range(1000):
    x,y,z = next(ds_gen)
    time.sleep(0.002)
t2 = time.time()
for i, sample in enumerate(ds_gen):
    time.sleep(0.002)
    if i == 1000: break
t3 = time.time()
print(t1-t0,t2-t1, t3-t2)
