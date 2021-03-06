"""
Airplane system experiment, longitudinal and lateral motion, Dense-Net.
"""
import argparse
import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils import create_dataset, load_dataset, makedirs, modelFunc, my_mse, visualize

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--dataset_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--synthetic_derivative', type=bool, default=False,
                    help='Numerical derivatives? (default: False)')
args = parser.parse_args()

PLOT_DIR = 'plots/airplane_lat_long/densenet/'
TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

makedirs(PLOT_DIR)

if not os.path.isfile('experiments/datasets/airplane_lat_long_x_train.npy'):
    x_train, y_train, x_val, y_val = create_dataset(n_series=51)
x_train, y_train, x_val, y_val = load_dataset()
if args.synthetic_derivative:
    y_train = np.gradient(x_train)[1] / 0.01

data_dim = x_train.shape[-1]
x_train = np.reshape(x_train, (-1, data_dim))
y_train = np.reshape(y_train, (-1, data_dim))

x_val_extrap, y_val_extrap = x_val[0], y_val[0]
x_val_interp, y_val_interp = x_val[1], y_val[1]

x_val = np.reshape(x_val, (-1, data_dim))
y_val = np.reshape(y_val, (-1, data_dim))

c = np.arange(len(x_train))
np.random.shuffle(c)
x_train = x_train[c[::int(100/args.dataset_size)]]
y_train = y_train[c[::int(100/args.dataset_size)]]

model = Sequential()
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(1e-6),
                input_shape=(data_dim,)))
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(1e-6)))
model.add(Dense(data_dim))

adam = Adam(lr=args.lr)
model.compile(optimizer=adam, loss='mse', metrics=['mae', my_mse])

log_dir = ("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
           + '_airll_densenet_' + str(args.dataset_size))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)

epoch_multi = 10  # we can afford to train regular NN's for longer

def lr_scheduler(epoch):
    if epoch < 5*epoch_multi:
        return args.lr
    if epoch < 8*epoch_multi:
        return args.lr * 0.1
    if epoch < 10*epoch_multi:
        return args.lr * 0.01
    return args.lr * 0.001

learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

for epoch in range(10):
    model.fit(x_train, y_train,
              epochs=epoch_multi*(epoch+1),
              batch_size=args.batch_size,
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback, learning_rate_callback],
              initial_epoch=epoch_multi*epoch)

    print('Extrapolation:', model.evaluate(x_val_extrap, y_val_extrap, verbose=0))
    print('Interpolation:', model.evaluate(x_val_interp, y_val_interp, verbose=0))
    visualize(modelFunc(model), x_val, PLOT_DIR, TIME_OF_RUN, args,
                ode_model=True, epoch=(epoch+1)*epoch_multi)
print(model.summary())