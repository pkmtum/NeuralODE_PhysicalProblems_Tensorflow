"""
Airplane system experiment, longitudinal and lateral motion, DenseNet.
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
from utils import create_dataset, load_dataset, makedirs, my_mse, modelFunc, visualize

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--dataset_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--viz', action='store_true')
parser.set_defaults(viz=True)
args = parser.parse_args()


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
PLOT_DIR = 'plots/airplane_lat_long/densenet/'
TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if args.viz:
    makedirs(PLOT_DIR)

if not os.path.isfile('experiments/datasets/airplane_lat_long_x_train.npy'):
    x_train, y_train, x_val, y_val = create_dataset(n_series=51)
x_train, y_train, x_val, y_val = load_dataset()
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
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

l2_penalty = 1e-6# * 0
model = Sequential()
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(l2_penalty),
                input_shape=(data_dim,)))
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(l2_penalty)))
model.add(Dense(data_dim))

adam = Adam(lr=args.lr)
model.compile(optimizer=adam, loss='mse', metrics=['mae', my_mse])
log_dir = ("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
           + '|msp|densenet|' + str(args.dataset_size))
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

    print('extrap:', model.evaluate(x_val_extrap, y_val_extrap))
    print('interp:', model.evaluate(x_val_interp, y_val_interp))
    if args.viz:
        visualize(modelFunc(model), x_val, PLOT_DIR, TIME_OF_RUN, args,
                  ode_model=True, epoch=(epoch+1)*epoch_multi)
    with tf.GradientTape(persistent=True) as g:
        x = tf.zeros(shape=(1, 8))
        g.watch(x)
        y = model(x)
    jac = g.jacobian(y, x)[0, :, 0]
    np.set_printoptions(suppress=True, precision=4)
    print(jac.numpy())
