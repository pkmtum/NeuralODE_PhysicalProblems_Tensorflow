"""
Airplane system experiment, longitudinal and lateral motion, Dense-Net.
"""
import argparse
import datetime
import json
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
parser.add_argument('--system', type=str, default='mass_spring_damper')
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--dataset_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--synthetic_derivative', type=bool, default=False,
                    help='Use numerical derivatives? (default: False)')
args = parser.parse_args()

with open('experiments/environments.json') as json_file:
    environment_configs = json.load(json_file)

config = environment_configs[args.system]

PLOT_DIR = 'plots/' + config['name'] + '/densenet/'
TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

makedirs(PLOT_DIR)

if not os.path.isfile('experiments/datasets/' + config['name'] + '_x_train.npy'):
    x_train, y_train, x_val, y_val = create_dataset(n_series=51, config=config)
x_train, y_train, x_val, y_val = load_dataset(config)
if args.synthetic_derivative:
    y_train = np.gradient(x_train)[1] / config['delta_t']

x_train = np.reshape(x_train, (-1, config['dof']))
y_train = np.reshape(y_train, (-1, config['dof']))

c = np.arange(len(x_train))
np.random.shuffle(c)
x_train = x_train[c[::int(100/args.dataset_size)]]
y_train = y_train[c[::int(100/args.dataset_size)]]

model = Sequential()
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(1e-6),
                input_shape=(config['dof'],)))
model.add(Dense(64, 'sigmoid', kernel_regularizer=l2(1e-6)))
model.add(Dense(config['dof']))

adam = Adam(lr=args.lr)
model.compile(optimizer=adam, loss='mse', metrics=['mae', my_mse])

log_dir = ("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
           + '_' + config['name'] + '_densenet_' + str(args.dataset_size))
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
              validation_data=(x_val.reshape(-1, config['dof']),
                               y_val.reshape(-1, config['dof'])),
              callbacks=[tensorboard_callback, learning_rate_callback],
              initial_epoch=epoch_multi*epoch)

    for i, trajectory in enumerate(config['validation']):
        print(trajectory, model.evaluate(x_val[i], y_val[i], verbose=0))

    visualize(modelFunc(model), x_val, PLOT_DIR, TIME_OF_RUN, args, config,
              ode_model=True, epoch=(epoch+1)*epoch_multi)
print(model.summary())
