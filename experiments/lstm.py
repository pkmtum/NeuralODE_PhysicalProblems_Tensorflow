"""
Airplane experiment lateral and longitudinal motion, LSTM.
"""
import argparse
import datetime
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils import create_dataset, load_dataset, lr_scheduler, makedirs, modelFunc, my_mse, visualize

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])

parser = argparse.ArgumentParser()
parser.add_argument('--system', type=str, default='mass_spring_damper')
parser.add_argument('--dataset_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_time', type=int, default=16)
args = parser.parse_args()

with open('experiments/environments.json') as json_file:
    environment_configs = json.load(json_file)

config = environment_configs[args.system]

PLOT_DIR = 'plots/' + config['name'] + '/lstm/'
TIME_OF_RUN = datetime.datetime.now()

makedirs(PLOT_DIR)

if not os.path.isfile('experiments/datasets/' + config['name'] + '_x_train.npy'):
    x_train, _, x_val, _ = create_dataset(n_series=51, config=config)
x_train, _, x_val, _ = load_dataset(config)
# Offset input/output by one timestep
x_val_ref = x_val
y_train = x_train[:, 1:]
x_train = x_train[:, :-1]
y_val = x_val[:, 1:]
x_val = x_val[:, :-1]


class TrainDatagen(tf.keras.utils.Sequence):

    def __len__(self):
        return x_train.shape[0]*x_train.shape[1]//args.batch_size

    def __getitem__(self, index):
        # pick random data series
        n = np.random.choice(
            np.arange(y_train.shape[0],
                      dtype=np.int64), args.batch_size,
            replace=True)
        # pick random starting time
        s = np.random.choice(
            np.arange(y_train.shape[1] - args.batch_time,
                      dtype=np.int64), args.batch_size,
            replace=True)
        # (B, T, D)
        batch_x = np.stack([x_train[n, s+i] for i in range(args.batch_time)], axis=1)
        batch_y = np.stack([y_train[n, s+i] for i in range(args.batch_time)], axis=1)
        return batch_x, batch_y


model = Sequential()
model.add(LSTM(config['hidden_dim'], kernel_regularizer=l2(1e-5), return_sequences=True,
               input_shape=(None, config['dof'],)))
model.add(Dense(config['dof'], kernel_regularizer=l2(1e-5)))
adam = Adam(lr=args.lr)

model.compile(loss='mse', optimizer=adam, metrics=['mae', my_mse])

log_dir = ("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
           + '_' + config['name'] + '_lstm_' + str(args.dataset_size) + '_'
           + str(args.batch_time) + '_' + str(args.batch_size))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)

epoch_multi = 10

scheduler = lr_scheduler(epoch_multi, args.lr)

learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

train_datagen = TrainDatagen()
for epoch in range(10):
    model.fit(train_datagen,
              epochs=epoch_multi*(epoch+1),
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback, learning_rate_callback],
              initial_epoch=epoch_multi*epoch)
    visualize(modelFunc(model), x_val_ref, PLOT_DIR, TIME_OF_RUN, args,
              config, ode_model=False, epoch=(epoch+1)*epoch_multi)
print(model.summary())
