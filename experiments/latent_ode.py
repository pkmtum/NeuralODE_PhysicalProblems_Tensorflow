"""
Airplane system experiment, longitudinal and lateral motion, NODE-e2e.
"""
import argparse
import datetime
import json
import os
import time
import numpy as np
import tensorflow as tf
from utils import create_dataset, load_dataset, makedirs, RunningAverageMeter, visualize

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--system', type=str, default='mass_spring_damper')
parser.add_argument('--method', type=str, choices=['dopri5', 'midpoint'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1001)
parser.add_argument('--dataset_size', type=int, choices=[100], default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_time', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=500)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32')
args = parser.parse_args()

tf.keras.backend.set_floatx(args.dtype)

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint

with open('experiments/environments.json') as json_file:
    environment_configs = json.load(json_file)

config = environment_configs[args.system]

PLOT_DIR = 'plots/' + config['name'] + '/learnedode/'
TIME_OF_RUN = datetime.datetime.now()
device = 'gpu:' + str(args.gpu) if len(gpus) else 'cpu:0'

t = tf.range(0., args.data_size) * config['delta_t']
if args.dtype == 'float64':
    t = tf.cast(t, tf.float64)

if not os.path.isfile('experiments/datasets/' + config['name'] + '_x_train.npy'):
    x_train, _, x_val, _ = create_dataset(n_series=51, config=config)
x_train, _, x_val, _ = load_dataset(config)
x_train = x_train.astype(args.dtype)
x_val = x_val.astype(args.dtype)

makedirs(PLOT_DIR)


def get_batch():
    # pick random data series
    n = np.random.choice(
        np.arange(x_train.shape[0],
                  dtype=np.int64), args.batch_size,
        replace=True)
    # pick random starting time
    s = np.random.choice(
        np.arange(args.data_size - args.batch_time,
                  dtype=np.int64), args.batch_size,
        replace=False)

    batch_x0 = tf.convert_to_tensor(x_train[n, s])  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_x = tf.stack([x_train[n, s + i] for i in range(args.batch_time)], axis=0)  # (T, M, D)
    return batch_x0, batch_t, batch_x


class ODEFunc(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ODEFunc, self).__init__(**kwargs)

        self.x1 = tf.keras.layers.Dense(config['hidden_dim'], activation='sigmoid')
        self.x2 = tf.keras.layers.Dense(config['hidden_dim'], activation='sigmoid')
        self.y = tf.keras.layers.Dense(config['dof'])
        self.nfe = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, t, y):
        self.nfe.assign_add(1.)
        x = self.x1(y)
        x = self.x2(x)
        y = self.y(x)
        return y


if __name__ == '__main__':

    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    with tf.device(device):
        func = ODEFunc()
        lr = tf.Variable(args.lr)
        optimizer = tf.keras.optimizers.Adam(lr, clipvalue=0.5)

        for itr in range(1, args.niters + 1):
            with tf.GradientTape() as tape:
                batch_x0, batch_t, batch_x = get_batch()
                pred_x = odeint(func, batch_x0, batch_t, method=args.method)  # (T, B, D)
                ex_loss = tf.reduce_sum(tf.math.square(pred_x - batch_x), axis=-1)
                loss = tf.reduce_mean(ex_loss)
                # Uncomment to try FFT-based loss
                # fft_batch = tf.signal.rfft(tf.transpose(batch_x, [0, 2, 1]))
                # fft_pred = tf.signal.rfft(tf.transpose(pred_x, [0, 2, 1]))
                # fft_diff = fft_batch-fft_pred
                # loss = tf.reduce_mean(tf.math.abs(fft_diff))
                weights = [v for v in func.trainable_variables if 'bias' not in v.name]
                l2_loss = tf.add_n([tf.reduce_sum(tf.math.square(v)) for v in weights]) * 1e-6
                loss = loss + l2_loss

            nfe = func.nfe.numpy()
            func.nfe.assign(0.)
            grads = tape.gradient(loss, func.trainable_variables)
            nbe = func.nfe.numpy()
            func.nfe.assign(0.)
            print('NFE: {}, NBE: {}'.format(nfe, nbe))

            grad_vars = zip(grads, func.trainable_variables)
            optimizer.apply_gradients(grad_vars)
            time_meter.update(time.time() - end)
            loss_meter.update(loss.numpy())
            if itr % args.test_freq == 0:
                print('Iter {:04d} | Seconds/batch {:,.4f}'.format(itr, time_meter.avg))
                visualize(func, np.array(x_val), PLOT_DIR, TIME_OF_RUN, args,
                          config, ode_model=True, epoch=itr)
            if itr == int(args.niters*0.5):  # learning rate decay
                optimizer.lr = optimizer.lr * 0.1
            elif itr == int(args.niters*0.7):
                optimizer.lr = optimizer.lr * 0.1
            elif itr == int(args.niters*0.9):
                optimizer.lr = optimizer.lr * 0.1
            end = time.time()
