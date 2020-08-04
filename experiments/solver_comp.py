"""
Single Pendulum experiment, NODE-e2e.
"""
import argparse
import datetime
import os
import time
import numpy as np
import tensorflow as tf
from utils import create_dataset, load_dataset, makedirs, RunningAverageMeter

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--system', type=str, choices=['single_pendulum'], default='single_pendulum')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'euler', 'midpoint'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1001)
parser.add_argument('--dataset_size', type=int, choices=[100], default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_time', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niters', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--create_video', type=bool, default=False)
parser.set_defaults(viz=True)
args = parser.parse_args()

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint

with open('experiments/environments.json') as json_file:
    environment_configs = json.load(json_file)

config = environment_configs[args.system]

PLOT_DIR = 'plots/' + config['name'] + '/learnedode/'
TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
device = 'gpu:' + str(args.gpu) if len(gpus) else 'cpu:0'

t = tf.linspace(0., 10., args.data_size)

if not os.path.isfile('experiments/datasets/single_pendulum_x_train.npy'):
    x_train, _, x_val, _ = create_dataset(n_series=51, config=config)
x_train, _, x_val, _ = load_dataset(n_series=51, config=config)

x_val = tf.convert_to_tensor(x_val.reshape(-1, 1, 2))

if args.viz:
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

        self.x1 = tf.keras.layers.Dense(8, activation='relu')
        self.x2 = tf.keras.layers.Dense(8, activation='relu')
        self.y = tf.keras.layers.Dense(2)
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

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
        optimizer = tf.keras.optimizers.Adam(lr)

        for itr in range(1, args.niters + 1):
            with tf.GradientTape() as tape:
                batch_x0, batch_t, batch_x = get_batch()
                pred_x = odeint(func, batch_x0, batch_t, method=args.method) # (T, B, D)
                loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(pred_x - batch_x), axis=-1))
                weights = [v for v in func.trainable_variables if 'bias' not in v.name]
                l2_loss = tf.add_n([tf.reduce_sum(tf.math.square(v)) for v in weights]) * 0.00001
                loss = loss + l2_loss

            grads = tape.gradient(loss, func.trainable_variables)
            grad_vars = zip(grads, func.trainable_variables)
            optimizer.apply_gradients(grad_vars)
            time_meter.update(time.time() - end)
            loss_meter.update(loss.numpy())
            t0t = time.time()
            if itr % args.test_freq == 0:
                pred_x = odeint(func, x_val[0], t, method='adams')
                loss = tf.reduce_mean(tf.abs(pred_x - x_val))
                print('Adams', loss.numpy())
                pred_x = odeint(func, x_val[0], t, method='dopri5')
                loss = tf.reduce_mean(tf.abs(pred_x - x_val))
                print('DOPRI5', loss.numpy())
                pred_x = odeint(func, x_val[0], t, method='euler')
                loss = tf.reduce_mean(tf.abs(pred_x - x_val))
                print('Euler', loss.numpy())
                pred_x = odeint(func, x_val[0], t, method='midpoint')
                loss = tf.reduce_mean(tf.abs(pred_x - x_val))
                print('Midpoint', loss.numpy())
                print('Iter {:04d} | Total Loss {:.6f} | Time for batch {:,.4f}'.format(itr,
                                                                                        loss.numpy(),
                                                                                        time_meter.avg))
            if itr == int(args.niters*0.5): # aligns with the other datasets
                optimizer.lr = optimizer.lr * 0.1
            if itr == int(args.niters*0.7):
                optimizer.lr = optimizer.lr * 0.1
            if itr == int(args.niters*0.9):
                optimizer.lr = optimizer.lr * 0.1
            end = time.time()
