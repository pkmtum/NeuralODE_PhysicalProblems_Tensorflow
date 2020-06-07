"""
Definition of ODENet and Augmented ODENet.
Ported from https://github.com/EmilienDupont/augmented-neural-odes/blob/master/anode/models.py
"""
import time
import datetime
from DoublePendulum import DoublePendulum
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
from tfdiffeq import odeint_adjoint as odeint


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['densenet', 'odenet'], default='odenet')
args = parser.parse_args()

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
PLOT_DIR = 'plots/double_pendulum/' + args.network + '/'


class modelFunc(tf.keras.Model):
    """Converts a standard tf.keras.Model to a model compatible with odeint."""

    def __init__(self, model):
        super(modelFunc, self).__init__()
        self.model = model

    def call(self, t, x):
        return self.model(tf.expand_dims(x, axis=0))[0]


def visualize_double_pendulum(model, epoch=0):
    dt = 0.01
    x0 = tf.stack([1.01, 4., 0., 0.])
    model_func = modelFunc(model)
    x_t = odeint(model_func, x0, tf.linspace(0., 10., int(10./dt))).numpy()

    ref_pendulum = DoublePendulum(theta1=1.01, theta1_dt=5.)
    x_t_ref = np.array(ref_pendulum.step(dt=999*0.01, n_steps=1000))
    plt.close()
    plt.scatter(x_t[:, 0], x_t[:, 1], c=tf.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.savefig(PLOT_DIR + 'phase_plot_theta_dp{}.png'.format(epoch))
    plt.close()
    plt.scatter(x_t[:, 2], x_t[:, 3], c=tf.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.savefig(PLOT_DIR + 'phase_plot_theta_dt_dp{}.png'.format(epoch))
    plt.close()

    x1 = np.sin(x_t[:, 0])
    y1 = -np.cos(x_t[:, 0])

    x2 = np.sin(x_t[:, 1]) + x1
    y2 = -np.cos(x_t[:, 1]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*0.01))
        return line, time_text

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(x_t)),
                                  interval=0.01*1000, blit=True, init_func=init)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(PLOT_DIR + 'dp{}.mp4'.format(epoch), writer=writer)


def load_dataset_double_pendulum():
    x_train = np.load('experiments/datasets/double_pendulum_x_train.npy')
    y_train = np.load('experiments/datasets/double_pendulum_y_train.npy')
    x_val = np.load('experiments/datasets/double_pendulum_x_val.npy')
    y_val = np.load('experiments/datasets/double_pendulum_y_val.npy')
    return x_train, y_train, x_val, y_val


def create_dataset_double_pendulum(N, samples_per_series=1024):
    pendulum = DoublePendulum(theta1=1.01)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
        y_t = np.array([pendulum.call(0., x_t[i]) for i in range(len(x_t))])

    for i in range(50):
        pendulum = DoublePendulum(theta1=np.random.random()*np.pi, theta2=np.random.random()*np.pi)
        delta_t = 0.01
        with tf.device('/gpu:0'):
            x_t_i = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
            y_t_i = np.array([pendulum.call(0., x_t_i[i]) for i in range(len(x_t_i))])
        x_t = np.concatenate([x_t_i, x_t], axis=0)
        y_t = np.concatenate([y_t_i, y_t], axis=0)

    pendulum = DoublePendulum(theta1=1.01, theta1_dt=5.)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t_val = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
        y_t_val = np.array([pendulum.call(0., x_t_val[i]) for i in range(len(x_t_val))])

    np.save('experiments/datasets/double_pendulum_x_train.npy', x_t)
    np.save('experiments/datasets/double_pendulum_y_train.npy', y_t)
    np.save('experiments/datasets/double_pendulum_x_val.npy', x_t_val)
    np.save('experiments/datasets/double_pendulum_y_val.npy', y_t_val)
    return x_t, y_t, x_t_val, y_t_val


class ODEFunc(tf.keras.Model):
    def __init__(self, hidden_dim, augment_dim=0, time_dependent=True, **kwargs):
        dynamic = kwargs.pop('dynamic', True)
        super(ODEFunc, self).__init__(**kwargs, dynamic=dynamic)
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.dense1 = Dense(hidden_dim, activation='relu', kernel_regularizer=l2(0.00001))
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

    # @tf.function
    def call(self, t, x):
        self.nfe.assign_add(1.)
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = tf.ones([x.shape[0], 1], dtype=t.dtype) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = tf.concat([t_vec, x], axis=-1)
            # Shape (batch_size, hidden_dim)
            out = self.dense1(t_and_x)
        else:
            out = self.dense1(x)
        return out


class ODEBlock(tf.keras.Model):

    def __init__(self, odefunc, tol=1e-3, solver='dopri5', **kwargs):
        """
        Solves ODE defined by odefunc.
        # Arguments:
            odefunc : ODEFunc instance or Conv2dODEFunc instance
                Function defining dynamics of system.
            is_conv : bool
                If True, treats odefunc as a convolutional model.
            tol : float
                Error tolerance.
            solver: ODE solver. Defaults to DOPRI5.
        """
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.tol = tol
        self.method = solver
        self.channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        if solver == "dopri5":
            self.options = {'max_num_steps': MAX_NUM_STEPS}
        else:
            self.options = None

    def call(self, x, training=None, eval_times=None, **kwargs):
        """
        Solves ODE starting from x.
        # Arguments:
            x: Tensor. Shape (batch_size, self.odefunc.data_dim)
        # Returns:
            Output tensor of forward pass.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe.assign(0.)

        if eval_times is None:
            integration_time = tf.cast(tf.linspace(0., 1., 2), dtype=x.dtype)
        else:
            integration_time = tf.cast(eval_times, x.dtype)
        if self.odefunc.augment_dim > 0:
            # Add augmentation
            aug = tf.zeros([x.shape[0], self.odefunc.augment_dim], dtype=x.dtype)
            # Shape (batch_size, data_dim + augment_dim)
            x_aug = tf.concat([x, aug], axis=-1)
        else:
            x_aug = x
        out = odeint(self.odefunc, x_aug, integration_time,
                     rtol=self.tol, atol=self.tol, method=self.method,
                     options=self.options)
        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    @property
    def nbe(self):
        return self.odefunc.nbe

    @nbe.setter
    def nbe(self, value):
        self.odefunc.nbe = value

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

    def compute_output_shape(self, input_shape):
        if self.odefunc.augment_dim > 0:
            channels = input_shape[1]
            channels += self.odefunc.augment_dim
            output_shape = tf.TensorShape([input_shape[0], channels])
        else:
            output_shape = input_shape
        return output_shape


class ODENet(tf.keras.Model):

    def __init__(self, hidden_dim, output_dim):
        super(ODENet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, kernel_regularizer=l2(0.00001))
        odefunc = ODEFunc(hidden_dim+0, augment_dim=0)
        self.odeblock = ODEBlock(odefunc)
        self.dense2 = Dense(output_dim, activation=None, kernel_regularizer=l2(0.00001))

    def call(self, x):
        out = self.dense1(x)
        out = self.odeblock(out)
        out = self.dense2(out)
        return out


N = 1200
# x_train, y_train, x_val, y_val = create_dataset_double_pendulum(N)
x_train, y_train, x_val, y_val = load_dataset_double_pendulum()

if args.network == 'odenet':
    model = ODENet(hidden_dim=8, output_dim=y_train.shape[-1])
    adam = Adam(lr=3e-3)
elif args.network == 'densenet':
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(
        0.00001), input_shape=(x_train.shape[-1], )))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.00001)))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.00001)))
    model.add(Dense(y_train.shape[-1], activation=None, kernel_regularizer=l2(0.00001)))

    adam = Adam(lr=3e-4, clipnorm=1)

model.compile(optimizer=adam, loss='mse', metrics=['mae'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)

print(x_train.shape)
for epoch in range(10):
    model.fit(x_train, y_train,
              epochs=10*(epoch+1),
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback],
              initial_epoch=10*epoch)

    print(model.summary())
    visualize_double_pendulum(model, epoch)
