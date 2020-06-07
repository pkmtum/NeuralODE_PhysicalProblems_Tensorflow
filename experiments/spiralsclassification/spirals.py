"""
Definition of ODENet and Augmented ODENet.
Ported from https://github.com/EmilienDupont/augmented-neural-odes/blob/master/anode/models.py
"""
from tfdiffeq import odeint_adjoint as odeint
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, choices=['densenet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=160)
args = parser.parse_args()

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


def create_spirals(N):
    """ Creates a list of points which form a spiral
        # Arguments:
            N: Integers, number of points to be generated per class
        # Output:
            x: list of x-y-coordinates
            y: list of integers, either 0 or 1, signifies which class of point
    """
    np.random.seed(0)
    theta = np.sqrt(np.random.rand(N))*2*np.pi  # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N, 2)

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(N, 2)

    res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
    res_b = np.append(x_b, np.ones((N, 1)), axis=1)
    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    return res[:, :2], res[:, 2]


class ODEFunc(tf.keras.Model):
    def __init__(self, hidden_dim, augment_dim=0, time_dependent=True, **kwargs):
        dynamic = kwargs.pop('dynamic', True)
        super(ODEFunc, self).__init__(**kwargs, dynamic=dynamic)
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.dense1 = Dense(hidden_dim, activation='relu')
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

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
        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        odefunc = ODEFunc(hidden_dim+0, augment_dim=0)
        self.odeblock = ODEBlock(odefunc)
        self.dense2 = Dense(output_dim, activation='sigmoid')

    def call(self, x):
        out = self.dense1(x)
        out = self.odeblock(out)
        out = self.dense2(out)
        return out


if args.network == 'odenet':
    model = ODENet(hidden_dim=12, output_dim=1)
    adam = Adam(lr=3e-2)
elif args.network == 'densenet':
    model = Sequential()
    model.add(Dense(12, input_shape=(2,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=3e-2)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)

N = 1200
x, y = create_spirals(N)
x_train, x_val = x[:int(x.shape[0]*0.8)], x[int(x.shape[0]*0.8):]
y_train, y_val = y[:int(x.shape[0]*0.8)], y[int(x.shape[0]*0.8):]
for epoch in range(15):
    model.fit(x_train, y_train,
              epochs=1*(epoch+1),
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback],
              initial_epoch=1*epoch)
    xp = np.linspace(-20, 20, 20)
    yp = xp
    xv, yv = np.meshgrid(xp, yp)
    xv = np.reshape(xv, (-1))
    yv = np.reshape(yv, (-1))
    preds = model.predict(np.vstack([xv, yv]).T)
    plt.contourf(xp, yp, np.reshape(preds, (xp.shape[0], yp.shape[0])))
    plt.scatter((x[y.argsort()])[:N, 0], (x[y.argsort()])[:N, 1])
    plt.scatter((x[y.argsort()])[N:, 0], (x[y.argsort()])[N:, 1])
    plt.savefig('plots/spiral/' + args.network + '/pred{}.png'.format(epoch))
    plt.close()
print(model.summary())
