"""
Airplane system experiment, DenseNet and NODE-Net.
"""
import argparse
import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
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
parser.add_argument('--adjoint', type=bool, default=False)
parser.add_argument('--viz', action='store_true')
parser.set_defaults(viz=True)
args = parser.parse_args()

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
PLOT_DIR = 'plots/airplane_lat_long/odenet/'
TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if args.viz:
    makedirs(PLOT_DIR)

class ODEFunc(tf.keras.Model):
    def __init__(self, hidden_dim, augment_dim=0, time_dependent=True, **kwargs):
        dynamic = kwargs.pop('dynamic', True)
        super(ODEFunc, self).__init__(**kwargs, dynamic=dynamic)
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.dense1 = Dense(hidden_dim, activation='sigmoid', kernel_regularizer=l2(0.00001))
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)
        self.t_vec = tf.ones(dtype=tf.float32, shape=[32, 1])

    # @tf.function
    def call(self, t, x):
        self.nfe.assign_add(1.)
        if self.time_dependent:
            # Shape (batch_size, 1)
            if x.shape[0] != self.t_vec.shape[0]:
                self.t_vec = tf.ones(dtype=tf.float32, shape=[x.shape[0], 1])
            t_vec = tf.multiply(self.t_vec, t)
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
        self.output_dim = output_dim
        self.dense1 = Dense(hidden_dim, 'relu', kernel_regularizer=l2(0.00001))
        odefunc = ODEFunc(hidden_dim+0, augment_dim=0)
        self.odeblock = ODEBlock(odefunc, solver='dopri5')
        self.dense2 = Dense(output_dim, kernel_regularizer=l2(0.00001))

    def call(self, x):
        out = self.dense1(x)
        out = self.odeblock(out)
        out = self.dense2(out)
        return out

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.output_dim])

if not os.path.isfile('experiments/datasets/airplane_lat_long_x_train.npy'):
    x_train, y_train, x_val, y_val = create_dataset(n_series=51)
x_train, y_train, x_val, y_val = load_dataset()
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
x_train = np.reshape(x_train, (-1, 4))
y_train = np.reshape(y_train, (-1, 4))

x_val_extrap, y_val_extrap = x_val[0], y_val[0]
x_val_interp, y_val_interp = x_val[1], y_val[1]

x_val = np.reshape(x_val, (-1, 4))
y_val = np.reshape(y_val, (-1, 4))

c = np.arange(len(x_train))
np.random.shuffle(c)
x_train = x_train[c[::int(100/args.dataset_size)]]
y_train = y_train[c[::int(100/args.dataset_size)]]

model = ODENet(hidden_dim=64, output_dim=8)

adam = Adam(lr=args.lr)
model.compile(optimizer=adam, loss='mse', metrics=['mae', my_mse])
log_dir = ("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
           + '|msp|odenet|' + str(args.dataset_size))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=0)

epoch_multi = 5

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
        x = tf.zeros(shape=(1, 4))
        g.watch(x)
        y = model(x)
    jacobian = g.jacobian(y, x)
    print(jacobian)
