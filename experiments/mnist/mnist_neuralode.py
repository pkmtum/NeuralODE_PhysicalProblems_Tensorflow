"""
Replicates the original experiments done in
https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py

Adapted from https://github.com/titu1994/tfdiffeq/blob/master/examples/odenet_mnist.py
and https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py

Logs all metrics to logs/fit/* which can be viewed with TensorBoard

Also implements Augmented Neural ODEs from https://arxiv.org/pdf/1904.01681.pdf
"""
import argparse
import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Activation, Add, Conv2D, Dense, Flatten,
                                     GlobalAveragePooling2D, Input, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32')
parser.add_argument('--save', type=str, default='./')
args = parser.parse_args()

MAX_NUM_STEPS = 1000
if args.dtype == 'float64':
    tf.keras.backend.set_floatx('float64')
else:
    tf.keras.backend.set_floatx('float32')

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape.as_list()),
                                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = tf.stack(group_shape)
        inputs = tf.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (tf.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = tf.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = tf.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = tf.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape


tf.keras.utils.get_custom_objects().update({'GroupNormalization': GroupNormalization})


class Conv2dTime(tf.keras.Model):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """

    def __init__(self, num_filters, kernel_size, strides=1, padding='valid', dilation=1,
                 activation=None, bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(Conv2dTime, self).__init__()

        self._layer = Conv2D(
            num_filters, kernel_size=kernel_size, strides=strides,
            padding=padding, dilation_rate=dilation, activation=activation, use_bias=bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer
        )
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        if self.channel_axis == 1:
            # Shape (batch_size, 1, height, width)
            tt = tf.ones_like(x[:, :1, :, :], dtype=t.dtype) * t  # channel dim = 1
        else:
            # Shape (batch_size, height, width, 1)
            tt = tf.ones_like(x[:, :, :, :1], dtype=t.dtype) * t  # channel dim = -1

        ttx = tf.concat([tt, x], axis=self.channel_axis)  # concat at channel dim
        return self._layer(ttx)


class Conv2dODEFunc(tf.keras.Model):

    def __init__(self, num_filters, augment_dim=0,
                 time_dependent=False, activation=None, groups=None, **kwargs):
        """
        Convolutional block modeling the derivative of ODE system.
        # Arguments:
            num_filters : int
                Number of convolutional filters.
            augment_dim: int
                Number of augmentation channels to add. If 0 does not augment ODE.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            activation : string
                One of 'relu' and 'softplus'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODEFunc, self).__init__(**kwargs, dynamic=dynamic)

        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = tf.Variable(0., trainable=False)  # Number of function evaluations
        self.groups = groups if groups is not None else self.num_filters
        # Inits are weird to replicate PyTorch's initialization
        self.kernel_init = VarianceScaling(scale=1/3., distribution='uniform')
        self.bias_init = VarianceScaling(scale=1/3. * 1/9., distribution='uniform')
        if time_dependent:
            self.conv1 = Conv2dTime(self.num_filters,
                                    kernel_size=(3, 3), strides=(1, 1),
                                    padding='same',
                                    kernel_initializer=self.kernel_init,
                                    bias_initializer=self.bias_init)
            self.conv2 = Conv2dTime(self.num_filters,
                                    kernel_size=(3, 3), strides=(1, 1),
                                    padding='same',
                                    kernel_initializer=self.kernel_init,
                                    bias_initializer=self.bias_init)
        else:
            self.conv1 = Conv2D(self.num_filters,
                                kernel_size=(3, 3), strides=(1, 1),
                                padding='same',
                                kernel_initializer=self.kernel_init,
                                bias_initializer=self.bias_init)
            self.conv2 = Conv2D(self.num_filters,
                                kernel_size=(3, 3), strides=(1, 1),
                                padding='same',
                                kernel_initializer=self.kernel_init,
                                bias_initializer=self.bias_init)
        self.norm1 = GroupNormalization(self.groups)
        self.norm2 = GroupNormalization(self.groups)
        self.norm3 = GroupNormalization(self.groups)
        self.activation = Activation(activation)

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Parameters
        ----------
        t : Tensor
            Current time.
        x : Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe.assign_add(1.)
        if self.time_dependent:
            out = self.norm1(x)
            out = self.activation(out)
            out = self.conv1(t, out)
            out = self.norm2(out)
            out = self.activation(out)
            out = self.conv2(t, out)
            out = self.norm3(out)
        else:
            out = self.norm1(x)
            out = self.activation(out)
            out = self.conv1(out)
            out = self.norm2(out)
            out = self.activation(out)
            out = self.conv2(out)
            out = self.norm3(out)
        return out


class ODEBlock(tf.keras.Model):

    def __init__(self, odefunc, is_conv=False, tol=1e-3, solver='dopri5', **kwargs):
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
        dynamic = kwargs.pop('dynamic', True)
        super(ODEBlock, self).__init__(**kwargs, dynamic=dynamic)

        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol
        self.method = solver
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if solver == "dopri5":
            self.options = {'max_num_steps': MAX_NUM_STEPS}
        else:
            self.options = None

    def call(self, x, training=None, eval_times=None, **kwargs):
        """
        Solves ODE starting from x.
        # Arguments:
            x: Tensor. Shape (batch_size, self.odefunc.data_dim)
            eval_times: None or tf.Tensor.
                If None, returns solution of ODE at final time t=1. If tf.Tensor
                then returns full ODE trajectory evaluated at points in eval_times.
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
            if self.is_conv:
                # Add augmentation
                batch_size, height, width, channels = x.shape
                if self.channel_axis == 1:
                    aug = tf.zeros([batch_size, self.odefunc.augment_dim,
                                    height, width], dtype=x.dtype)
                else:
                    aug = tf.zeros([batch_size, height, width,
                                    self.odefunc.augment_dim], dtype=x.dtype)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = tf.concat([x, aug], axis=self.channel_axis)
            else:
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
            if self.is_conv:
                # Add augmentation
                height, width, channels = input_shape[1], input_shape[2], input_shape[3]
                if self.channel_axis == 1:
                    height += self.odefunc.augment_dim
                else:
                    channels += self.odefunc.augment_dim
                output_shape = tf.TensorShape([input_shape[0], height, width, channels])
            else:
                channels = input_shape[1]
                channels += self.odefunc.augment_dim
                output_shape = tf.TensorShape([input_shape[0], channels])
        else:
            output_shape = input_shape
        return output_shape


def ResBlock(**conv_params):
    '''Helper to build a conv -> BN -> relu block'''
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', 1)
    padding = conv_params.setdefault('padding', 'same')
    # kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1.e-4))

    def f(input):
        x = Conv2D(filters, kernel_size=1, strides=strides,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(input)
        conv = GroupNormalization()(input)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(input)
        conv = GroupNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      padding=padding, kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv)
        conv = Add()([conv, x])
        return conv
    return f


class RunningAverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if args.network == 'odenet':
    nfe_ram = RunningAverageMeter()
    nbe_ram = RunningAverageMeter()

# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype(args.dtype)
x_test = x_test.astype(args.dtype)
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Defining the models
# Weight initialization is done this way to reproduce PyTorch's behavior
kernel_init = VarianceScaling(scale=1/3., distribution='uniform')
bias_init = VarianceScaling(scale=1/3.*1/9, distribution='uniform')
dense_bias_init = VarianceScaling(scale=1/3.*1/4, distribution='uniform')


if args.network == 'odenet':
    class Conv2dODENet(tf.keras.Model):
        """Creates a tf.keras.Model which replicates the results of the
        Neural ODE paper on MNIST.
        Parameters
        ----------
        img_size : tuple of ints
            Tuple of (channels, height, width).
        num_filters : int
            Number of convolutional filters.
        output_dim : int
            Dimension of output after hidden layer. Should be 1 for regression or
            num_classes for classification.
        augment_dim: int
            Number of augmentation channels to add. If 0 does not augment ODE.
        time_dependent : bool
            If True adds time as input, making ODE time dependent.
        activation : string
            One of 'relu' and 'softplus'
        tol : float
            Error tolerance.
        return_sequences : bool
            Whether to return the Convolution outputs, or the features after an
            affine transform.
        solver: ODE solver. Defaults to DOPRI5.
        """

        def __init__(self,
                     time_dependent=False, out_kernel_size=(1, 1),
                     activation='relu', out_strides=(1, 1),
                     tol=1e-10, solver='dopri5', **kwargs):

            dynamic = kwargs.pop('dynamic', True)
            super(Conv2dODENet, self).__init__(**kwargs, dynamic=dynamic)

            self.time_dependent = time_dependent
            self.tol = tol
            self.solver = solver
            self.output_kernel = out_kernel_size
            self.output_strides = out_strides
            self.conv1 = Conv2D(64, kernel_size=3, strides=1,
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                input_shape=input_shape)
            self.norm1 = GroupNormalization(32)
            self.act1 = Activation(activation)
            self.padding1 = ZeroPadding2D(padding=((1, 1), (1, 1)))
            self.conv2 = Conv2D(64, kernel_size=4, strides=2,
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init)
            self.norm2 = GroupNormalization(32)
            self.act2 = Activation(activation)
            self.padding2 = ZeroPadding2D(padding=((1, 1), (1, 1)))
            self.conv3 = Conv2D(64, kernel_size=4, strides=2,
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init)

            odefunc1 = Conv2dODEFunc(64, augment_dim=0, groups=32,
                                     time_dependent=self.time_dependent, activation='relu')
            self.odeblock1 = ODEBlock(odefunc1, is_conv=True, tol=tol, solver=self.solver)

            self.norm3 = GroupNormalization(32)
            self.act3 = Activation(activation)
            self.pool = GlobalAveragePooling2D()
            self.flatten = Flatten()

            self.dense1 = Dense(10, kernel_initializer=kernel_init,
                                bias_initializer=dense_bias_init)

        def call(self, x, training=None, return_features=False):
            features = self.conv1(x)
            features = self.norm1(features)
            features = self.act1(features)
            features = self.padding1(features)
            features = self.conv2(features)
            features = self.norm2(features)
            features = self.act2(features)
            features = self.padding2(features)
            features = self.conv3(features)

            features = self.odeblock1(features, training=training)

            features = self.norm3(features)
            features = self.act3(features)
            features = self.pool(features)
            features = self.flatten(features)
            pred = self.dense1(features)

            if return_features:
                return features, pred
            return pred

    model = Conv2dODENet(tol=args.tol, solver='dopri5', time_dependent=True)
elif args.network == 'resnet':
    model_input = Input(shape=input_shape)
    features = Conv2D(filters=64, kernel_size=3, strides=1,
                      kernel_initializer=kernel_init, bias_initializer=bias_init,
                      input_shape=input_shape)(model_input)
    features = GroupNormalization(32)(features)
    features = Activation('relu')(features)
    features = ZeroPadding2D(padding=((1, 1), (1, 1)))(features)
    features = Conv2D(64, kernel_size=4, strides=2, activation='relu',
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(features)
    features = GroupNormalization(32)(features)
    features = Activation('relu')(features)
    features = ZeroPadding2D(padding=((1, 1), (1, 1)))(features)
    features = Conv2D(64, kernel_size=4, strides=2,
                      kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)
    features = ResBlock(filters=64, kernel_size=3, strides=1, activation='relu')(features)

    features = GroupNormalization(32)(features)
    features = Activation('relu')(features)
    features = GlobalAveragePooling2D()(features)
    features = Flatten()(features)
    model_output = Dense(10,
                         kernel_initializer=kernel_init,
                         bias_initializer=dense_bias_init)(features)
    model = Model(inputs=[model_input], outputs=[model_output])

optimizer = SGD(lr=0.1, momentum=0.9)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])


def lr_scheduler(epoch):
    if epoch < 40:
        return 0.1
    elif epoch < 80:
        return 0.01
    elif epoch < 140:
        return 0.001
    return 0.0001


# Callbacks
learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


class NFENBECallback(tf.keras.callbacks.Callback):
    """Callback that keeps track of the average number of function evaluations and
    prints them at the end of an epoch
    """
    def on_train_batch_end(self, batch, logs=None):
        if args.network == 'odenet':
            nfe_ram.update(model.layers[9].nfe.numpy())

    def on_epoch_end(self, epoch, logs=None):
        if args.network == 'odenet':
            print('\navg. NFE: ', nfe_ram.avg)
            nfe_ram.reset()


model.predict(x_train[:1])  # Can't print summary without building the model first
print(model.summary())

model.fit(x_train, y_train,
          epochs=args.nepochs,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, learning_rate_callback, NFENBECallback()],
          batch_size=args.batch_size,
          workers=4,
          use_multiprocessing=True)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
