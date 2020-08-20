"""
Provides functions that are useful across all model architectures.
"""
import os
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
import environments


class modelFunc(tf.keras.Model):
    """Converts a standard tf.keras.Model to a model compatible with odeint."""
    def __init__(self, model):
        super(modelFunc, self).__init__()
        self.model = model

    def call(self, t, x):
        return self.model(x)


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


def create_dataset(n_series, config, n_steps=1001):
    """Creates a dataset with n_series data series that are each simulated for
    n_steps time steps. The timesteps are delta_t seconds apart.
    # Arguments:
        n_series: int, number of series to create
        config: json, config of the system
        n_steps: int, number of samples per series
        save_dataset: bool, whether to save the dataset to disk
    # Returns:
        x_train: np.ndarray, shape=(n_series, n_steps, dof)
        y_train: np.ndarray, shape=(n_series, n_steps, dof)
        x_val: np.ndarray, shape=(n_series, n_steps, dof)
        y_val: np.ndarray, shape=(n_series, n_steps, dof)
    """
    # Get the correct reference model from the environments package
    model_func = getattr(getattr(environments, config['ref_model']), config['ref_model'])

    # Sample initial conditions and compute true trajectories
    x0 = (2 * tf.random.uniform((n_series, config['dof'])) - 1)
    model = model_func(x0=x0)
    with tf.device('/gpu:0'):
        t = tf.linspace(0., (n_steps-1)*config['delta_t'], n_steps)
        x_train = odeint(model, x0, t)
        y_train = np.array(model.call(0., x_train))
    x_train = np.transpose(x_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    x_val = []
    y_val = []
    # Extrapolation
    if 'extrapolation' in config['validation']:
        x0 = tf.constant(config['validation']['extrapolation'])
        model = model_func(x0=x0)
        with tf.device('/gpu:0'):
            t = tf.linspace(0., (n_steps-1)*config['delta_t'], n_steps)
            x_val.append(odeint(model, x0, t))
            y_val.append(np.array(model.call(0., x_val[-1])))
    # Interpolation
    if 'interpolation' in config['validation']:
        x0 = tf.constant(config['validation']['interpolation'])
        model = model_func(x0=x0)
        with tf.device('/gpu:0'):
            t = tf.linspace(0., (n_steps-1)*config['delta_t'], n_steps)
            x_val.append(odeint(model, x0, t))
            y_val.append(np.array(model.call(0., x_val[-1])))
    x_val = np.stack(x_val)
    y_val = np.stack(y_val)

    np.save('experiments/datasets/' + config['name'] + '_x_train.npy', x_train)
    np.save('experiments/datasets/' + config['name'] + '_y_train.npy', y_train)
    np.save('experiments/datasets/' + config['name'] + '_x_val.npy', x_val)
    np.save('experiments/datasets/' + config['name'] + '_y_val.npy', y_val)
    return x_train, y_train, x_val, y_val


def load_dataset(config):
    x_train = np.load('experiments/datasets/' + config['name'] + '_x_train.npy')
    x_train = x_train.astype(np.float32)
    y_train = np.load('experiments/datasets/' + config['name'] + '_y_train.npy')
    y_train = y_train.astype(np.float32)
    x_val = np.load('experiments/datasets/' + config['name'] + '_x_val.npy')
    x_val = x_val.astype(np.float32)
    y_val = np.load('experiments/datasets/' + config['name'] + '_y_val.npy')
    y_val = y_val.astype(np.float32)
    return x_train, y_train, x_val, y_val


def lr_scheduler(epoch_multi, lr):
    def scheduler(epoch):
        if epoch < 5*epoch_multi:
            return lr
        if epoch < 8*epoch_multi:
            return lr * 0.1
        if epoch < 10*epoch_multi:
            return lr * 0.01
        return lr * 0.001
    return scheduler


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def my_mse(y_true, y_pred):
    """Needed because Keras' MSE implementation includes L2 penalty """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def predict_time_series(model, x_val, config, ode_model, is_mdn=False):
    """Uses the model to predict the validation trajectories contained in x_val
    # Arguments:
        model: tf.keras.Model - the model to evaluate
        x_val np.ndarray - the validation trajectories
        config: dict - configuration of the system as defined in .json
        ode_model: bool - whether the model outputs the next timestep (false)
                          or the derivative at the current timestep (true)
        is_mdn: bool - whether the model is a mixture density model
    """
    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    x_t = np.zeros_like(x_val)
    # Compute the predicted trajectories
    if ode_model:
        x0 = tf.convert_to_tensor(x_val[:, 0])
        x_t = odeint(model, x0, t, method='dopri5', options={'max_num_steps': 1000}).numpy()
        x_t = tf.transpose(x_t, [1, 0, 2])
    else:  # LSTM model
        x_t[:, 0] = x_val[:, 0]
        # Always injects the entire time series because keras is slow
        # when using varying series lengths and the future timesteps
        # don't affect the predictions before it anyways.
        for i in range(1, len(t)):
            x_t[:, i:i+1] = model(0., x_t)[:, i-1:i]
        if is_mdn:
            import mdn
            for i in range(1, len(t)):
                pred = model(0., x_t)[:, i-1:i]
                x_t[i:i+1] = mdn.sample_from_output(pred.numpy()[:, 0], 2, 5, temp=1.)
    return x_t


def predict_vector_field(model, config, ode_model, axes=[0, 1]):
    """Evaluates the model on a plane.
    # Arguments:
        model: tf.keras.Model - the model to evaluate
        config: dict - configuration of the system as defined in .json
        ode_model: bool - whether the model outputs the next timestep (false)
                          or the derivative at the current timestep (true)
        axes: list - list of two ints that define the two axes of the plane
    """
    steps = 81
    lim_l = config['viz_options']['vector_field']['axes_range'][0]
    lim_u = config['viz_options']['vector_field']['axes_range'][1]

    y, x = np.mgrid[lim_l:lim_u:complex(0, steps), lim_l:lim_u:complex(0, steps)]
    zeros = np.zeros((steps, steps, config['dof']))
    zeros[..., axes[0]] = x
    zeros[..., axes[1]] = y
    ref_func = getattr(getattr(environments, config['ref_model']), config['ref_model'])()
    dydt_ref = ref_func(0., zeros.reshape(steps * steps, config['dof'])).numpy()
    mag_ref = 1e-8+np.linalg.norm(dydt_ref, axis=-1).reshape(steps, steps)
    dydt_ref = dydt_ref.reshape(steps, steps, config['dof'])

    if ode_model:  # is Dense-Net or NODE-Net or NODE-e2e
        dydt = model(0., zeros.reshape(steps * steps, config['dof'])).numpy()
    else:  # is LSTM
        # Compute artificial x_dot by numerically diffentiating:
        # x_dot \approx (x_{t+1}-x_t)/dt
        yt_1 = model(0., zeros.reshape(steps * steps, 1, config['dof']))[:, 0].numpy()
        dydt = (yt_1-zeros.reshape(steps * steps, config['dof'])) / config['delta_t']

    dydt_abs = dydt.reshape(steps, steps, config['dof'])
    dydt_unit = dydt_abs / np.linalg.norm(dydt_abs, axis=-1, keepdims=True)

    # Clip for better visualization
    abs_dif = np.clip(np.linalg.norm(dydt_abs-dydt_ref, axis=-1), 0., 3.)
    rel_dif = np.clip(abs_dif / mag_ref, 0., 1.)
    return dydt_unit, abs_dif, rel_dif


def visualize(model, x_val, PLOT_DIR, TIME_OF_RUN, args, config,
              ode_model=True, epoch=0, is_mdn=False):
    # Predict validation trajectories
    x_t = predict_time_series(model, x_val, config, ode_model, is_mdn)
    dydt_unit, abs_dif, rel_dif = predict_vector_field(model, config, ode_model)

    # Do data-set-specific calculation
    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    log_file_path = (PLOT_DIR + TIME_OF_RUN.strftime("%Y%m%d-%H%M%S") + "results"
                     + str(args.lr) + str(args.dataset_size) + str(args.batch_size)
                     + ".csv")
    # Run the visualize function of the system
    getattr(getattr(environments, config['ref_model']),
            config['ref_model']).visualize(t, x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                           PLOT_DIR, TIME_OF_RUN, log_file_path, epoch)
    # Print Jacobian
    if ode_model:
        np.set_printoptions(suppress=True, precision=4, linewidth=150)
        # The first Jacobian is averaged over 10 points sampled from U(-1, 1)
        jac = tf.zeros((config['dof'], config['dof']))
        for i in range(10):
            with tf.GradientTape(persistent=True) as g:
                x = (2 * tf.random.uniform((1, config['dof'])) - 1)
                g.watch(x)
                y = model(0, x)
            jac = jac + g.jacobian(y, x)[0, :, 0]
        print('Average Jacobian at 10 random points')
        print(jac.numpy()/10)

        with tf.GradientTape(persistent=True) as g:
            x = tf.zeros([1, config['dof']])
            g.watch(x)
            y = model(0, x)
        print('Jacobian at 0')
        print(g.jacobian(y, x)[0, :, 0].numpy())
