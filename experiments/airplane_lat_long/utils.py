"""
Provides functions that are useful across all model architectures.
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tfdiffeq import odeint
from AirplaneLatLong import AirplaneLatLong


class Lambda(tf.keras.Model):

    def __init__(self):
        super(Lambda, self).__init__()
        self.A = tf.constant([[-0.01, -0.49, -0.046, -0.001, 0., 0., 0., 0.],
                              [0.11, 0.0003, 1.14, 0.043, 0., 0., 0., 0.],
                              [-0.11, 0.0003, -1.14, 0.957, 0., 0., 0., 0.],
                              [0.1, 0.0, -15.34, -3.00, 0., 0., 0., 0.],
                              [0., 0., 0., 0., -0.87, 6.47, -0.41, 0.],
                              [0., 0., 0., 0., -1, -0.38, 0, 0.07],
                              [0., 0., 0., 0., 0.91, -18.8, -0.65, 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0.]])

    def call(self, t, y):
        return tf.matmul(tf.cast(self.A, y.dtype), tf.expand_dims(y, -1))[..., 0]

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


def create_dataset(n_series=51, samples_per_series=1001, save_to_disk=True):
    """Creates a dataset with n_series data series that are each simulated for samples_per_series
    time steps. The timesteps are delta_t seconds apart.
    # Arguments:
        n_series: int, number of series to create
        samples_per_series: int, number of samples per series
        save_dataset: bool, whether to save the dataset to disk
    # Returns:
        x_train: np.ndarray, shape=(n_series, samples_per_series, 8)
        y_train: np.ndarray, shape=(n_series, samples_per_series, 8)
        x_val: np.ndarray, shape=(n_series, samples_per_series, 8)
        y_val: np.ndarray, shape=(n_series, samples_per_series, 8)

    """
    delta_t = 0.1
    x0 = (2 * tf.random.uniform((n_series, 8)) - 1)
    airplane = AirplaneLatLong(x0=x0)  # compute all trajectories at once
    with tf.device('/gpu:0'):
        x_train = airplane.step(dt=(samples_per_series-1)*delta_t, n_steps=samples_per_series)
        y_train = np.array(airplane.call(0., x_train))
    x_train = np.transpose(x_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    x_val = []
    y_val = []
    # Extrapolation
    airplane = AirplaneLatLong(x0=tf.constant([1.5, 1.5, 1., .5, 1.5, 1., 1.5, 1.5]))
    with tf.device('/gpu:0'):
        x_val.append(airplane.step(dt=(samples_per_series-1)*delta_t, n_steps=samples_per_series))
        y_val.append(np.array(airplane.call(0., x_val[-1])))
    # Interpolation
    airplane = AirplaneLatLong(x0=tf.constant([.5, .5, .5, .5, .5, .5, .5, .5]))
    with tf.device('/gpu:0'):
        x_val.append(airplane.step(dt=(samples_per_series-1)*delta_t, n_steps=samples_per_series))
        y_val.append(np.array(airplane.call(0., x_val[-1])))
    x_val = np.stack(x_val)
    y_val = np.stack(y_val)

    if save_to_disk:
        np.save('experiments/datasets/airplane_lat_long_x_train.npy', x_train)
        np.save('experiments/datasets/airplane_lat_long_y_train.npy', y_train)
        np.save('experiments/datasets/airplane_lat_long_x_val.npy', x_val)
        np.save('experiments/datasets/airplane_lat_long_y_val.npy', y_val)
    return x_train, y_train, x_val, y_val


def load_dataset():
    x_train = np.load('experiments/datasets/airplane_lat_long_x_train.npy').astype(np.float32)
    y_train = np.load('experiments/datasets/airplane_lat_long_y_train.npy').astype(np.float32)
    x_val = np.load('experiments/datasets/airplane_lat_long_x_val.npy').astype(np.float32)
    y_val = np.load('experiments/datasets/airplane_lat_long_y_val.npy').astype(np.float32)
    return x_train, y_train, x_val, y_val


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def my_mse(y_true, y_pred):
    """Needed because Keras' MSE implementation includes L2 penalty """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def relative_phase_error(x_pred, x_val):
    """Computes the relative phase error of x_pred w.r.t. x_true.
    This is done by finding the locations of the zero crossings in both signals,
    then corresponding crossings are compared to each other.
    # Arguments:
        x_pred: numpy.ndarray shape=(n_datapoints, 8) - predicted time series
        x_true: numpy.ndarray shape=(n_datapoints, 8) - reference time series
    """
    # long period
    ref_crossings = zero_crossings(x_val[:, 0])
    pred_crossings = zero_crossings(x_pred[:, 0])
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error_lp = t_ref/t_pred - 1
    if len(pred_crossings) < len(ref_crossings) - 2:
        phase_error_lp = np.nan
    # short period
    ref_crossings = zero_crossings(x_val[:, 2])
    pred_crossings = zero_crossings(x_pred[:, 2])
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error_sp = t_ref/t_pred - 1
    if len(pred_crossings) < len(ref_crossings) - 2:
        phase_error_sp = np.nan
    # laterals
    ref_crossings = zero_crossings(x_val[:, 5])
    pred_crossings = zero_crossings(x_pred[:, 5])
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error_lat_r = t_ref/t_pred - 1

    ref_crossings = zero_crossings(x_val[:, 7])
    pred_crossings = zero_crossings(x_pred[:, 7])
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error_lat_p = t_ref/t_pred - 1
    return phase_error_lp, phase_error_sp, phase_error_lat_r, phase_error_lat_p


def trajectory_error(x_pred, x_val):
    return np.mean(np.abs(x_pred[..., :4] - x_val[..., :4]))


def visualize(model, x_val, PLOT_DIR, TIME_OF_RUN, args, ode_model=True, epoch=0, is_mdn=False):
    """Visualize a tf.keras.Model for a 8-dof airplane model.
    # Arguments:
        model: A Keras model, that accepts t and x when called
        x_val: np.ndarray, shape=(1, samples_per_series, 8) or (samples_per_series, 8)
                The reference time series, against which the model will be compared
        PLOT_DIR: Directory to plot in
        TIME_OF_RUN: Time at which the run began
        ode_model: whether the model outputs the derivative of the current step (True),
                   or the value of the next step (False)
        args: input arguments from main script
    """
    data_dim = 8
    x_val = x_val.reshape(2, -1, data_dim)
    dt = 0.1
    t = tf.linspace(0., 100., int(100/dt)+1)
    # Compute the predicted trajectories
    if ode_model:
        x0 = tf.convert_to_tensor(x_val[:, 0])
        x_t = odeint(model, x0, t, rtol=1e-5, atol=1e-5).numpy()
        x_t_extrap = x_t[:, 0]
        x_t_interp = x_t[:, 1]
    else:  # LSTM model
        x_t_extrap = np.zeros_like(x_val[0])
        x_t_extrap[0] = x_val[0, 0]
        x_t_interp = np.zeros_like(x_val[1])
        x_t_interp[0] = x_val[1, 0]
        # Always injects the entire time series because keras is slow when using
        # varying series lengths and the future timesteps don't affect the predictions
        # before it anyways.
        for i in range(1, len(t)):
            x_t_extrap[i:i+1] = model(0., np.expand_dims(x_t_extrap, axis=0))[0, i-1:i]
            x_t_interp[i:i+1] = model(0., np.expand_dims(x_t_interp, axis=0))[0, i-1:i]

    x_t = np.stack([x_t_extrap, x_t_interp], axis=0)
    # Plot the generated trajectories
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax_traj = fig.add_subplot(331, frameon=False)
    ax_phase = fig.add_subplot(332, frameon=False)
    ax_vecfield = fig.add_subplot(333, frameon=False)
    ax_traj_lat = fig.add_subplot(334, frameon=False)
    ax_phase_lat = fig.add_subplot(335, frameon=False)
    ax_vec_error_abs = fig.add_subplot(336, frameon=False)
    ax_vec_error_rel = fig.add_subplot(337, frameon=False)
    ax_3d = fig.add_subplot(338, projection='3d')
    ax_3d_lat = fig.add_subplot(339, projection='3d')

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('V,gamma')
    for i in range(4):
        ax_traj.plot(t.numpy(), x_val[0, :, i], 'g-')
        ax_traj.plot(t.numpy(), x_t[0, :, i], 'b--')
    ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_traj_lat.cla()
    ax_traj_lat.set_title('Trajectories')
    ax_traj_lat.set_xlabel('t')
    ax_traj_lat.set_ylabel('V,gamma')
    for i in range(4, 8):
        ax_traj_lat.plot(t.numpy(), x_val[0, :, i], 'g-')
        ax_traj_lat.plot(t.numpy(), x_t[0, :, i], 'b--')
    ax_traj_lat.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj_lat.set_ylim(-2, 2)
    ax_traj_lat.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait phugoid')
    ax_phase.set_xlabel('V')
    ax_phase.set_ylabel('gamma')
    ax_phase.plot(x_val[0, :, 0], x_val[0, :, 1], 'g-')
    ax_phase.plot(x_t[0, :, 0], x_t[0, :, 1], 'b--')
    ax_phase.plot(x_val[1, :, 0], x_val[1, :, 1], 'g-')
    ax_phase.plot(x_t[1, :, 0], x_t[1, :, 1], 'b--')
    ax_phase.set_xlim(-6, 6)
    ax_phase.set_ylim(-2, 2)


    ax_phase_lat.cla()
    ax_phase_lat.set_title('Phase Portrait dutch')
    ax_phase_lat.set_xlabel('r')
    ax_phase_lat.set_ylabel('beta')
    ax_phase_lat.plot(x_val[0, :, 4], x_val[0, :, 5], 'g-')
    ax_phase_lat.plot(x_t[0, :, 4], x_t[0, :, 5], 'b--')
    ax_phase_lat.plot(x_val[1, :, 4], x_val[1, :, 5], 'g-')
    ax_phase_lat.plot(x_t[1, :, 4], x_t[1, :, 5], 'b--')
    ax_phase_lat.set_xlim(-6, 6)
    ax_phase_lat.set_ylim(-2, 2)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('V')
    ax_vecfield.set_ylabel('gamma')

    steps = 61
    y, x = np.mgrid[-6:6:complex(0, steps), -6:6:complex(0, steps)]
    zeros = tf.zeros_like(x)
    input_grid = np.stack([x, y, zeros, zeros, zeros, zeros, zeros, zeros], -1)
    ref_func = Lambda()
    dydt_ref = ref_func(0., input_grid.reshape(steps * steps, 8)).numpy()
    mag_ref = 1e-8+np.linalg.norm(dydt_ref, axis=-1).reshape(steps, steps)
    dydt_ref = dydt_ref.reshape(steps, steps, data_dim)

    if ode_model:  # is Dense-Net or NODE-Net or NODE-e2e
        dydt = model(0., input_grid.reshape(steps * steps, data_dim)).numpy()
    else:  # is LSTM
        # Compute artificial x_dot by numerically diffentiating:
        # x_dot \approx (x_{t+1}-x_t)/d
        yt_1 = model(0., input_grid.reshape(steps * steps, 1, data_dim))[:, 0]
        dydt = (np.array(yt_1)-input_grid.reshape(steps * steps, data_dim)) / dt

    dydt_abs = dydt.reshape(steps, steps, data_dim)
    dydt_unit = dydt_abs / np.linalg.norm(dydt_abs, axis=-1, keepdims=True)

    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-4, 4)
    ax_vecfield.set_ylim(-2, 2)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of V\', gamma\'')
    ax_vec_error_abs.set_xlabel('V')
    ax_vec_error_abs.set_ylabel('gamma')
    abs_dif = np.clip(np.linalg.norm(dydt_abs-dydt_ref, axis=-1), 0., 3.)
    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of V\', gamma\'')
    ax_vec_error_rel.set_xlabel('V')
    ax_vec_error_rel.set_ylabel('gamma')

    rel_dif = np.clip(abs_dif / mag_ref, 0., 1.)
    c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
    plt.colorbar(c2, ax=ax_vec_error_rel)

    ax_vec_error_rel.set_xlim(-6, 6)
    ax_vec_error_rel.set_ylim(-6, 6)

    ax_3d.cla()
    ax_3d.set_title('3D Trajectory')
    ax_3d.set_xlabel('V')
    ax_3d.set_ylabel('gamma')
    ax_3d.set_zlabel('alpha')
    ax_3d.scatter(x_val[0, :, 0], x_val[0, :, 1], x_val[0, :, 2], c='g', s=4, marker='^')
    ax_3d.scatter(x_t[0, :, 0], x_t[0, :, 1], x_t[0, :, 2], c='b', s=4, marker='o')
    ax_3d.view_init(elev=40., azim=60.)
    ax_3d_lat.cla()
    ax_3d_lat.set_title('3D Trajectory')
    ax_3d_lat.set_xlabel('r')
    ax_3d_lat.set_ylabel('beta')
    ax_3d_lat.set_zlabel('p')
    ax_3d_lat.scatter(x_val[0, :, 4], x_val[0, :, 5], x_val[0, :, 6], c='g', s=4, marker='^')
    ax_3d_lat.scatter(x_t[0, :, 4], x_t[0, :, 5], x_t[0, :, 6], c='b', s=4, marker='o')
    ax_3d_lat.view_init(elev=1., azim=90.)

    fig.tight_layout()
    plt.savefig(PLOT_DIR + '/{:03d}'.format(epoch))
    plt.close()

    # Compute Metrics
    pe_extrap_lp_long, pe_extrap_sp_long, pe_extrap_lat_r, pe_extrap_lat_p = relative_phase_error(x_t[0], x_val[0])
    traj_error_extrap = trajectory_error(x_t[0], x_val[0])

    pe_interp_lp_long, pe_interp_sp_long, pe_interp_lat_r, pe_interp_lat_p = relative_phase_error(x_t[1], x_val[1])
    traj_error_interp = trajectory_error(x_t[1], x_val[1])


    wall_time = (datetime.datetime.now()
                 - datetime.datetime.strptime(TIME_OF_RUN, "%Y%m%d-%H%M%S")).total_seconds()
    string = "{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(
        wall_time, epoch,
        pe_interp_lp_long, pe_interp_sp_long,
        pe_extrap_lp_long, pe_extrap_sp_long,
        pe_interp_lat_r, pe_extrap_lat_r,
        pe_interp_lat_p, pe_extrap_lat_p,
        traj_error_interp, traj_error_extrap)

    file_path = (PLOT_DIR + TIME_OF_RUN + "results"
                 + str(args.lr) + str(args.dataset_size) + str(args.batch_size)
                 + ".csv")
    if not os.path.isfile(file_path):
        title_string = ("wall_time,epoch,"
                        + "phase_error_interp_lp_long,phase_error_interp_sp_long,"
                        + "phase_error_extrap_lp_long,phase_error_extrap_sp_long,"
                        + "phase_error_interp_lat_r,phase_error_extrap_lat_r,"
                        + "phase_error_interp_lat_p,phase_error_extrap_lat_p,"
                        + "traj_err_interp, traj_err_extrap\n")
        fd = open(file_path, 'a')
        fd.write(title_string)
        fd.close()
    fd = open(file_path, 'a')
    fd.write(string)
    fd.close()

    # Print Jacobian
    if ode_model:
        np.set_printoptions(suppress=True, precision=4, linewidth=150)
        # The first Jacobian is averaged over 100 randomly sampled points from U(-1, 1)
        jac = tf.zeros((8, 8))
        for i in range(100):
            with tf.GradientTape(persistent=True) as g:
                x = (2 * tf.random.uniform((1, 8)) - 1)
                g.watch(x)
                y = model(0, x)
            jac = jac + g.jacobian(y, x)[0, :, 0]
        print(jac.numpy()/100)

        with tf.GradientTape(persistent=True) as g:
            x = tf.zeros([1, 8])
            g.watch(x)
            y = model(0, x)
        print(g.jacobian(y, x)[0, :, 0])


def zero_crossings(x):
    """Find indices of zeros crossings"""
    return np.array(np.where(np.diff(np.sign(x)))[0])
