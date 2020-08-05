"""
Provides functions that are useful across all model architectures.
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        x_train = model.step(dt=(n_steps-1)*config['delta_t'], n_steps=n_steps)
        y_train = np.array(model.call(0., x_train))
    x_train = np.transpose(x_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    x_val = []
    y_val = []
    # Extrapolation
    if 'extrapolation' in config['validation']:
        model = model_func(x0=tf.constant(config['validation']['extrapolation']))
        with tf.device('/gpu:0'):
            x_val.append(model.step(dt=(n_steps-1)*config['delta_t'], n_steps=n_steps))
            y_val.append(np.array(model.call(0., x_val[-1])))
    # Interpolation
    if 'interpolation' in config['validation']:
        model = model_func(x0=tf.constant(config['validation']['interpolation']))
        with tf.device('/gpu:0'):
            x_val.append(model.step(dt=(n_steps-1)*config['delta_t'], n_steps=n_steps))
            y_val.append(np.array(model.call(0., x_val[-1])))
    x_val = np.stack(x_val)
    y_val = np.stack(y_val)

    np.save('experiments/datasets/' + config['name'] + '_x_train.npy', x_train)
    np.save('experiments/datasets/' + config['name'] + '_y_train.npy', y_train)
    np.save('experiments/datasets/' + config['name'] + '_x_val.npy', x_val)
    np.save('experiments/datasets/' + config['name'] + '_y_val.npy', y_val)
    return x_train, y_train, x_val, y_val


def load_dataset(config):
    x_train = np.load('experiments/datasets/' + config['name'] + '_x_train.npy').astype(np.float32)
    y_train = np.load('experiments/datasets/' + config['name'] + '_y_train.npy').astype(np.float32)
    x_val = np.load('experiments/datasets/' + config['name'] + '_x_val.npy').astype(np.float32)
    y_val = np.load('experiments/datasets/' + config['name'] + '_y_val.npy').astype(np.float32)
    return x_train, y_train, x_val, y_val


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def my_mse(y_true, y_pred):
    """Needed because Keras' MSE implementation includes L2 penalty """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def relative_phase_error(x_pred, x_val, check=True):
    """Computes the relative phase error of x_pred w.r.t. x_true.
    Finds the locations of the zero crossings in both signals,
    then compares corresponding crossings to each other.
    # Arguments:
        x_pred: numpy.ndarray shape=(n_datapoints) - predicted time series
        x_true: numpy.ndarray shape=(n_datapoints) - reference time series
        check: bool - set phase error to NaN if the crossings are too different
    """
    ref_crossings = zero_crossings(x_val)
    pred_crossings = zero_crossings(x_pred)
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error = t_ref/t_pred - 1
    if check and len(pred_crossings) < len(ref_crossings) - 2:
        phase_error = np.nan
    return phase_error


def trajectory_error(x_pred, x_val):
    return np.mean(np.abs(x_pred - x_val))


def predict_time_series(model, x_val, config, ode_model, is_mdn):
    """Uses the model to predict the validation trajectories contained in x_val
    """
    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    x_t = np.zeros_like(x_val)
    # Compute the predicted trajectories
    if ode_model:
        x0 = tf.convert_to_tensor(x_val[:, 0])
        x_t = odeint(model, x0, t, rtol=1e-5, atol=1e-5).numpy()
        x_t = tf.transpose(x_t, [1, 0, 2])
    else:  # LSTM model
        x_t[:, 0] = x_val[:, 0]
        # Always injects the entire time series because keras is slow when using
        # varying series lengths and the future timesteps don't affect the predictions
        # before it anyways.
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
    """
    steps = 61
    y, x = np.mgrid[-6:6:complex(0, steps), -6:6:complex(0, steps)]
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
        yt_1 = model(0., zeros.reshape(steps * steps, 1, config['dof']))[:, 0]
        dydt = (np.array(yt_1)-zeros.reshape(steps * steps, config['dof'])) / config['delta_t']

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
    if config['name'] == 'airplane_lat_long':
        visualize_airplane_lat_long(model,
                                    x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                    PLOT_DIR, TIME_OF_RUN, args, config, epoch)
    elif config['name'] == 'airplane_long':
        visualize_airplane_long(model,
                                x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                PLOT_DIR, TIME_OF_RUN, args, config, epoch)
    elif config['name'] == 'mass_spring_damper':
        visualize_mass_spring_damper(model,
                                     x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                     PLOT_DIR, TIME_OF_RUN, args, config, epoch)
    elif config['name'] == 'single_pendulum':
        visualize_single_pendulum(model,
                                  x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                  PLOT_DIR, TIME_OF_RUN, args, config, epoch)

    # Print Jacobian
    if ode_model:
        np.set_printoptions(suppress=True, precision=4, linewidth=150)
        # The first Jacobian is averaged over 100 points sampled from U(-1, 1)
        jac = tf.zeros((config['dof'], config['dof']))
        for i in range(100):
            with tf.GradientTape(persistent=True) as g:
                x = (2 * tf.random.uniform((1, config['dof'])) - 1)
                g.watch(x)
                y = model(0, x)
            jac = jac + g.jacobian(y, x)[0, :, 0]
        print(jac.numpy()/100)

        with tf.GradientTape(persistent=True) as g:
            x = tf.zeros([1, config['dof']])
            g.watch(x)
            y = model(0, x)
        print(g.jacobian(y, x)[0, :, 0])



def visualize_airplane_lat_long(model,
                                x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                PLOT_DIR, TIME_OF_RUN, args, config, epoch=0):
    """Visualize a tf.keras.Model for a 8-dof airplane model.
    # Arguments:
        model: tf.keras.Model - Accepts t and x when called
        x_val: np.ndarray, shape=(2, samples_per_series, 8) -
               The reference time series against which the model will be compared
        PLOT_DIR: str - Directory to plot in
        TIME_OF_RUN: Time at which the run began
        args: Input arguments from main script
        ode_model: whether the model outputs the derivative of the current step (True),
                   or the value of the next step (False)
    """
    t = tf.range(0., x_val.shape[1]) * config['delta_t']
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

    y, x = np.mgrid[-6:6:complex(0, 61), -6:6:complex(0, 61)]
    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-4, 4)
    ax_vecfield.set_ylim(-2, 2)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of V\', gamma\'')
    ax_vec_error_abs.set_xlabel('V')
    ax_vec_error_abs.set_ylabel('gamma')
    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of V\', gamma\'')
    ax_vec_error_rel.set_xlabel('V')
    ax_vec_error_rel.set_ylabel('gamma')

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
    plt.savefig(PLOT_DIR + '{:03d}'.format(epoch))
    plt.close()

    # Compute metrics and save them to csv.
    pe_extrap_lp_long = relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
    pe_extrap_sp_long = relative_phase_error(x_t[0, :, 2], x_val[0, :, 2])
    pe_extrap_lat_r = relative_phase_error(x_t[0, :, 5], x_val[0, :, 5], check=False)
    pe_extrap_lat_p = relative_phase_error(x_t[0, :, 7], x_val[0, :, 7], check=False)
    traj_error_extrap = trajectory_error(x_t[0, :, :4], x_val[0, :, :4])

    pe_interp_lp_long = relative_phase_error(x_t[1, :, 0], x_val[1, :, 0])
    pe_interp_sp_long = relative_phase_error(x_t[1, :, 2], x_val[1, :, 2])
    pe_interp_lat_r = relative_phase_error(x_t[1, :, 5], x_val[1, :, 5], check=False)
    pe_interp_lat_p = relative_phase_error(x_t[1, :, 7], x_val[1, :, 7], check=False)
    traj_error_interp = trajectory_error(x_t[1, :, :4], x_val[1, :, :4])

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


def visualize_airplane_long(model,
                            x_val, x_t, dydt_unit, abs_dif, rel_dif,
                            PLOT_DIR, TIME_OF_RUN, args, config, epoch=0):
    """Visualize a tf.keras.Model for an aircraft model.
    # Arguments:
        model: A Keras model, that accepts t and x when called
        x_val: np.ndarray, shape=(2, samples_per_series, 4)
                The reference time series, against which the model will be compared
        PLOT_DIR: Directory to plot in
        TIME_OF_RUN: Time at which the run began
        ode_model: whether the model outputs the derivative of the current step (True),
                   or the value of the next step (False)
        args: input arguments from main script
    """
    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    # Plot the generated trajectories
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax_traj = fig.add_subplot(231, frameon=False)
    ax_phase = fig.add_subplot(232, frameon=False)
    ax_vecfield = fig.add_subplot(233, frameon=False)
    ax_vec_error_abs = fig.add_subplot(234, frameon=False)
    ax_vec_error_rel = fig.add_subplot(235, frameon=False)
    ax_3d = fig.add_subplot(236, projection='3d')
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

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('V')
    ax_vecfield.set_ylabel('gamma')

    y, x = np.mgrid[-6:6:complex(0, 61), -6:6:complex(0, 61)]
    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-4, 4)
    ax_vecfield.set_ylim(-2, 2)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of V\', gamma\'')
    ax_vec_error_abs.set_xlabel('V')
    ax_vec_error_abs.set_ylabel('gamma')
    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of V\', gamma\'')
    ax_vec_error_rel.set_xlabel('V')
    ax_vec_error_rel.set_ylabel('gamma')

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

    fig.tight_layout()
    plt.savefig(PLOT_DIR + '{:03d}'.format(epoch))
    plt.close()

    # Compute Metrics
    phase_error_extrap_lp = relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
    phase_error_extrap_sp = relative_phase_error(x_t[0, :, 2], x_val[0, :, 2])
    traj_error_extrap = trajectory_error(x_t[0], x_val[0])

    phase_error_interp_lp = relative_phase_error(x_t[1, :, 0], x_val[1, :, 0])
    phase_error_interp_sp = relative_phase_error(x_t[1, :, 2], x_val[1, :, 2])
    traj_error_interp = trajectory_error(x_t[1], x_val[1])

    wall_time = (datetime.datetime.now()
                 - datetime.datetime.strptime(TIME_OF_RUN, "%Y%m%d-%H%M%S")).total_seconds()
    string = "{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(
        wall_time, epoch,
        phase_error_interp_lp, phase_error_interp_sp,
        phase_error_extrap_lp, phase_error_extrap_sp,
        traj_error_interp, traj_error_extrap)

    file_path = (PLOT_DIR + TIME_OF_RUN + "results"
                 + str(args.lr) + str(args.dataset_size) + str(args.batch_size)
                 + ".csv")
    if not os.path.isfile(file_path):
        title_string = ("wall_time,epoch,"
                        + "phase_error_interp_lp,phase_error_interp_sp,"
                        + "phase_error_extrap_lp,phase_error_extrap_sp,"
                        + "traj_err_interp, traj_err_extrap\n")
        fd = open(file_path, 'a')
        fd.write(title_string)
        fd.close()
    fd = open(file_path, 'a')
    fd.write(string)
    fd.close()


def visualize_mass_spring_damper(model,
                                 x_val, x_t, dydt_unit, abs_dif, rel_dif,
                                 PLOT_DIR, TIME_OF_RUN, args, config, epoch=0):
    """Visualize a tf.keras.Model for a single pendulum.
    # Arguments:
        model: A Keras model, that accepts t and x when called
        x_val: np.ndarray, shape=(1, samples_per_series, 2) or (samples_per_series, 2)
                The reference time series, against which the model will be compared
        PLOT_DIR: Directory to plot in
        TIME_OF_RUN: Time at which the run began
        ode_model: whether the model outputs the derivative of the current step (True),
                   or the value of the next step (False)
        args: input arguments from main script
    """
    def total_energy(state, k=1, m=1):
        """Calculates total energy of a mass-spring-damper system given a state."""
        return 0.5*k*state[..., 0]*state[..., 0]+0.5*m*state[..., 1]*state[..., 1]

    def relative_energy_drift(x_pred, x_true, t=-1):
        """Computes the relative energy drift of x_pred w.r.t. x_true
        # Arguments:
            x_pred: numpy.ndarray shape=(n_datapoints, 2) - predicted time series
            x_true: numpy.ndarray shape=(n_datapoints, 2) - reference time series
            t: int, index at which to compute the energy drift, (default: -1)
        """
        energy_pred = total_energy(x_pred[t])
        energy_true = total_energy(x_true[t])
        return (energy_pred-energy_true) / energy_true

    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    # Plot the generated trajectories
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax_traj = fig.add_subplot(231, frameon=False)
    ax_phase = fig.add_subplot(232, frameon=False)
    ax_vecfield = fig.add_subplot(233, frameon=False)
    ax_vec_error_abs = fig.add_subplot(234, frameon=False)
    ax_vec_error_rel = fig.add_subplot(235, frameon=False)
    ax_energy = fig.add_subplot(236, frameon=False)
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), x_val[0, :, 0], t.numpy(), x_val[0, :, 1], 'g-')
    ax_traj.plot(t.numpy(), x_t[0, :, 0], '--', t.numpy(), x_t[0, :, 1], 'b--')
    ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj.set_ylim(-6, 6)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('x_dt')
    ax_phase.plot(x_val[0, :, 0], x_val[0, :, 1], 'g--')
    ax_phase.plot(x_t[0, :, 0], x_t[0, :, 1], 'b--')
    ax_phase.plot(x_val[1, :, 0], x_val[1, :, 1], 'g--')
    ax_phase.plot(x_t[1, :, 0], x_t[1, :, 1], 'b--')
    ax_phase.set_xlim(-6, 6)
    ax_phase.set_ylim(-6, 6)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('x_dt')

    y, x = np.mgrid[-6:6:complex(0, 61), -6:6:complex(0, 61)]
    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-6, 6)
    ax_vecfield.set_ylim(-6, 6)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of xdot')
    ax_vec_error_abs.set_xlabel('x')
    ax_vec_error_abs.set_ylabel('x_dt')

    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of xdot')
    ax_vec_error_rel.set_xlabel('x')
    ax_vec_error_rel.set_ylabel('x_dt')

    c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
    plt.colorbar(c2, ax=ax_vec_error_rel)

    ax_vec_error_rel.set_xlim(-6, 6)
    ax_vec_error_rel.set_ylim(-6, 6)

    ax_energy.cla()
    ax_energy.set_title('Total Energy')
    ax_energy.set_xlabel('t')
    ax_energy.plot(t.numpy(), np.array([total_energy(x_) for x_ in x_t[1]]))

    fig.tight_layout()
    plt.savefig(PLOT_DIR + '{:03d}'.format(epoch))
    plt.close()

    # Compute Metrics
    energy_drift_extrap = relative_energy_drift(x_t[0], x_val[0])
    phase_error_extrap = relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
    traj_error_extrap = trajectory_error(x_t[0], x_val[0])

    energy_drift_interp = relative_energy_drift(x_t[1], x_val[1])
    phase_error_interp = relative_phase_error(x_t[1, :, 0], x_val[1, :, 0])
    traj_error_interp = trajectory_error(x_t[1], x_val[1])

    wall_time = (datetime.datetime.now()
                 - datetime.datetime.strptime(TIME_OF_RUN, "%Y%m%d-%H%M%S")).total_seconds()
    string = "{},{},{},{},{},{},{},{}\n".format(wall_time, epoch,
                                                energy_drift_interp, energy_drift_extrap,
                                                phase_error_interp, phase_error_extrap,
                                                traj_error_interp, traj_error_extrap)
    file_path = (PLOT_DIR + TIME_OF_RUN + "results"
                 + str(args.lr) + str(args.dataset_size) + str(args.batch_size)
                 + ".csv")
    if not os.path.isfile(file_path):
        title_string = ("wall_time,epoch,energy_drift_interp,energy_drift_extrap, phase_error_interp,"
                        + "phase_error_extrap, traj_err_interp, traj_err_extrap\n")
        fd = open(file_path, 'a')
        fd.write(title_string)
        fd.close()
    fd = open(file_path, 'a')
    fd.write(string)
    fd.close()


def visualize_single_pendulum(model,
                              x_val, x_t, dydt_unit, abs_dif, rel_dif,
                              PLOT_DIR, TIME_OF_RUN, args, config, epoch=0):
    """Visualize a tf.keras.Model for a single pendulum.
    # Arguments:
        model: a Keras model
        x_val: np.ndarray, shape=(1, samples_per_series, 2) or (samples_per_series, 2)
                The reference time series, against which the model will be compared
        PLOT_DIR: dir to plot in
        TIME_OF_RUN: time of the run
        ode_model: whether the model outputs the derivative of the current step
        args: input arguments from main script
    """
    def total_energy(state, l=1., g=9.81):
        """Calculates total energy of a pendulum system given a state."""
        return (1-np.cos(state[..., 0]))*l*g+state[..., 1]*state[..., 1]*0.5

    def relative_energy_drift(x_pred, x_true, t=-1):
        """Computes the relative energy drift of x_pred w.r.t. x_true
        # Arguments:
            x_pred: numpy.ndarray shape=(n_datapoints, 2) - predicted time series
            x_true: numpy.ndarray shape=(n_datapoints, 2) - reference time series
            t: int, index at which to compute the energy drift, (default: -1)
        """
        energy_pred = total_energy(x_pred[t])
        energy_true = total_energy(x_true[t])
        return (energy_pred-energy_true) / energy_true

    t = tf.range(0., x_val.shape[1]) * config['delta_t']
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax_traj = fig.add_subplot(231, frameon=False)
    ax_phase = fig.add_subplot(232, frameon=False)
    ax_vecfield = fig.add_subplot(233, frameon=False)
    ax_vec_error_abs = fig.add_subplot(234, frameon=False)
    ax_vec_error_rel = fig.add_subplot(235, frameon=False)
    ax_energy = fig.add_subplot(236, frameon=False)
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), x_val[0, :, 0], t.numpy(), x_val[0, :, 1], 'g-')
    ax_traj.plot(t.numpy(), x_t[0, :, 0], '--', t.numpy(), x_t[0, :, 1], 'b--')
    ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj.set_ylim(-6, 6)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('theta')
    ax_phase.set_ylabel('theta_dt')
    ax_phase.plot(x_val[0, :, 0], x_val[0, :, 1], 'g--')
    ax_phase.plot(x_t[0, :, 0], x_t[0, :, 1], 'b--')
    ax_phase.set_xlim(-6, 6)
    ax_phase.set_ylim(-6, 6)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('theta')
    ax_vecfield.set_ylabel('theta_dt')

    y, x = np.mgrid[-6:6:complex(0, 61), -6:6:complex(0, 61)]
    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-6, 6)
    ax_vecfield.set_ylim(-6, 6)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of thetadot')
    ax_vec_error_abs.set_xlabel('theta')
    ax_vec_error_abs.set_ylabel('theta_dt')

    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of thetadot')
    ax_vec_error_rel.set_xlabel('theta')
    ax_vec_error_rel.set_ylabel('theta_dt')

    c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
    plt.colorbar(c2, ax=ax_vec_error_rel)

    ax_vec_error_rel.set_xlim(-6, 6)
    ax_vec_error_rel.set_ylim(-6, 6)

    ax_energy.cla()
    ax_energy.set_title('Total Energy')
    ax_energy.set_xlabel('t')
    ax_energy.plot(t.numpy(), np.array([total_energy(x_) for x_ in x_t]))
    ax_energy.plot(t.numpy(), total_energy(x_t))

    fig.tight_layout()
    plt.savefig(PLOT_DIR + '{:03d}'.format(epoch))
    plt.close()

    # Compute Metrics
    energy_drift_interp = relative_energy_drift(x_t, x_val)
    phase_error_interp = relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
    traj_err_interp = trajectory_error(x_t, x_val)

    wall_time = (datetime.datetime.now()
                 - datetime.datetime.strptime(TIME_OF_RUN, "%Y%m%d-%H%M%S")).total_seconds()
    string = "{},{},{},{},{}\n".format(wall_time, epoch,
                                       energy_drift_interp,
                                       phase_error_interp,
                                       traj_err_interp)
    file_path = (PLOT_DIR + TIME_OF_RUN + "results"
                 + str(args.lr) + str(args.dataset_size) + str(args.batch_size)
                 + ".csv")
    if not os.path.isfile(file_path):
        title_string = "wall_time,epoch,energy_interp,phase_interp,traj_err_interp\n"
        fd = open(file_path, 'a')
        fd.write(title_string)
        fd.close()
    fd = open(file_path, 'a')
    fd.write(string)
    fd.close()

    # if args.create_video:
    #     x1 = np.sin(x_t[:, 0])
    #     y1 = -np.cos(x_t[:, 0])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    #     ax.set_aspect('equal')
    #     ax.grid()

    #     line, = ax.plot([], [], 'o-', lw=2)
    #     time_template = 'time = %.1fs'
    #     time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    #     def animate(i):
    #         thisx = [0, x1[i]]
    #         thisy = [0, y1[i]]

    #         line.set_data(thisx, thisy)
    #         time_text.set_text(time_template % (i*0.01))
    #         return line, time_text
    #     def init():
    #         line.set_data([], [])
    #         time_text.set_text('')
    #         return line, time_text

    #     ani = animation.FuncAnimation(fig, animate, range(1, len(x1)),
    #                                   interval=dt*len(x1), blit=True, init_func=init)
    #     Writer = animation.writers['ffmpeg']
    #     writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2400)
    #     ani.save(PLOT_DIR + 'sp{}.mp4'.format(epoch), writer=writer)

    #     x1 = np.sin(x_val[:, 0])
    #     y1 = -np.cos(x_val[:, 0])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    #     ax.set_aspect('equal')
    #     ax.grid()

    #     line, = ax.plot([], [], 'o-', lw=2)
    #     time_template = 'time = %.1fs'
    #     time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    #     ani = animation.FuncAnimation(fig, animate, range(1, len(x_t)),
    #                                   interval=dt*len(x_t), blit=True, init_func=init)
    #     Writer = animation.writers['ffmpeg']
    #     writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2400)
    #     ani.save(PLOT_DIR + 'sp_ref.mp4', writer=writer)
    #     plt.close()


def zero_crossings(x):
    """Find indices of zeros crossings"""
    return np.array(np.where(np.diff(np.sign(x)))[0])
