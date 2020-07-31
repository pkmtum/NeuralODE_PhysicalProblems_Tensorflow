"""
Provides functions that are useful across all model architectures.
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tfdiffeq import odeint
from SinglePendulum import SinglePendulum

class Lambda(tf.keras.Model):

    def call(self, t, y):
        return tf.stack([y[:, 1], -9.81*tf.math.sin(y[:, 0])], axis=-1)


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
        x_train: np.ndarray, shape=(n_series, samples_per_series, 2)
        y_train: np.ndarray, shape=(n_series, samples_per_series, 2)
        x_val: np.ndarray, shape=(n_series, samples_per_series, 2)
        y_val: np.ndarray, shape=(n_series, samples_per_series, 2)

    """
    delta_t = 0.01
    theta0_in = np.random.random((n_series//2))
    theta0_out = np.random.random((n_series-n_series//2)) + np.pi - 1
    theta0 = np.concatenate([theta0_in, theta0_out])
    pendulum = SinglePendulum(theta=theta0, theta_dt=tf.zeros_like(theta0)) # compute all trajectories at once
    with tf.device('/gpu:0'):
        x_train = pendulum.step(dt=(samples_per_series-1)*delta_t, n_steps=samples_per_series)
        y_train = np.array(pendulum.call(0., x_train))
    x_train = np.transpose(x_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    pendulum = SinglePendulum(theta=1.5, theta_dt=0.5)
    with tf.device('/gpu:0'):
        x_val = pendulum.step(dt=(samples_per_series-1)*delta_t, n_steps=samples_per_series)
        y_val = np.array(pendulum.call(0., x_val))
    x_val = np.reshape(x_val, (1, -1, 2))
    y_val = np.reshape(y_val, (1, -1, 2))

    if save_to_disk:
        np.save('experiments/datasets/single_pendulum_x_train.npy', x_train)
        np.save('experiments/datasets/single_pendulum_y_train.npy', y_train)
        np.save('experiments/datasets/single_pendulum_x_val.npy', x_val)
        np.save('experiments/datasets/single_pendulum_y_val.npy', y_val)
    return x_train, y_train, x_val, y_val


def load_dataset():
    x_train = np.load('experiments/datasets/single_pendulum_x_train.npy').astype(np.float32)
    y_train = np.load('experiments/datasets/single_pendulum_y_train.npy').astype(np.float32)
    x_val = np.load('experiments/datasets/single_pendulum_x_val.npy').astype(np.float32)
    y_val = np.load('experiments/datasets/single_pendulum_y_val.npy').astype(np.float32)
    return x_train, y_train, x_val, y_val


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def my_mse(y_true, y_pred):
    """Needed because Keras' MSE implementation includes L2 penalty """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


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


def relative_phase_error(x_pred, x_val):
    """Computes the relative phase error of x_pred w.r.t. x_true.
    This is done by finding the locations of the zero crossings in both signals,
    then corresponding crossings are compared to each other.

    # Arguments:
        x_pred: numpy.ndarray shape=(n_datapoints, 2) - predicted time series
        x_true: numpy.ndarray shape=(n_datapoints, 2) - reference time series
    """
    ref_crossings = zero_crossings(x_val[:, 0])
    pred_crossings = zero_crossings(x_pred[:, 0])
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error = t_ref/t_pred - 1
    if len(pred_crossings) < len(ref_crossings) - 2:
        phase_error = np.nan
    return phase_error


def trajectory_error(x_pred, x_val):
    return np.mean(np.abs(x_pred - x_val))


def visualize(model, x_val, PLOT_DIR, TIME_OF_RUN, args, ode_model=True, latent=False, epoch=0):
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
    x_val = x_val.reshape(-1, 2)
    dt = 0.01
    t = tf.linspace(0., 10., int(10./dt)+1)
    # Compute the predicted trajectories
    if ode_model:
        x0 = tf.stack([[1.5, .5]])
        x_t = odeint(model, x0, t, rtol=1e-5, atol=1e-5).numpy()[:, 0]
    else: # is LSTM
        x_t = np.zeros_like(x_val[0])
        x_t[0] = x_val[0]
        for i in range(1, len(t)):
            x_t[1:i+1] = model(0., np.expand_dims(x_t, axis=0))[0, :i]

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
    ax_traj.plot(t.numpy(), x_val[:, 0], t.numpy(), x_val[:, 1], 'g-')
    ax_traj.plot(t.numpy(), x_t[:, 0], '--', t.numpy(), x_t[:, 1], 'b--')
    ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj.set_ylim(-6, 6)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('theta')
    ax_phase.set_ylabel('theta_dt')
    ax_phase.plot(x_val[:, 0], x_val[:, 1], 'g--')
    ax_phase.plot(x_t[:, 0], x_t[:, 1], 'b--')
    ax_phase.set_xlim(-6, 6)
    ax_phase.set_ylim(-6, 6)

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('theta')
    ax_vecfield.set_ylabel('theta_dt')

    steps = 61
    y, x = np.mgrid[-6:6:complex(0, steps), -6:6:complex(0, steps)]
    ref_func = Lambda()
    dydt_ref = ref_func(0., np.stack([x, y], -1).reshape(steps * steps, 2)).numpy()
    mag_ref = 1e-8+np.linalg.norm(dydt_ref, axis=-1).reshape(steps, steps)
    dydt_ref = dydt_ref.reshape(steps, steps, 2)

    if ode_model: # is Dense-Net or NODE-Net or NODE-e2e
        dydt = model(0., np.stack([x, y], -1).reshape(steps * steps, 2)).numpy()
    else: # is LSTM
        # Compute artificial x_dot by numerically diffentiating:
        # x_dot \approx (x_{t+1}-x_t)/dt
        yt_1 = model(0., np.stack([x, y], -1).reshape(steps * steps, 1, 2))[:, 0]
        dydt = (np.array(yt_1)-np.stack([x, y], -1).reshape(steps * steps, 2)) / dt

    dydt_abs = dydt.reshape(steps, steps, 2)
    dydt_unit = dydt_abs / np.linalg.norm(dydt_abs, axis=-1, keepdims=True)

    ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1], color="black")
    ax_vecfield.set_xlim(-6, 6)
    ax_vecfield.set_ylim(-6, 6)

    ax_vec_error_abs.cla()
    ax_vec_error_abs.set_title('Abs. error of thetadot')
    ax_vec_error_abs.set_xlabel('theta')
    ax_vec_error_abs.set_ylabel('theta_dt')

    abs_dif = np.clip(np.linalg.norm(dydt_abs-dydt_ref, axis=-1), 0., 3.)
    c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
    plt.colorbar(c1, ax=ax_vec_error_abs)

    ax_vec_error_abs.set_xlim(-6, 6)
    ax_vec_error_abs.set_ylim(-6, 6)

    ax_vec_error_rel.cla()
    ax_vec_error_rel.set_title('Rel. error of thetadot')
    ax_vec_error_rel.set_xlabel('theta')
    ax_vec_error_rel.set_ylabel('theta_dt')

    rel_dif = np.clip(abs_dif / mag_ref, 0., 1.)
    c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
    plt.colorbar(c2, ax=ax_vec_error_rel)

    ax_vec_error_rel.set_xlim(-6, 6)
    ax_vec_error_rel.set_ylim(-6, 6)

    ax_energy.cla()
    ax_energy.set_title('Total Energy')
    ax_energy.set_xlabel('t')
    ax_energy.plot(np.arange(0., x_t.shape[0]*dt, dt), np.array([total_energy(x_) for x_ in x_t]))
    ax_energy.plot(np.arange(0., x_t.shape[0]*dt, dt), total_energy(x_t))

    fig.tight_layout()
    plt.savefig(PLOT_DIR + '/{:03d}'.format(epoch))
    plt.close()

    # Compute Metrics
    energy_drift_interp = relative_energy_drift(x_t, x_val)
    phase_error_interp = relative_phase_error(x_t, x_val)
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

    # Print Jacobian
    if ode_model:
        np.set_printoptions(suppress=True, precision=4, linewidth=150)
        # The first Jacobian is averaged over 100 randomly sampled points from U(-1, 1)
        jac = tf.zeros((2, 2))
        for i in range(100):
            with tf.GradientTape(persistent=True) as g:
                x = (2 * tf.random.uniform((1, 2)) - 1)
                g.watch(x)
                y = model(0, x)
            jac = jac + g.jacobian(y, x)[0, :, 0]
        print(jac.numpy()/100)

        with tf.GradientTape(persistent=True) as g:
            x = tf.zeros([1, 2])
            g.watch(x)
            y = model(0, x)
        print(g.jacobian(y, x)[0, :, 0])

    if args.create_video:
        x1 = np.sin(x_t[:, 0])
        y1 = -np.cos(x_t[:, 0])

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def animate(i):
            thisx = [0, x1[i]]
            thisy = [0, y1[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*0.01))
            return line, time_text
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text


        ani = animation.FuncAnimation(fig, animate, range(1, len(x1)),
                                      interval=dt*len(x1), blit=True, init_func=init)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2400)
        ani.save(PLOT_DIR + 'sp{}.mp4'.format(epoch), writer=writer)

        x1 = np.sin(x_val[:, 0])
        y1 = -np.cos(x_val[:, 0])

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        ani = animation.FuncAnimation(fig, animate, range(1, len(x_t)),
                                      interval=dt*len(x_t), blit=True, init_func=init)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2400)
        ani.save(PLOT_DIR + 'sp_ref.mp4', writer=writer)
        plt.close()


def zero_crossings(x):
    """Find indices of zeros crossings"""
    return np.array(np.where(np.diff(np.sign(x)))[0])
