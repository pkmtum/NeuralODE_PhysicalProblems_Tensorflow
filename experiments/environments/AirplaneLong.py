import datetime
import os
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
import matplotlib.pyplot as plt
from . import metrics


class AirplaneLong(tf.keras.Model):
    """Class that provides a longitudinal motion model of a Boeing B777.
    State has the following structure: (V*, gamma, alpha, q).T
    """

    def __init__(self, x0=0.):
        """
        # Arguments:
            x0: tf.Tensor, shape=(4,), state at x_0
        """
        super(AirplaneLong, self).__init__()
        self.x = x0
        self.A = tf.constant([[-0.01, -0.49, -0.046, -0.001],
                              [0.11, 0.0003, 1.14, 0.043],
                              [-0.11, -0.0003, -1.14, 0.957],
                              [0.1, 0.0, -15.34, -3.00]])

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(4,) - states of system

        # Returns:
            dx: tf.Tensor, shape=(4,) - time derivatives of the system
        """

        dx = tf.matmul(tf.cast(self.A, x.dtype), tf.expand_dims(x, -1))[..., 0]
        return dx

    def step(self, dt=0.01, n_steps=10, *args, **kwargs):
        """
        Steps the system forward by dt.
        Uses tfdiffeq.odeint for integration.

        # Arguments:
            dt: Float - time step
            n_steps: Int - number of sub-steps to return values for.
                           The integrator may decide to use more steps to
                           achieve the set tolerance.
        # Returns:
            x: tf.Tensor, shape=(4,) - new state of the system
        """
        t = tf.linspace(0., dt, n_steps)
        self.x = odeint(self.call, self.x, t, *args, **kwargs)
        return self.x

    @staticmethod
    def visualize(t, x_val, x_t, dydt_unit, abs_dif, rel_dif,
                  PLOT_DIR, TIME_OF_RUN, log_file_path, epoch=0):
        """Visualize a tf.keras.Model for a 4-dof airplane model.
        # Arguments:
            t: np.ndarray, shape=(samples_per_series) - time of the points in x_val/x_t
            x_val: np.ndarray, shape=(2, samples_per_series, 4) -
                The reference time series against which the model will be compared
            x_t: np.ndarray, shape=(2, samples_per_series, 4) -
                The predicted time series by the model
            dydt_unit: np.ndarray, shape(61, 61, 2) -
                Vector field normalized to unit length
            abs_dif: np.ndarray, shape(61, 61, 2) -
                Vector field of the absolute difference to the reference model
            rel_dif: np.ndarray, shape(61, 61, 2) -
                Vector field of the relative difference to the reference model-
            PLOT_DIR: str - Directory to plot in
            TIME_OF_RUN: str - Time at which the run began
            log_file_path: str - Where to save the log data
            epoch: int
        """
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
        phase_error_extrap_lp = metrics.relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
        phase_error_extrap_sp = metrics.relative_phase_error(x_t[0, :, 2], x_val[0, :, 2])
        traj_error_extrap = metrics.trajectory_error(x_t[0], x_val[0])

        phase_error_interp_lp = metrics.relative_phase_error(x_t[1, :, 0], x_val[1, :, 0])
        phase_error_interp_sp = metrics.relative_phase_error(x_t[1, :, 2], x_val[1, :, 2])
        traj_error_interp = metrics.trajectory_error(x_t[1], x_val[1])

        wall_time = (datetime.datetime.now() - TIME_OF_RUN).total_seconds()

        string = "{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(
            wall_time, epoch,
            phase_error_interp_lp, phase_error_interp_sp,
            phase_error_extrap_lp, phase_error_extrap_sp,
            traj_error_interp, traj_error_extrap)

        if not os.path.isfile(log_file_path):
            title_string = ("wall_time,epoch,"
                            + "phase_error_interp_lp,phase_error_interp_sp,"
                            + "phase_error_extrap_lp,phase_error_extrap_sp,"
                            + "traj_err_interp, traj_err_extrap\n")
            fd = open(log_file_path, 'a')
            fd.write(title_string)
            fd.close()
        fd = open(log_file_path, 'a')
        fd.write(string)
        fd.close()