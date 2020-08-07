import datetime
import os
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
import matplotlib.pyplot as plt
from . import metrics


class MassSpringDamper(tf.keras.Model):
    """Class that provides a customizable version of a mass-spring-damper system.
       All parameters are customizable.
    """

    def __init__(self, m=1., c=1., d=0., x0=0.,):
        """
        # Arguments:
            m: Float, mass
            d: Float, damper coefficient (default: 0.)
            c: Float, spring coefficient
            x: tf.Tensor, shape=(2,), state at x_0
        """
        super(MassSpringDamper, self).__init__()
        self.x = x0
        self.A = tf.constant([[0., 1.],
                              [-c/m, -d/m]])

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(2,) - content: [x, x_dt]

        # Returns:
            dx: tf.Tensor, shape=(2,) - time derivatives of the system
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
        """Visualize a tf.keras.Model for a mass-spring-damper.
        # Arguments:
            t: np.ndarray, shape=(samples_per_series) -
                time of the points in x_val/x_t
            x_val: np.ndarray, shape=(2, samples_per_series, 8) -
                The reference time series against which the model will be compared
            x_t: np.ndarray, shape=(2, samples_per_series, 8) -
                The predicted time series by the model
            dydt_unit: np.ndarray, shape=dydt_unit.shape -
                Vector field normalized to unit length
            abs_dif: np.ndarray, shape=dydt_unit.shape -
                Vector field of the absolute difference to the reference model
            rel_dif: np.ndarray, shape=dydt_unit.shape -
                Vector field of the relative difference to the reference model
            PLOT_DIR: str - Directory to plot in
            TIME_OF_RUN: str - Time at which the run began
            log_file_path: str - Where to save the log data
            epoch: int
        """
        def total_energy(state, k=1, m=1):
            """Calculates total energy of a mass-spring-damper system given a state."""
            return 0.5*k*state[..., 0]*state[..., 0]+0.5*m*state[..., 1]*state[..., 1]

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

        steps = dydt_unit.shape[0]
        y, x = np.mgrid[-6:6:complex(0, steps), -6:6:complex(0, steps)]
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
        energy_extrap_pred = total_energy(x_t[0][-1])
        energy_extrap_true = total_energy(x_val[0][-1])
        energy_drift_extrap = metrics.relative_energy_drift(energy_extrap_pred,
                                                            energy_extrap_true)
        phase_error_extrap = metrics.relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
        traj_error_extrap = metrics.trajectory_error(x_t[0], x_val[0])

        energy_interp_pred = total_energy(x_t[1][-1])
        energy_interp_true = total_energy(x_val[1][-1])
        energy_drift_interp = metrics.relative_energy_drift(energy_interp_pred,
                                                            energy_interp_true)
        phase_error_interp = metrics.relative_phase_error(x_t[1, :, 0], x_val[1, :, 0])
        traj_error_interp = metrics.trajectory_error(x_t[1], x_val[1])

        wall_time = (datetime.datetime.now() - TIME_OF_RUN).total_seconds()

        string = "{},{},{},{},{},{},{},{}\n".format(wall_time, epoch,
                                                    energy_drift_interp, energy_drift_extrap,
                                                    phase_error_interp, phase_error_extrap,
                                                    traj_error_interp, traj_error_extrap)
        if not os.path.isfile(log_file_path):
            title_string = ("wall_time,epoch,energy_drift_interp,energy_drift_extrap,phase_error_interp,"
                            + "phase_error_extrap,traj_err_interp,traj_err_extrap\n")
            fd = open(log_file_path, 'a')
            fd.write(title_string)
            fd.close()
        fd = open(log_file_path, 'a')
        fd.write(string)
        fd.close()