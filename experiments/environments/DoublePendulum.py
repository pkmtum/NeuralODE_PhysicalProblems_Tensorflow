import datetime
import os
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
import matplotlib.pyplot as plt
from . import metrics


class DoublePendulum(tf.keras.Model):
    """Class that provides a customizable version of a pendulum.
       All parameters are customizable.
    """

    def __init__(self, l1=1., l2=1., m1=1., m2=1., x0=0., g=9.81):
        """
        # Arguments:
            l1: Float, length of the first arm
            l2: Float, length of the second arm
            m1: Float, mass of upper mass
            m2: Float, mass of lower mass
            x0: shape=(4,) - theta1, theta2, theta1_dt, theta2_dt
            g: Float, gravity
        """
        super(DoublePendulum, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.mu = 1+m1/m2
        self.x = x0
        self.g = g

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(4,) - content: [theta, theta_dt]

        # Returns:
            dx: tf.Tensor, shape=(4,) - time derivatives of the system
        """

        """
         Returns time-derivatives of the system.

         # Arguments
             t: Float - current time, irrelevant
             x: tf.Tensor, shape=(4,) - content: [theta1, theta2, theta1_dt, theta2_dt]

         # Returns:
             dx: tf.Tensor, shape=(4,) - time derivatives of the system
         """

        [th1, th2, th1_dt, th2_dt] = tf.unstack(x, axis=-1)
        diff = th1-th2

        den1 = self.l1*(self.mu-tf.math.cos(diff)*tf.math.cos(diff))
        den2 = self.l2*(self.mu-tf.math.cos(diff)*tf.math.cos(diff))
        th1_dt_dt = ((self.g*(tf.math.sin(th2)*tf.math.cos(diff)-self.mu*tf.math.sin(th1))
                      - (self.l2*th2_dt*th2_dt + self.l1*th1_dt*th1_dt*tf.math.cos(diff)) * tf.math.sin(diff))
                     / den1)
        th2_dt_dt = ((self.g*self.mu*(tf.math.sin(th1)*tf.math.cos(diff)-tf.math.sin(th2))
                      + (self.mu*self.l1*th1_dt*th1_dt+self.l2*th2_dt*th2_dt*tf.math.cos(diff))*tf.math.sin(diff))
                    / den2)
        dx = tf.stack([th1_dt, th2_dt, th1_dt_dt, th2_dt_dt], axis=-1)
        return dx

    def step(self, dt=0.01, n_steps=10, *args, **kwargs):
        """
        Convenience function, steps the system forward by dt.
        Uses tfdiffeq's odeint for integration.

        # Arguments:
            dt: Float - time step
            n_steps: Int - number of sub-steps to return values for.
                           The integrator may decide to use more steps to achieve the
                           set tolerance.
        # Returns:
            x: tf.Tensor, shape=(2,) - new state of the system
        """

        t = tf.linspace(0., dt, n_steps)
        self.x = odeint(self.call, self.x, t, *args, **kwargs)
        return self.x

    @staticmethod
    def visualize(t, x_val, x_t, dydt_unit, abs_dif, rel_dif,
                  PLOT_DIR, TIME_OF_RUN, log_file_path, epoch=0):
        """Visualize a tf.keras.Model for a single pendulum.
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
            TIME_OF_RUN: datetime object - Time at which the run began
            log_file_path: str - Where to save the log data
            epoch: int
        """
        def total_energy(state, l1=1., l2=1., m1=1., m2=1., g=9.81):
            [th1, th2, th1d, th2d] = tf.unstack(state, axis=-1)
            """Calculates total energy of a pendulum system given a state."""
            V = (m1+m2)*l1*g*(1-np.cos(th1)) + m2*l2*g*(1-np.cos(th2))
            T = 0.5*m1*tf.math.square(l1*th1d) + 0.5*m2*(tf.math.square(l1*th1d)
                + tf.math.square(l2*th2d) + 2*l1*l2*th1d*th2d*np.cos(th1-th2))
            return T + V

        fig = plt.figure(figsize=(12, 12), facecolor='white')
        ax_traj_interp = fig.add_subplot(331, frameon=False)
        ax_traj_extrap = fig.add_subplot(332, frameon=False)
        ax_phase_interp = fig.add_subplot(334, frameon=False)
        ax_phase_extrap = fig.add_subplot(335, frameon=False)
        ax_vecfield = fig.add_subplot(333, frameon=False)
        ax_vec_error_abs = fig.add_subplot(336, frameon=False)
        ax_vec_error_rel = fig.add_subplot(339, frameon=False)
        ax_energy_interp = fig.add_subplot(337, frameon=False)
        ax_energy_extrap = fig.add_subplot(338, frameon=False)
        ax_traj_interp.cla()
        ax_traj_interp.set_title('Trajectories (interpolation)')
        ax_traj_interp.set_xlabel('t')
        ax_traj_interp.set_ylabel('x,y')
        ax_traj_interp.plot(t.numpy(), x_val[1, :, 0], t.numpy(), x_val[1, :, 1], 'g-')
        ax_traj_interp.plot(t.numpy(), x_t[1, :, 0], '--', t.numpy(), x_t[1, :, 1], 'b--')
        ax_traj_interp.set_xlim(min(t.numpy()), max(t.numpy()))
        ax_traj_interp.set_ylim(-3, 3)

        ax_traj_extrap.cla()
        ax_traj_extrap.set_title('Trajectories (extrapolation)')
        ax_traj_extrap.set_xlabel('t')
        ax_traj_extrap.set_ylabel('x,y')
        ax_traj_extrap.plot(t.numpy(), x_val[0, :, 0], t.numpy(), x_val[0, :, 1], 'g-')
        ax_traj_extrap.plot(t.numpy(), x_t[0, :, 0], '--', t.numpy(), x_t[0, :, 1], 'b--')
        ax_traj_extrap.set_xlim(min(t.numpy()), max(t.numpy()))
        ax_traj_extrap.set_ylim(-3, 3)

        ax_phase_interp.cla()
        ax_phase_interp.set_title('Phase Portrait (interpolation)')
        ax_phase_interp.set_xlabel('theta')
        ax_phase_interp.set_ylabel('theta_dt')
        ax_phase_interp.plot(x_val[1, :, 0], x_val[1, :, 1], 'g--')
        ax_phase_interp.plot(x_t[1, :, 0], x_t[1, :, 1], 'b--')
        ax_phase_interp.set_xlim(-1, 1)
        ax_phase_interp.set_ylim(-1, 1)

        ax_phase_extrap.cla()
        ax_phase_extrap.set_title('Phase Portrait (extrapolation)')
        ax_phase_extrap.set_xlabel('theta')
        ax_phase_extrap.set_ylabel('theta_dt')
        ax_phase_extrap.plot(x_val[0, :, 0], x_val[0, :, 1], 'g--')
        ax_phase_extrap.plot(x_t[0, :, 0], x_t[0, :, 1], 'b--')
        ax_phase_extrap.set_xlim(-3, 3)
        ax_phase_extrap.set_ylim(-3, 3)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('theta')
        ax_vecfield.set_ylabel('theta_dt')

        steps = dydt_unit.shape[0]
        y, x = np.mgrid[-3:3:complex(0, steps), -3:3:complex(0, steps)]
        ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1],
                               color="black")
        ax_vecfield.set_xlim(-3, 3)
        ax_vecfield.set_ylim(-3, 3)

        ax_vec_error_abs.cla()
        ax_vec_error_abs.set_title('Abs. error of thetadot')
        ax_vec_error_abs.set_xlabel('theta')
        ax_vec_error_abs.set_ylabel('theta_dt')

        c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
        plt.colorbar(c1, ax=ax_vec_error_abs)

        ax_vec_error_abs.set_xlim(-3, 3)
        ax_vec_error_abs.set_ylim(-3, 3)

        ax_vec_error_rel.cla()
        ax_vec_error_rel.set_title('Rel. error of thetadot')
        ax_vec_error_rel.set_xlabel('theta')
        ax_vec_error_rel.set_ylabel('theta_dt')

        c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
        plt.colorbar(c2, ax=ax_vec_error_rel)

        ax_vec_error_rel.set_xlim(-3, 3)
        ax_vec_error_rel.set_ylim(-3, 3)

        ax_energy_interp.cla()
        ax_energy_interp.set_title('Total Energy (interpolation)')
        ax_energy_interp.set_xlabel('t')
        ax_energy_interp.plot(t.numpy(), total_energy(x_val[1]))
        ax_energy_interp.plot(t.numpy(), total_energy(x_t[1]))

        ax_energy_extrap.cla()
        ax_energy_extrap.set_title('Total Energy (extrapolation)')
        ax_energy_extrap.set_xlabel('t')
        ax_energy_extrap.plot(t.numpy(), total_energy(x_val[0]))
        ax_energy_extrap.plot(t.numpy(), total_energy(x_t[0]))

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