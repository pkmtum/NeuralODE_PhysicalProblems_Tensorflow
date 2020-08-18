import datetime
import os
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
import matplotlib.pyplot as plt
from . import metrics
from .PhysicalSystem import PhysicalSystem


class SinglePendulum(PhysicalSystem):
    """Class that provides a customizable version of a pendulum.
       All parameters are customizable.
    """

    def __init__(self, l=1., x0=0., g=9.81):
        """
        # Arguments:
            l: Float, length of the first arm
            theta: Float, starting angle of first arm
            theta_dt: Float, starting angular velocity of pendulum
            g: Float, gravity
        """
        super(SinglePendulum, self).__init__()
        self.l = l
        self.x = x0
        self.g = g

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(2,) - content: [theta, theta_dt]

        # Returns:
            dx: tf.Tensor, shape=(2,) - time derivatives of the system
        """

        [self.theta, self.theta_dt] = tf.unstack(x, axis=-1)
        theta_dt_dt = -self.g/self.l*tf.math.sin(self.theta)
        theta_dt = self.theta_dt

        dx = tf.stack([theta_dt, theta_dt_dt], axis=-1)
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
        def total_energy(state, m=1., l=1., g=9.81):
            """Calculates total energy of a pendulum system given a state."""
            return m*((1-np.cos(state[..., 0]))*l*g+state[..., 1]*state[..., 1]*l*l*0.5)

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

        steps = dydt_unit.shape[0]
        y, x = np.mgrid[-6:6:complex(0, steps), -6:6:complex(0, steps)]
        ax_vecfield.streamplot(x, y, dydt_unit[:, :, 0], dydt_unit[:, :, 1],
                               color="black")
        ax_vecfield.set_xlim(-6, 6)
        ax_vecfield.set_ylim(-6, 6)

        ax_vec_error_abs.cla()
        ax_vec_error_abs.set_title('Abs. error of thetadot')
        ax_vec_error_abs.set_xlabel('theta')
        ax_vec_error_abs.set_ylabel('theta_dt')

        c1 = ax_vec_error_abs.contourf(x, y, abs_dif, 100)
        plt.colorbar(c1, ax=ax_vec_error_abs)
        for c in c1.collections:
            c.set_edgecolor("face")

        ax_vec_error_abs.set_xlim(-6, 6)
        ax_vec_error_abs.set_ylim(-6, 6)

        ax_vec_error_rel.cla()
        ax_vec_error_rel.set_title('Rel. error of thetadot')
        ax_vec_error_rel.set_xlabel('theta')
        ax_vec_error_rel.set_ylabel('theta_dt')

        c2 = ax_vec_error_rel.contourf(x, y, rel_dif, 100)
        plt.colorbar(c2, ax=ax_vec_error_rel)
        for c in c2.collections:
            c.set_edgecolor("face")

        ax_vec_error_rel.set_xlim(-6, 6)
        ax_vec_error_rel.set_ylim(-6, 6)

        ax_energy.cla()
        ax_energy.set_title('Total Energy')
        ax_energy.set_xlabel('t')
        ax_energy.plot(t.numpy(), total_energy(x_val[0]))
        ax_energy.plot(t.numpy(), total_energy(x_t[0]))

        fig.tight_layout()
        plt.savefig(PLOT_DIR + '{:03d}.pdf'.format(epoch), bbox_inches='tight', pad_inches=0.)
        plt.close()

        # Compute Metrics
        energy_pred = total_energy(x_t[0][-1])
        energy_true = total_energy(x_val[0][-1])
        energy_drift_interp = metrics.relative_energy_drift(energy_pred, energy_true)
        phase_error_interp = metrics.relative_phase_error(x_t[0, :, 0], x_val[0, :, 0])
        traj_err_interp = metrics.trajectory_error(x_t, x_val)

        wall_time = (datetime.datetime.now() - TIME_OF_RUN).total_seconds()

        string = "{},{},{},{},{}\n".format(wall_time, epoch,
                                           energy_drift_interp,
                                           phase_error_interp,
                                           traj_err_interp)

        print(string)

        if not os.path.isfile(log_file_path):
            title_string = "wall_time,epoch,energy_interp,phase_interp,traj_err_interp\n"
            fd = open(log_file_path, 'a')
            fd.write(title_string)
            fd.close()
        fd = open(log_file_path, 'a')
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
