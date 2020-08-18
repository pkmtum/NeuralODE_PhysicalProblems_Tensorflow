import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class PhysicalSystem(tf.keras.Model):
    """Abstract class that provides basic functionality"""
    def __init__(self):
        super(PhysicalSystem, self).__init__()

    @staticmethod
    def visualize(t, x_val, x_t, dydt_unit, abs_dif, rel_dif,
                  PLOT_DIR, TIME_OF_RUN, log_file_path, epoch=0):
        """Basic visualization for a tf.keras.Model.
        # Arguments:
            t: np.ndarray, shape=(samples_per_series) - time of the points in x_val/x_t
            x_val: np.ndarray, shape=(n_series, samples_per_series, dof) -
                The reference time series against which the model will be compared
            x_t: np.ndarray, shape=(n_series, samples_per_series, dof) -
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
        # Plot the generated trajectories
        dof = x_val.shape[-1]
        cols = 4
        rows = dof // cols + 1
        for traj in range(x_val.shape[0]):
            fig = plt.figure(figsize=(4*cols, 4*rows), facecolor='white')
            ax_traj = [fig.add_subplot(rows*100+cols*10+1+i, frameon=False) for i in range(dof)]
            for i, ax in enumerate(ax_traj):
                ax.cla()
                ax.set_title('Trajectories {}'.format(i))
                ax.set_xlabel('t')
                ax.set_ylabel('x[{}]'.format(i))
                ax.plot(t.numpy(), x_val[traj, :, i], 'g--')
                ax.plot(t.numpy(), x_t[traj, :, i], 'b--')
                ax.set_xlim(min(t.numpy()), max(t.numpy()))
                ul = np.max(x_val[traj, :, i])
                ll = np.min(x_val[traj, :, i])
                ax.set_ylim(int(ll - 1.), int(ul + 1.))
            fig.tight_layout()
            plt.savefig(PLOT_DIR + 'Trajs{}_{:03d}.pdf'.format(traj, epoch), bbox_inches='tight', pad_inches=0.)
            plt.close()