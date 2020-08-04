from tfdiffeq import odeint
import tensorflow as tf


class SinglePendulum(tf.keras.Model):
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
