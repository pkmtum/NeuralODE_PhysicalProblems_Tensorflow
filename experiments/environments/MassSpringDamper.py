from tfdiffeq import odeint
import tensorflow as tf


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
