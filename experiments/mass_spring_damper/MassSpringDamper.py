from tfdiffeq import odeint
import tensorflow as tf
import matplotlib.pyplot as plt

class MassSpringDamper(tf.keras.Model):
    """Class that provides a customizable version of a mass-spring-damper system.
       All parameters are customizable.
    """

    def __init__(self, m=1., c=1., d=0., x=0., x_dt=0.):
        """
        # Arguments:
            m: Float, mass
            d: Float, damper coefficient (default: 0.)
            c: Float, spring coefficient
            x: Float, starting position of the mass
            x_dt: Float, starting velocity of the mass
        """

        self.m = m
        self.d = d
        self.c = c
        self.y = tf.stack([x, x_dt], axis=-1)

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

        x, x_dt = tf.unstack(x, axis=-1)
        x_dt_dt = -(self.d*x_dt+self.c*x)/self.m
        dx = tf.stack([x_dt, x_dt_dt], axis=-1)
        return dx

    def total_energy(self):
        return 0.5*(self.c*self.y[0]*self.y[0]+self.m*self.y[1]*self.y[1])

    def step(self, dt=0.01, n_steps=10, *args, **kwargs):
        """
        Steps the system forward by dt.
        Uses tfdiffeq.odeint for integration.

        # Arguments:
            dt: Float - time step
            n_steps: Int - number of sub-steps to return values for.
                           The integrator may decide to use more steps to achieve the
                           set tolerance.
        # Returns:
            x: tf.Tensor, shape=(4,) - new state of the system
        """

        t = tf.linspace(0., dt, n_steps)
        self.y = odeint(self.call, self.y, t, *args, **kwargs)
        return self.y
