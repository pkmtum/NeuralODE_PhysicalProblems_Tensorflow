from tfdiffeq import odeint
import tensorflow as tf

class AirplaneLatLong(tf.keras.Model):
    """Class that provides a longitudinal motion model of a Boeing B777.
    State has the following structure: (V*, gamma, alpha, q, r, beta, p, phi).T
    """

    def __init__(self, x0=0.):
        """
        # Arguments:
            x0: tf.Tensor, shape=(8,), state at x_0
        """
        super(AirplaneLatLong, self).__init__()
        self.x = x0
        self.A = tf.constant([[-0.01, -0.49, -0.046, -0.001, 0., 0., 0., 0.],
                              [0.11, 0.0003, 1.14, 0.043, 0., 0., 0., 0.],
                              [-0.11, 0.0003, -1.14, 0.957, 0., 0., 0., 0.],
                              [0.1, 0.0, -15.34, -3.00, 0., 0., 0., 0.],
                              [0., 0., 0., 0., -0.87, 6.47, -0.41, 0.],
                              [0., 0., 0., 0., -1, -0.38, 0, 0.07],
                              [0., 0., 0., 0., 0.91, -18.8, -0.65, 0.],
                              [0., 0., 0., 0., 0., 0., 1., 0.]])

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(8,) - states of system

        # Returns:
            dx: tf.Tensor, shape=(8,) - time derivatives of the system
        """

        dx = tf.matmul(self.A, tf.expand_dims(x, -1))[..., 0]
        return dx

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
            x: tf.Tensor, shape=(8,) - new state of the system
        """
        t = tf.linspace(0., dt, n_steps)
        self.x = odeint(self, self.x, t, *args, **kwargs)
        return self.x
