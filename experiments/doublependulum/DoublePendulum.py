from tfdiffeq import odeint
import tensorflow as tf
import matplotlib.pyplot as plt

class DoublePendulum(tf.keras.Model):
    """Class that provides a customizable version of a double pendulum.
       All parameters are customizable.
    """

    def __init__(self, l1=1., l2=1., m1=1., m2=1., theta1=0., theta2=0., theta1_dt=0., theta2_dt=0., g=9.81):
        """
        # Arguments:
            l1: Float, length of the first arm
            l2: Float, length of the second arm
            m1: Float, mass of upper mass
            m2: Float, mass of lower mass
            theta1: Float, starting angle of first arm
            theta2: Float, starting angle of second arm
            theta1_dt: Float, starting angular velocity of first arm
            theta2_dt: Float, starting angular velocity of second arm
            g: Float, gravity
        """

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.mu = 1+m1/m2
        self.x = tf.stack([theta1, theta2, theta1_dt, theta2_dt])
        self.g = g

    @tf.function
    def call(self, t, x):
        """
        Returns time-derivatives of the system.

        # Arguments
            t: Float - current time, irrelevant
            x: tf.Tensor, shape=(4,) - content: [theta1, theta2, theta1_dt, theta2_dt]

        # Returns:
            dx: tf.Tensor, shape=(4,) - time derivatives of the system
        """

        [self.theta1, self.theta2, self.theta1_dt, self.theta2_dt] = tf.unstack(x)
        difference = self.theta1-self.theta2
        t1 = self.theta1
        t2 = self.theta2

        theta1_dt = self.theta1_dt
        theta2_dt = self.theta2_dt
        den1 = self.l1*(self.mu-tf.math.cos(difference)*tf.math.cos(difference))
        den2 = self.l2*(self.mu-tf.math.cos(difference)*tf.math.cos(difference))
        theta1_dt_dt =((self.g*(tf.math.sin(t2)*tf.math.cos(difference)-self.mu*tf.math.sin(t1)) - (self.l2*theta2_dt*theta2_dt + self.l1*theta1_dt*theta1_dt*tf.math.cos(difference))*tf.math.sin(difference))
                     / den1)
        theta2_dt_dt =((self.g*self.mu*(tf.math.sin(t1)*tf.math.cos(difference)-tf.math.sin(t2)) + (self.mu*self.l1*theta1_dt*theta1_dt+self.l2*theta2_dt*theta2_dt*tf.math.cos(difference))*tf.math.sin(difference))
                     / den2)
        dx = tf.stack([theta1_dt, theta2_dt, theta1_dt_dt, theta2_dt_dt])
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
            x: tf.Tensor, shape=(4,) - new state of the system
        """

        t = tf.linspace(0., dt, n_steps)
        self.x = odeint(self.call, self.x, t, *args, **kwargs)
        return self.x
