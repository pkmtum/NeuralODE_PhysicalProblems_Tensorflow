from scipy.integrate import odeint
from my_tfdiffeq import odeint as odeint_adjoint
import tensorflow as tf
import time

class LotkaVolterra(tf.keras.Model):

  def __init__(self, a, b, c, d,):
    super().__init__()
    self.a, self.b, self.c, self.d = a, b, c, d

  @tf.function
  def call(self, t, y):
    # y = [R, F]
    r, f = tf.unstack(y)

    dR_dT = self.a * r - self.b * r * f
    dF_dT = -self.c * f + self.d * r * f

    return tf.stack([dR_dT, dF_dT])

y0 = tf.constant([2., 1.])
t = tf.linspace(0., 10., 101)

func = LotkaVolterra(1.,2.,0.3,0.2)

t0 = time.time()
# ys = odeint(func, y0, t,tfirst=True)
t1 = time.time()
# ys = odeint_adjoint(func, y0, t)
t2 = time.time()
for i in range(1000):
    i = tf.constant(i, dtype=tf.float64)
    f = [i for k in range(100)]
    x = sum(f)
t3 = time.time()
for i in range(1000):
    i = tf.constant(i, dtype=tf.float64)
    f = [i for k in range(100)]
    y = tf.math.reduce_sum(f)
t4 = time.time()
for i in range(1000):
    i = tf.constant(i, dtype=tf.float64)
    f = [i for k in range(100)]
    y = tf.math.add_n(f)
t5 = time.time()
print(x==y)
print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)
