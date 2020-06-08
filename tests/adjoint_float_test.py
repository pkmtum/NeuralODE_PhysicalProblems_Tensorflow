from typing import Iterable
import numpy as np
import tensorflow as tf

from tfdiffeq import odeint
from tfdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

dtype = tf.float64
# wir wollen gradient bzgl self.b
class ODE(tf.keras.Model):
    def __init__(self, a, b):
        super(ODE, self).__init__(dtype=dtype)
        self.a = a
        self.b = b

    def call(self, t, x):
        dX_dT = self.a*tf.math.exp(self.b*t)
        return dX_dT

def euler(func, y0, t, return_intermediates=True):
    solution = [y0]
    y = y0
    for t0, t1 in zip(t[:-1], t[1:]):
        y = y + (t1-t0)*func(t0, y)
        solution.append(y)
    if not return_intermediates:
        return y[-1]
    return tf.convert_to_tensor(solution)

def exact_solution(a, b, T):
    return a/b*(np.exp(b*T)-1)

def exact_derivative(a, b, T):
    return a*(T/b*np.exp(b*T)-(np.exp(b*T)-1)/(b*b))

x_0 = tf.constant(0., dtype=dtype)
a = tf.constant(2., dtype=dtype)
b = tf.constant(2., dtype=dtype)
T = tf.constant(2., dtype=tf.float64)
t = tf.cast(tf.linspace(0., T, 2), dtype=tf.float64)

odemodel = ODE(a, b)

sol = []

with tf.device('/gpu:0'):
    for i in range(10):
        with tf.GradientTape(persistent=True) as g:
            g.watch(odemodel.b)
            y_sol = odeint(odemodel, x_0, t)
            y_sol = y_sol[-1]
        dF_dB = g.gradient(y_sol, odemodel.b)
        sol.append([a, odemodel.b, T, dF_dB])

sol = np.array(sol)
dYdX_adjoint = sol[:, -1]
dYdX_exact = exact_derivative(sol[:, 0], sol[:, 1], sol[:, 2])
print(dYdX_exact, dYdX_adjoint, dYdX_exact/dYdX_adjoint)
