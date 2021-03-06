"""Comparision of the adjoint method with regular backpropagation.
Uses the simple test equation:
$ f(T) = \int_{0}^{T}x dt$
subject to $ \dot{x} = bx $
$ x(0) = a $
Also compares tf.float32 to tf.float64.
"""
import time
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
from tfdiffeq import odeint_adjoint

dtypes = [tf.float32, tf.float64]
tf.keras.backend.set_floatx('float64')

class ODE(tf.keras.Model):

    def __init__(self, a, b, dtype):
        super(ODE, self).__init__(dtype=dtype)
        self.a = tf.Variable(a, trainable=True)
        self.b = tf.Variable(b, trainable=True)
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

    def call(self, t, x):
        self.nfe.assign_add(1.)
        dX_dT = self.a*tf.math.exp(self.b*t)
        return dX_dT

def exact_solution(a, b, T):
    return a/b*(np.exp(b*T)-1)

def exact_derivative(a, b, T):
    return a*(T/b*np.exp(b*T)-(np.exp(b*T)-1)/(b*b))

file_path = 'plots/dtype/adjoint_float_test.csv'
title_string = "dtype,rtol,method,error,fwd_pass,bwd_pass,nfe,nbe\n"
fd = open(file_path, 'w')
fd.write(title_string)
fd.close()

for dtype in dtypes:
    if dtype == tf.float32:
        tf.keras.backend.set_floatx('float32')
    else:
        tf.keras.backend.set_floatx('float64')
    x_0 = tf.constant(1., dtype=dtype)  # not important for Gradient
    a = tf.constant(2., dtype=dtype)
    b = tf.constant(2., dtype=dtype)
    T = tf.constant(2., dtype=dtype)
    t = tf.cast(tf.linspace(0., T, 2), dtype)

    odemodel = ODE(a, b, dtype)
    for rtol in np.logspace(-13, 0, 14)[::-1]:
        print('rtol:', rtol)
        # Run forward and backward passes, while tracking the time
        with tf.device('/gpu:0'):
            t0 = time.time()
            with tf.GradientTape() as g:
                y_sol = odeint(odemodel, x_0, t, rtol=rtol, atol=1e-10)[-1]
            t1 = time.time()
            dYdX_backprop = g.gradient(y_sol, odemodel.b).numpy()
            t2 = time.time()
            with tf.GradientTape() as g:
                y_sol_adj = odeint_adjoint(odemodel, x_0, t, rtol=rtol, atol=1e-10)[-1]
            t3 = time.time()
            dYdX_adjoint = g.gradient(y_sol_adj, odemodel.b).numpy()
            t4 = time.time()
        dYdX_exact = exact_derivative(a, b, T).numpy()
        rel_err_adj = abs(dYdX_adjoint-dYdX_exact)/dYdX_exact
        rel_err_bp = abs(dYdX_backprop-dYdX_exact)/dYdX_exact
        print('Adjoint:', rel_err_adj, dtype)
        print('Backprop:', rel_err_bp, dtype)
        fd = open(file_path, 'a')
        fd.write('{},{},adjoint,{},{},{},{},{}\n'.format(dtype,
                                                         rtol,
                                                         rel_err_adj,
                                                         t3-t2,
                                                         t4-t3,
                                                         odemodel.nfe.numpy(),
                                                         odemodel.nbe.numpy()))
        fd.write('{},{},backprop,{},{},{},{},{}\n'.format(dtype,
                                                          rtol,
                                                          rel_err_bp,
                                                          t1-t0,
                                                          t2-t1,
                                                          odemodel.nfe.numpy(),
                                                          0))
        fd.close()
