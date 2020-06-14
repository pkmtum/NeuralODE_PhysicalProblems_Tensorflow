"""Comparision of the adjoint method with regular backpropagation.
Also compares tf.float32 to tf.float64.
"""
import time
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
from tfdiffeq import odeint_adjoint

file_path = 'tests/adjoint_float_test.csv'
title_string = "dtype,rtol,method,error,fwd_pass,bwd_pass,nfe,nbe\n"
fd = open(file_path, 'w')
fd.write(title_string)
fd.close()

dtypes = [tf.float32, tf.float64]

for rtol in np.logspace(-13, 0, 14)[::-1]:
    print('rtol:', rtol)
    for dtype in dtypes:
        # wir wollen gradient bzgl self.b
        class ODE(tf.keras.Model):
            def __init__(self, a, b):
                super(ODE, self).__init__(dtype=dtype)
                self.a = tf.Variable(a)
                self.b = tf.Variable(b)
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

        x_0 = tf.constant(1., dtype=dtype) # not important for Gradient
        a = tf.constant(2., dtype=dtype)
        b = tf.constant(2., dtype=dtype)
        T = tf.constant(2., dtype=dtype)
        t = tf.cast(tf.linspace(0., T, 2), dtype=dtype)

        odemodel = ODE(a, b)

        with tf.device('/gpu:0'):
            var = odemodel.b
            t0 = time.time()
            with tf.GradientTape() as g:
                g.watch(var)
                y_sol = odeint(odemodel, x_0, t, rtol=rtol, atol=1e-10)[-1]
            t1 = time.time()
            dYdX_backprop = g.gradient(y_sol, var).numpy()
            t2 = time.time()
            with tf.GradientTape() as g:
                g.watch(var)
                y_sol_adj = odeint_adjoint(odemodel, x_0, t, rtol=rtol, atol=1e-10)[-1]
            t3 = time.time()
            dYdX_adjoint = g.gradient(y_sol_adj, var).numpy()
            t4 = time.time()
        dYdX_exact = exact_derivative(a, b, T).numpy()

        print('Adjoint:', abs(dYdX_adjoint-dYdX_exact)/dYdX_exact, dtype)
        print('Backprop:', abs(dYdX_backprop-dYdX_exact)/dYdX_exact, dtype)
        fd = open(file_path, 'a')
        fd.write('{},{},adjoint,{},{},{},{},{}\n'.format(dtype,
                                                         rtol,
                                                         abs(dYdX_adjoint-dYdX_exact)/dYdX_exact,
                                                         t3-t2,
                                                         t4-t3,
                                                         odemodel.nfe.numpy(),
                                                         odemodel.nbe.numpy()))
        fd.write('{},{},backprop,{},{},{},{},{}\n'.format(dtype,
                                                          rtol,
                                                          abs(dYdX_backprop-dYdX_exact)/dYdX_exact,
                                                          t1-t0,
                                                          t2-t1,
                                                          odemodel.nfe.numpy(),
                                                          0))
        fd.close()
