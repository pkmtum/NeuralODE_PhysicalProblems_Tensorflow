"""Comparision of the adjoint method with regular backpropagation.
Also compares tf.float32 to tf.float64.
"""
import time
import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
from tfdiffeq import odeint_adjoint
tf.random.set_seed(0)

dtypes = [tf.float32, tf.float64]
tf.keras.backend.set_floatx('float64')

class ODE(tf.keras.Model):
    def __init__(self, dtype):
        super(ODE, self).__init__(dtype=dtype)
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2)
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

    def call(self, t, x):
        self.nfe.assign_add(1.)
        x = self.dense1(x)
        dX_dT = self.dense2(x)
        return dX_dT

file_path = 'tests/adjoint_float_test_nn.csv'
title_string = "dtype,rtol,method,error,fwd_pass,bwd_pass,nfe,nbe\n"
fd = open(file_path, 'a')
fd.write(title_string)
fd.close()



x_0 = tf.constant([[1., 10.]], dtype=tf.float64)
t = tf.cast(tf.linspace(0., 2., 2), dtype=tf.float64)
odemodel_exact = ODE(dtype=tf.float64)

with tf.device('/gpu:0'):
    with tf.GradientTape() as g:
        y_sol = odeint_adjoint(odemodel_exact, x_0, t, rtol=1e-13, atol=1e-13)[-1]
    dYdX_exact = g.gradient(y_sol, odemodel_exact.trainable_variables)

for dtype in dtypes:
    if dtype == tf.float32:
        tf.keras.backend.set_floatx('float32')
    else:
        tf.keras.backend.set_floatx('float64')

    x_0 = tf.constant([[1., 10.]], dtype=dtype)
    t = tf.cast(tf.linspace(0., 2., 2), dtype=dtype)

    odemodel = ODE(dtype=dtype)
    # build model
    odemodel(t, x_0)
    # set identical weights
    odemodel.dense1.kernel.assign(tf.cast(odemodel_exact.dense1.kernel, dtype=dtype))
    odemodel.dense1.bias.assign(tf.cast(odemodel_exact.dense1.bias, dtype=dtype))
    odemodel.dense2.kernel.assign(tf.cast(odemodel_exact.dense2.kernel, dtype=dtype))
    odemodel.dense2.bias.assign(tf.cast(odemodel_exact.dense2.bias, dtype=dtype))

    for rtol in np.logspace(-11, 0, 12)[::-1]:
        print('rtol:', rtol)
        with tf.device('/gpu:0'):
            t0 = time.time()
            with tf.GradientTape() as g:
                y_sol = odeint(odemodel, x_0, t, rtol=rtol, atol=1e-12)[-1]
            t1 = time.time()
            dYdX_backprop = g.gradient(y_sol, odemodel.trainable_variables)
            t2 = time.time()
            with tf.GradientTape() as g:
                y_sol_adj = odeint_adjoint(odemodel, x_0, t, rtol=rtol, atol=1e-12)[-1]
            t3 = time.time()
            dYdX_adjoint = g.gradient(y_sol_adj, odemodel.trainable_variables)
            t4 = time.time()
        max_adj = 0
        for x, x_ex in zip(dYdX_adjoint, dYdX_exact):
            max_adj = max(tf.reduce_max(tf.math.abs(x - tf.cast(x_ex, dtype))).numpy(), max_adj)

        max_bp = 0
        for x, x_ex in zip(dYdX_backprop, dYdX_exact):
            max_bp = max(tf.reduce_max(tf.math.abs(x - tf.cast(x_ex, dtype))).numpy(), max_bp)
        print('Adjoint:', max_adj, dtype)
        print('Backprop:', max_bp, dtype)
        fd = open(file_path, 'a')
        fd.write('{},{},adjoint,{},{},{},{},{}\n'.format(dtype,
                                                         rtol,
                                                         max_adj,
                                                         t3-t2,
                                                         t4-t3,
                                                         odemodel.nfe.numpy(),
                                                         odemodel.nbe.numpy()))
        fd.write('{},{},backprop,{},{},{},{},{}\n'.format(dtype,
                                                         rtol,
                                                         max_bp,
                                                         t1-t0,
                                                         t2-t1,
                                                         odemodel.nfe.numpy(),
                                                         0))
        fd.close()
