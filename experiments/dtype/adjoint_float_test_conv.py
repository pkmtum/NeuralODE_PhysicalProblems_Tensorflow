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
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='sigmoid')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='sigmoid')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='sigmoid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1568, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(1568)
        self.reshape = tf.keras.layers.Reshape((14, 14, 8))
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, t, x):
        self.nfe.assign_add(1.)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        dX_dT = self.dense2(x)
        return self.reshape(dX_dT)


file_path = 'plots/dtype/adjoint_float_test_nn.csv'
title_string = "dtype,rtol,method,error,fwd_pass,bwd_pass,nfe,nbe\n"
fd = open(file_path, 'w')
fd.write(title_string)
fd.close()


# Compute the reference gradient
x_0_64 = tf.random.uniform([16, 14, 14, 8], dtype=tf.float64)
t = tf.cast(tf.linspace(0., 2., 2), tf.float64)
odemodel_exact = ODE(dtype=tf.float64)

with tf.device('/gpu:0'):
    with tf.GradientTape() as g:
        y_sol = odeint(odemodel_exact, x_0_64, t, rtol=1e-15, atol=1e-15)[-1]
    dYdX_exact = g.gradient(y_sol, odemodel_exact.trainable_variables)
    with tf.GradientTape() as g:
        y_sol = odeint_adjoint(odemodel_exact, x_0_64, t, rtol=1e-15, atol=1e-15)[-1]
    dYdX_exact_adj = g.gradient(y_sol, odemodel_exact.trainable_variables)
    for x, x_ex in zip(dYdX_exact_adj, dYdX_exact):
        print((tf.norm(x-x_ex)/tf.norm(x_ex)).numpy())

print(odemodel_exact.summary())
for dtype in dtypes:
    if dtype == tf.float32:
        tf.keras.backend.set_floatx('float32')
    else:
        tf.keras.backend.set_floatx('float64')

    x_0 = tf.cast(x_0_64, dtype)
    t = tf.cast(tf.linspace(0., 2., 2), dtype)

    odemodel = ODE(dtype=dtype)
    # build model
    odemodel(t, x_0)
    # set identical weights
    for layer, exact_layer in zip(odemodel.layers, odemodel_exact.layers):
        if hasattr(exact_layer, 'kernel'):
            layer.kernel.assign(tf.cast(exact_layer.kernel, dtype))
        if hasattr(exact_layer, 'bias'):
            layer.bias.assign(tf.cast(exact_layer.bias, dtype))

    for rtol in np.logspace(-13, 0, 14)[::-1]:
        print('rtol:', rtol)
        # Don't run low tolerances with f32, they run for extremely long.
        if rtol <= 1e-11 and dtype == tf.float32:
            break
        with tf.device('/gpu:0'):
            t0 = time.time()
            with tf.GradientTape() as g:
                y_sol = odeint(odemodel, x_0, t, rtol=rtol, atol=1e-14)[-1]
            t1 = time.time()
            dYdX_backprop = g.gradient(y_sol, odemodel.trainable_variables)
            t2 = time.time()
            with tf.GradientTape() as g:
                y_sol_adj = odeint_adjoint(odemodel, x_0, t, rtol=rtol, atol=1e-14)[-1]
            t3 = time.time()
            dYdX_adjoint = g.gradient(y_sol_adj, odemodel.trainable_variables)
            t4 = time.time()
        # Compute relative error by finding the parameter set with the largest relative
        # gradient L2-norm error.
        rel_err_adj = 0
        for x, x_ex in zip(dYdX_adjoint, dYdX_exact):
            err = tf.norm(x-tf.cast(x_ex, dtype))/tf.norm(tf.cast(x_ex, dtype))
            rel_err_adj = max(err.numpy(), rel_err_adj)

        rel_err_bp = 0
        for x, x_ex in zip(dYdX_backprop, dYdX_exact):
            err = tf.norm(x-tf.cast(x_ex, dtype))/tf.norm(tf.cast(x_ex, dtype))
            rel_err_bp = max(err.numpy(), rel_err_bp)

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
