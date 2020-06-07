import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tfdiffeq import odeint
from tfdiffeq.misc import _flatten
import matplotlib.pyplot as plt
import time
# wir wollen gradient bzgl self.b!!
class ODE(tf.keras.Model):
    def __init__(self, a, b):
        super(ODE, self).__init__(dtype=tf.float64)
        self.a = a
        self.b = b

    def call(self, t, x):
        t = tf.cast(t, dtype=tf.float64)
        dX_dT = self.a*tf.math.exp(self.b*t)
        return dX_dT

def euler(func, y0, t, return_intermediates=True):
    solution = [y0]
    y = y0
    for t0, t1 in zip(t[:-1], t[1:]):
        # print(y, t1, func(t0, y))
        y = y + (t1-t0)*func(t0, y)
        solution.append(y)
    if not return_intermediates:
        return y[-1]
    tf.print(solution)
    return tf.convert_to_tensor(solution)

class _Arguments(object):

    def __init__(self, func, method, options, rtol, atol):
        self.func = func
        self.method = method
        self.options = options
        self.rtol = rtol
        self.atol = atol

@tf.custom_gradient
def OdeintAdjointMethod(*args):
    global _arguments
    # args = _arguments.args
    # kwargs = _arguments.kwargs
    func = _arguments.func
    method = _arguments.method
    options = _arguments.options
    rtol = _arguments.rtol
    atol = _arguments.atol

    y0, t = args[:-1], args[-1]

    # registers `t` as a Variable that needs a grad, then resets it to a Tensor
    # for the `odeint` function to work. This is done to force tf to allow us to
    # pass the gradient of t as output.
    # t = tf.get_variable('t', initializer=t)
    t = tf.convert_to_tensor(t, dtype=t.dtype)
    # euler(func, y0, t, return_intermediates=True)
    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    # @func_cast_double
    def grad(*grad_output, variables=None):
        global _arguments
        flat_params = _flatten(variables)

        func = _arguments.func
        method = _arguments.method
        options = _arguments.options
        rtol = _arguments.rtol
        atol = _arguments.atol

        n_tensors = len(ans)
        f_params = tuple(variables)

        # TODO: use a tf.keras.Model and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.

            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with tf.GradientTape() as tape:
                tape.watch(t)
                tape.watch(y)
                func_eval = func(t, y)
                func_eval = cast_double(func_eval)

            gradys = tf.stack(list(-adj_y_ for adj_y_ in adj_y))
            if len(gradys.shape) < len(func_eval.shape):
                gradys = tf.expand_dims(gradys, axis=0)
            vjp_t, *vjp_y_and_params = tape.gradient(
                func_eval,
                (t,) + y + f_params,
                output_gradients=gradys
            )

            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = tf.zeros_like(t, dtype=t.dtype) if vjp_t is None else vjp_t
            vjp_y = tuple(tf.zeros_like(y_, dtype=y_.dtype)
                          if vjp_y_ is None else vjp_y_
                          for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if _check_len(f_params) == 0:
                vjp_params = tf.convert_to_tensor(0., dtype=vjp_y[0].dype)
                vjp_params = move_to_device(vjp_params, vjp_y[0].device)

            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        if isinstance(grad_output, tf.Tensor) or isinstance(grad_output, tf.Variable):
            adj_y = [grad_output[-1]]
        else:
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        # adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        adj_params = tf.zeros_like(flat_params, dtype=flat_params.dtype)
        adj_time = move_to_device(tf.convert_to_tensor(0., dtype=t.dtype), t.device)
        time_vjps = []
        for i in range(T - 1, 0, -1):

            ans_i = tuple(ans_[i] for ans_ in ans)

            if isinstance(grad_output, tf.Tensor) or isinstance(grad_output, tf.Variable):
                grad_output_i = [grad_output[i]]
            else:
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)

            func_i = func(t[i], ans_i)
            func_i = cast_double(func_i)

            if not isinstance(func_i, Iterable):
                func_i = [func_i]

            # Compute the effect of moving the current time measurement point.
            dLd_cur_t = sum(
                tf.reshape(tf.matmul(tf.reshape(func_i_, [1, -1]), tf.reshape(grad_output_i_, [-1, 1])), [1])
                for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
            )
            adj_time = cast_double(adj_time)
            adj_time = adj_time - dLd_cur_t
            time_vjps.append(dLd_cur_t)

            # Run the augmented system backwards in time.
            if isinstance(adj_params, Iterable):
                count = _numel(adj_params)

                if count == 0:
                    adj_params = move_to_device(tf.convert_to_tensor(0., dtype=adj_y[0].dtype), adj_y[0].device)

            aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)

            aug_ans = odeint(
                augmented_dynamics,
                aug_y0,
                tf.convert_to_tensor([t[i], t[i - 1]]),
                rtol=rtol, atol=atol, method=method, options=options
            )

            # Unpack aug_ans.
            adj_y = aug_ans[n_tensors:2 * n_tensors]
            adj_time = aug_ans[2 * n_tensors]
            adj_params = aug_ans[2 * n_tensors + 1]

            adj_y = tuple(adj_y_[1] if _check_len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
            if _check_len(adj_time) > 0: adj_time = adj_time[1]
            if _check_len(adj_params) > 0: adj_params = adj_params[1]

            adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

            del aug_y0, aug_ans

        time_vjps.append(adj_time)
        time_vjps = tf.concat(time_vjps[::-1], 0)

        # reshape the parameters back into the correct variable shapes
        var_flat_lens = [_numel(v, dtype=tf.int32).numpy() for v in variables]
        var_shapes = [v.shape for v in variables]

        adj_params_splits = tf.split(adj_params, var_flat_lens)
        adj_params_list = [tf.reshape(p, v_shape)
                           for p, v_shape in zip(adj_params_splits, var_shapes)]
        return (*adj_y, time_vjps), adj_params_list

    return ans, grad


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, tf.keras.Model):
        raise ValueError('func is required to be an instance of tf.keras.Model')

    tensor_input = False
    if tf.debugging.is_numeric_tensor(y0):
        class TupleFunc(tf.keras.Model):

            def __init__(self, base_func, **kwargs):
                super(TupleFunc, self).__init__(dtype=tf.float64, **kwargs)
                self.base_func = base_func
            def call(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    # build the function to get its variables
    if not func.built:
        _ = func(t, y0)

    global _arguments
    _arguments = _Arguments(func, method, options, rtol, atol)
    ys = OdeintAdjointMethod(*y0, t)

    if tensor_input or type(ys) == tuple or type(ys) == list:
        ys = ys[0]

    return ys


# def adjoint_method(parameters, t0, t1, zt1, dLdzt1, func):
#     s0 = tf.stack([zt1, dLdzt1, tf.zeros(tf.shape(parameters))])
#
#     def aug_dynamics(t, zad):
#         z, a, _ = tf.unstack(tf.cast(zad, tf.float64))
#         dfdz = [[func.b]]
#         dfdparameters = [[z]]
#         print(tf.stack([tf.reshape(func(z, t),[1,1]), -tf.transpose([[a]])@dfdz,-tf.transpose([[a]])@dfdparameters]).shape)
#         return tf.stack([tf.reshape(func(z, t),[1,1]), -tf.transpose([[a]])@dfdz,-tf.transpose([[a]])@dfdparameters])
#     t = tf.linspace(t1, t0, 100)
#     [zt0, dLdz0, dLdparameters] = euler(aug_dynamics, s0, t)
#     return dLdzt0, dLdparameters
def exact_solution(a, b, T):
    return a/b*(np.exp(b*T)-1)

def exact_derivative(a, b, T):
    return a*(T/b*np.exp(b*T)-(np.exp(b*T)-1)/(b*b))

x_0 = tf.constant(0., dtype=tf.float64)
a = tf.constant(2., dtype=tf.float64)
b = tf.constant(2., dtype=tf.float64)
T = tf.constant(2., dtype=tf.float64)
t = tf.cast(tf.linspace(0., T, 10), dtype=tf.float64)

odemodel = ODE(a, b)
y_pred = np.array(euler(odemodel, x_0, t))
plt.scatter(np.array(t),y_pred)
plt.savefig('plots/ode.png')
plt.close()

sol = []
t0 = time.time()

with tf.device('/gpu:0'):
    for i in range(10):
        with tf.GradientTape(persistent=True) as g:
            g.watch(odemodel.b)
            y_sol = odeint_adjoint(odemodel, x_0, tf.cast(tf.linspace(0., T, 5), tf.float64))
            y_sol = y_sol[-1]
        dF_dB = tf.cast(g.gradient(y_sol, odemodel.b), dtype=tf.float64)
        # tf.print(dF_dB, a, odemodel.b, T)
        sol.append([a, odemodel.b, T, dF_dB])
sol = np.array(sol)
t1 = time.time()
print(t1-t0)
dYdX_adjoint = sol[:,-1]
dYdX_exact = exact_derivative(sol[:, 0], sol[:, 1], sol[:, 2])
print(dYdX_exact/dYdX_adjoint)
