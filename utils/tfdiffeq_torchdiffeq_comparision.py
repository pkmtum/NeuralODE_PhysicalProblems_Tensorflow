"""
Compare results of tfdiffeq with torchdiffeq.
"""

import torch
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import VarianceScaling
from torchdiffeq import odeint_adjoint as torch_odeint
from tfdiffeq import odeint_adjoint as tf_odeint

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=12)
tf.random.set_seed(0)
np.random.seed(0)


class tfODEfunc(tf.keras.Model):
    def __init__(self):
        super(tfODEfunc, self).__init__()
        kernel_initializer = VarianceScaling(scale=1/3., distribution='uniform', seed=0)
        bias_initializer = VarianceScaling(scale=1/3.*1/9, distribution='uniform', seed=0)
        self.conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer)
        # Track number of function evaluations
        self.nfe = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, t, x):
        self.nfe.assign_add(1.)
        out = self.conv1(x)
        return out


class torchODEfunc(torch.nn.Module):
    def __init__(self):
        super(torchODEfunc, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, 1, 1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(x)
        return out


tf_t = tf.linspace(0., 1., 50)
torch_t = torch.linspace(0., 1., 50)

tf_y0 = tf.ones([1, 30, 30, 1])
torch_y0 = torch.ones([1, 1, 30, 30])

tf_func = tfODEfunc()
torch_func = torchODEfunc()

# Build the tf model so we can set its weights
tf_func(t=0, x=tf_y0)

torch_model = torch.nn.Sequential(torch_func)
with torch.no_grad():
    torch_model[0].conv1.weight = torch.nn.Parameter(
        torch.from_numpy(np.transpose(tf_func.conv1.get_weights()[0], [2, 3, 0, 1])))
    torch_model[0].conv1.bias = torch.nn.Parameter(
        torch.from_numpy(tf_func.conv1.get_weights()[1]))


with tf.GradientTape() as tape:
    tf_y = tf_odeint(tf_func, tf_y0, tf_t, method='euler')
    print(tf_func.nfe.numpy())
    tf_func.nfe.assign(0.)
    tf_grad = tape.gradient(tf.reduce_sum(tf_y[-1]), tf_func.trainable_variables)
    print(tf_func.nfe.numpy())

torch_y = torch_odeint(torch_func, torch_y0, torch_t, method='euler')
print(torch_func.nfe)
torch.sum(torch_y[-1]).backward()
print(torch_func.nfe)
torch_grad = [torch_model[0].conv1.weight.grad, torch_model[0].conv1.bias.grad]


torch_grad_np = torch_grad[0].numpy()
tf_grad_np = tf_grad[0].numpy()
torch_grad_np = np.reshape(torch_grad_np, (3, 3))
tf_grad_np = np.reshape(tf_grad_np, (3, 3))

# weight grad
print((torch_grad_np-tf_grad_np)/np.linalg.norm(torch_grad_np))
# bias grad
print(((torch_grad[1]-tf_grad[1])/np.linalg.norm(torch_grad[1])).numpy())

# Print integration result (equal)
print((tf.reduce_sum(tf_y[-1])-torch.sum(torch_y[-1]).detach().numpy()).numpy())
