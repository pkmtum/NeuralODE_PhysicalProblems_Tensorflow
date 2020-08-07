"""
Single Pendulum experiment.
"""
from environments.SinglePendulum import SinglePendulum
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import argparse
import logging
import time
from torchdiffeq import odeint_adjoint as odeint


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['densenet', 'odenet'], default='odenet')
args = parser.parse_args()

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
PLOT_DIR = 'plots/single_pendulum/' + args.network + '/t'


class modelFunc(nn.Module):
    """Converts a standard tf.keras.Model to a model compatible with odeint."""

    def __init__(self, model):
        super(modelFunc, self).__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(torch.unsqueeze(x, dim=0))[0]


def visualize_single_pendulum_model(model, epoch=0):
    """Visualize a tf.keras.Model for a single pendulum.
       Writes
    """
    dt = 0.01
    x0 = torch.tensor([1.01, 5.], dtype=torch.float32, device=device)
    model_func = modelFunc(model)
    x_t = odeint(model_func, x0, torch.linspace(0., 10., int(10./dt)).to(x0))
    x_t = x_t.cpu().detach().numpy()

    ref_pendulum = SinglePendulum(x0=tf.constant([1.01, 5.]))
    x_t_ref = np.array(ref_pendulum.step(dt=999*0.01, n_steps=1000))
    plt.close()
    plt.scatter(x_t_ref[:, 0], x_t_ref[:, 1], c=np.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'phase_plot_theta_single_pendulum_ref.png'.format(epoch))
    plt.close()
    plt.scatter(x_t[:, 0], x_t[:, 1], c=np.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'phase_plot_theta_single_pendulum{}.png'.format(epoch))
    plt.close()

    xp = np.linspace(-6.3, 6.3, 60)
    yp = np.linspace(-6.3, 6.3, 60)
    xpv, ypv = np.meshgrid(xp, yp)
    xpv = np.reshape(xpv, (-1))
    ypv = np.reshape(ypv, (-1))
    inp = np.vstack([xpv, ypv]).T
    inp = np.reshape(inp, (-1, 2))

    preds = model(torch.tensor(inp).to(x0)).cpu().detach().numpy()
    preds = np.reshape(preds, (-1, 2))
    u_pred = preds[:, 0]
    v_pred = preds[:, 1]
    plt.quiver(xpv, ypv, u_pred, v_pred)
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'quiver_plot{}.png'.format(epoch))
    plt.close()

    u_true = inp[:,1]
    v_true =  -ref_pendulum.g/ref_pendulum.l*np.sin(inp[:,0])

    # plt.scatter(x_train[:,0], x_train[:,1], s=2, alpha=0.01)
    plt.quiver(xpv, ypv, u_true, v_true)
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'quiver_plot_ref.png'.format(epoch))
    plt.close()
    x_dif = u_pred-u_true
    y_dif = v_pred-v_true
    abs_dif = np.sqrt(np.abs(x_dif)+np.abs(y_dif))
    plt.contourf(xp, yp, np.reshape(np.clip(abs_dif, 0., 3.0), (len(xp), len(yp))), 100)
    plt.colorbar()
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'quiver_plot_{}_vs_ref_abs.png'.format(epoch))
    plt.close()

    rel_dif = np.clip(abs_dif / np.sqrt(np.square(u_true)+np.square(v_true)), 0, 1)
    plt.contourf(xp, yp, np.reshape(rel_dif, (len(xp), len(yp))), 100)
    plt.colorbar()
    plt.xlabel('theta')
    plt.ylabel('theta_dt')
    plt.savefig(PLOT_DIR + 'quiver_plot_{}_vs_ref_rel.png'.format(epoch))
    plt.close()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def load_dataset_single_pendulum():
    x_train = np.load('experiments/datasets/single_pendulum_x_train.npy')
    y_train = np.load('experiments/datasets/single_pendulum_y_train.npy')
    x_val = np.load('experiments/datasets/single_pendulum_x_val.npy')
    y_val = np.load('experiments/datasets/single_pendulum_y_val.npy')
    return x_train, y_train, x_val, y_val


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, augment_dim=0, time_dependent=True, **kwargs):
        super(ODEFunc, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.dense1 = nn.Linear(hidden_dim+1, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        self.nfe += 1
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        out = self.dense1(ttx)
        return self.relu(out)


class ODEBlock(nn.Module):

    def __init__(self, odefunc, tol=1e-3, solver='dopri5', **kwargs):
        """
        Solves ODE defined by odefunc.
        # Arguments:
            odefunc : ODEFunc instance or Conv2dODEFunc instance
                Function defining dynamics of system.
            is_conv : bool
                If True, treats odefunc as a convolutional model.
            tol : float
                Error tolerance.
            solver: ODE solver. Defaults to DOPRI5.
        """
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol
        self.method = solver

        if solver == "dopri5":
            self.options = {'max_num_steps': MAX_NUM_STEPS}
        else:
            self.options = None

    def forward(self, x, training=None, eval_times=None, **kwargs):
        """
        Solves ODE starting from x.
        # Arguments:
            x: Tensor. Shape (batch_size, self.odefunc.data_dim)
        # Returns:
            Output tensor of forward pass.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.nfe = 0
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=self.tol, atol=self.tol, method=self.method,
                     options=self.options)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODENet(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(ODENet, self).__init__()
        self.dense1 = nn.Linear(output_dim, hidden_dim)
        odefunc = ODEFunc(hidden_dim+0, augment_dim=0)
        self.odeblock = ODEBlock(odefunc, solver='dopri5')
        self.dense2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.dense1(x)
        out = self.odeblock(out)
        out = self.dense2(out)
        return out


N = 1200
device = torch.device('cuda:0')

x_train, y_train, x_val, y_val = load_dataset_single_pendulum()
criterion = nn.MSELoss().to(device)
model = nn.Sequential(ODENet(hidden_dim=8, output_dim=y_train.shape[-1])).to(device)
logger = logging.getLogger()
level = logging.INFO
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)
logger.addHandler(console_handler)
logger.info(model)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-2, weight_decay=0.00001)

loss_meter = RunningAverageMeter()
batch_time_meter = RunningAverageMeter()
print(x_train.shape)
for epoch in range(10):
    for itr in range(0, len(x_train)//16, 16):
        print(itr)
        end = time.time()
        x, y = x_train[itr*16:(itr+1)*16], y_train[itr*16:(itr+1)*16]
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        batch_time_meter.update(time.time() - end)

    with torch.no_grad():
        val_loss = criterion(model(torch.tensor(x_val, dtype=torch.float32, device=device)),
                             torch.tensor(y_val, dtype=torch.float32, device=device))
        train_loss = criterion(model(torch.tensor(x_train, dtype=torch.float32, device=device)),
                               torch.tensor(y_train, dtype=torch.float32, device=device))

        logger.info(
            "Epoch {:04d} | Time {:.3f} ({:.3f}) | "
            "Train Loss {:.4f} | Test Loss {:.4f}".format(
                epoch, batch_time_meter.val, batch_time_meter.avg,
                train_loss, val_loss
            )
        )

# model.save_weights('single_pendulum.h5')
    visualize_single_pendulum_model(model, epoch)
