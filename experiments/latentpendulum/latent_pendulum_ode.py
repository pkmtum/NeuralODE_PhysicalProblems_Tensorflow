import os
from tfdiffeq import move_to_device
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=768)])
import matplotlib.pyplot as plt
import argparse

import matplotlib
import numpy as np
import numpy.random as npr

matplotlib.use('agg')
parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='./experiments/latentpendulum/')
args = parser.parse_args()

TIME_OF_RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

if args.adjoint:
    from tfdiffeq import odeint_adjoint as odeint
else:
    from tfdiffeq import odeint


class RecognitionRNN(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = tf.keras.layers.Dense(nhidden, activation='tanh')
        self.h2o = tf.keras.layers.Dense(latent_dim * 2)

    def call(self, x, h):
        combined = tf.concat((x, h), axis=1)
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return tf.zeros([self.nbatch, self.nhidden], dtype=tf.float32)


class LatentODEfunc(tf.keras.Model):

    def __init__(self, output_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nhidden, activation='elu')
        self.fc2 = tf.keras.layers.Dense(nhidden, activation='elu')
        self.fc3 = tf.keras.layers.Dense(output_dim)
        self.nfe = tf.Variable(0., trainable=False)
        self.nbe = tf.Variable(0., trainable=False)

    def call(self, t, x):
        self.nfe.assign_add(1.)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nhidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(obs_dim)

    def call(self, z):
        out = self.fc1(z)
        out = self.fc2(out)
        return out


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

def total_energy(state):
    l = 1.
    g = 9.81
    if len(state.shape) == 1: # (state)
        return (1-tf.math.cos(state[0]))*l*g+tf.square(state[1])*0.5
    elif len(state.shape) == 2: #(batch, state)
        return (1-tf.math.cos(state[:, 0]))*l*g+tf.square(state[:, 1])*0.5
    else:
        raise ShapeError('NDIM must be 1 or 2 but is {}'.format(len(state.shape)))

def zero_crossings(x):
    return np.array(np.where(np.diff(np.sign(x)))[0])

def evaluate_physical_properties(model, iter=0):
    dt = 0.01
    # Extrapolation
    print(x_val.shape)
    rec.nbatch = 1
    h = rec.initHidden()
    for t in reversed(range(x_val.shape[1])):
        obs = x_val[:, t, :]
        out, h = rec(obs, h)

    z0 = out[:, :latent_dim]
    print(z0.shape)
    with tf.device('/gpu:0'):
        z_t = tf.transpose(odeint(func, z0, tf.linspace(0., 10., int(10./dt)+1), rtol=1.5e-8), [1, 0, 2]).numpy()
    x_t = dec(z_t)
    print(x_t.shape)
    print('Total Energy Reference (t=0):', total_energy(x_val[0,0]))
    print('Total Energy Reference (t=999):', total_energy(x_val[0,999]))
    print('Total Energy Pred (t=0):', total_energy(x_t[0,0]))
    extrapolation_energy = total_energy(x_t[0,999])
    print('Total Energy Pred (t=999):', extrapolation_energy)
    plt.plot(np.arange(int(10./dt)+1), np.array([total_energy(x_) for x_ in x_t[0,:]]))
    plt.savefig('./plots/latent_single_pendulum/energy{}'.format(iter))
    plt.close()
    if zero_crossings(x_t[0,:,0]).shape[0] == 3:
        extrapolation_phase_error = zero_crossings(x_val[0,:1025,0])[-1]-zero_crossings(x_t[0,:,0])[-1]
    else:
        extrapolation_phase_error = None
    print('Phase error third zero crossing ext:', extrapolation_phase_error)

def log_normal_pdf(x, mean, logvar):
    const = tf.convert_to_tensor(np.array([2. * np.pi]), dtype=tf.float32)
    const = move_to_device(const, device)
    const = tf.math.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / tf.math.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """Computes KL-divergence between two normal distributions (mean, log of variance)"""
    v1 = tf.math.exp(lv1)
    v2 = tf.math.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def save_states(orig_ts, x_val, samp_ts, x_train):
    ots = orig_ts.numpy()
    otjs = x_val.numpy()
    sts = samp_ts.numpy()
    stjs = x_train.numpy()

    orig_ts_path = os.path.join(args.train_dir, 'orig_ts')
    x_val_path = os.path.join(args.train_dir, 'x_val')
    samp_ts_path = os.path.join(args.train_dir, 'samp_ts')
    x_train_path = os.path.join(args.train_dir, 'x_train')

    np.save(orig_ts_path, ots)
    np.save(x_val_path, otjs)
    np.save(samp_ts_path, sts)
    np.save(x_train_path, stjs)


def restore_states():
    x_train = np.load('experiments/datasets/single_pendulum_x_train.npy')
    x_train = np.reshape(x_train, (-1, 1025, 2))
    new_x_train = x_train[:,:nsample]
    for i in range(x_train.shape[0]):
        t0 = np.argmax(npr.multinomial(1., [1./(1025-nsample)] * (1025-nsample))) + nsample
        new_x_train[i] = x_train[i, t0:t0+nsample]
    x_train = new_x_train
    x_val = np.load('experiments/datasets/single_pendulum_x_val.npy')
    x_val = np.reshape(x_val, (-1, 1025, 2))
    ts = np.linspace(0, 10.24, 1025)

    ots = tf.convert_to_tensor(ts, dtype=tf.float32)
    otjs = tf.convert_to_tensor(x_val, dtype=tf.float32)
    sts = tf.convert_to_tensor(ts[:nsample], dtype=tf.float32)
    stjs = tf.convert_to_tensor(x_train, dtype=tf.float32)

    states = dict(orig_ts=ots, x_val=otjs,
                  samp_ts=sts, x_train=stjs)

    return states


if __name__ == '__main__':
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 51
    nsample = 128//4
    gamma = 0.
    noise_std = 1.0
    device = 'gpu:' + str(args.gpu) if len(tf.config.experimental.list_physical_devices('GPU')) else 'cpu'
    with tf.device(device):
        # generate toy spiral data

        x_train = np.load('experiments/datasets/single_pendulum_x_train.npy')
        x_train = np.reshape(x_train, (-1, 1025, 2))
        new_x_train = x_train[:,:nsample]
        for i in range(x_train.shape[0]):
            t0 = np.argmax(npr.multinomial(1., [1./(1025-nsample)]*(1025-nsample)))#np.argmax(npr.multinomial(1., [1./(1025-2*nsample)] * (1025-2*nsample))) + nsample
            new_x_train[i] = x_train[i, t0:t0+nsample]
        x_train = new_x_train

        x_val = np.load('experiments/datasets/single_pendulum_x_val.npy')
        x_val = np.reshape(x_val, (-1, 1025, 2))
        x_val = x_val
        ts = np.linspace(0, 10.24, 1025)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        samp_ts = tf.convert_to_tensor(ts[:nsample], dtype=tf.float32)
        orig_ts = tf.convert_to_tensor(ts, dtype=tf.float32)
        x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)

        # model
        func = LatentODEfunc(latent_dim, nhidden)
        rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral)
        dec = Decoder(latent_dim, obs_dim, nhidden)
        optimizer = tf.keras.optimizers.Adam(args.lr)
        loss_meter = RunningAverageMeter()
        nfe_meter = RunningAverageMeter()
        nbe_meter = RunningAverageMeter()

        saver = tf.train.Checkpoint(func=func, dec=dec, optimizer=optimizer)

        if args.train_dir is not None:
            if not os.path.exists(args.train_dir):
                os.makedirs(args.train_dir)
            else:
                if tf.compat.v1.train.checkpoint_exists(args.train_dir):
                    path = tf.train.latest_checkpoint(args.train_dir)

                    if path is not None:
                        saver.restore(path)

                        states = restore_states()
                        x_val = states['x_val']
                        x_train = states['x_train']
                        orig_ts = states['orig_ts']
                        samp_ts = states['samp_ts']
                        print('Loaded ckpt from {}'.format(path))

        for itr in range(1, args.niters + 1):
            func.nfe.assign(0.)
            rec.nbatch = nspiral
            # backward in time to infer q(z_0)
            with tf.GradientTape(persistent=True) as tape:
                h = rec.initHidden()
                for t in reversed(range(x_train.shape[1])):
                    obs = x_train[:, t, :]
                    out, h = rec(obs, h)
                # z0 = x_train[:, 0, :] # <- no encoder, just x0 as z0
                # z0 = out[:, :latent_dim] # <- non-variational autoencoder
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                z0 = tf.random.normal(shape=[*qz0_mean.shape.as_list()],
                                      mean=qz0_mean,
                                      stddev=tf.math.exp(.5 * qz0_logvar),
                                      dtype=qz0_mean.dtype)

                # forward in time and solve ode for reconstructions
                pred_z = tf.transpose(odeint(func, z0, samp_ts, rtol=1.5e-8), [1, 0, 2])
                pred_x = dec(pred_z)
                # compute loss
                noise_std_ = tf.zeros(pred_x.shape, dtype=tf.float32) + noise_std
                noise_logvar = 2. * tf.math.log(noise_std_)
                logpx = tf.math.reduce_sum(log_normal_pdf(
                    x_train, pred_x, noise_logvar), axis=-1)
                logpx = tf.math.reduce_sum(logpx, axis=-1)

                pz0_mean = pz0_logvar = tf.zeros(z0.shape, dtype=tf.float32)
                analytic_kl = tf.math.reduce_sum(normal_kl(qz0_mean, qz0_logvar,
                                                           pz0_mean, pz0_logvar), axis=-1)
                loss = tf.math.reduce_mean(-logpx + analytic_kl, axis=0)

                # mse_loss = tf.math.reduce_sum(tf.math.square(x_train-pred_x), axis=-1)
                # discounted_mse_loss = tf.math.reduce_mean(tf.math.exp(-tf.linspace(0., gamma, pred_x.shape[1]))*mse_loss)
                # loss = discounted_mse_loss

            params = (list(func.trainable_variables) + list(dec.trainable_variables) + list(rec.trainable_variables))
            grad = tape.gradient(loss, params)
            grad_vars = zip(grad, params)



            optimizer.apply_gradients(grad_vars)
            nfe_meter.update(func.nfe.numpy())
            func.nfe.assign(0.)
            nbe_meter.update(func.nbe.numpy())
            loss_meter.update(loss.numpy())

            print('Iter: {}, running avg loss: {:.4f}, cur. loss: {:.4f}'.format(itr, loss_meter.avg, loss))
            print('NFE: {}, NBE: {:.4f}'.format(nfe_meter.avg, nbe_meter.avg))

            with open('./plots/latent_single_pendulum/' + TIME_OF_RUN + str(args.adjoint) + str(args.lr) + 'g' + str(gamma) + 'run.csv','a') as f:
                string = "{},{},{}\n".format(itr, loss_meter.val, nfe_meter.val)
                f.write(string)

            if itr != 0 and (itr + 1) % 100 == 0:
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt')

                    saver.save(ckpt_path)
                    save_states(orig_ts, x_val, samp_ts, x_train)
                    print('Stored ckpt at {}'.format(ckpt_path))
                    evaluate_physical_properties(func, itr)

            if (itr+1) % 100 == 0:
                optimizer.learning_rate.assign(optimizer.learning_rate.numpy()*0.1)

            if args.visualize and itr % 10 == 0:
                # sample from trajectorys' approx. posterior
                rec.nbatch = 1
                h = rec.initHidden()
                for t in reversed(range(x_val.shape[1])):
                    obs = x_val[:, t, :]
                    out, h = rec(obs, h)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = tf.convert_to_tensor(np.random.randn(
                    *qz0_mean.shape.as_list()), dtype=tf.float32)
                z0 = epsilon * tf.math.exp(.5 * qz0_logvar) + qz0_mean

                # z0 = out[:, :latent_dim]#x_val[:, 0, :]
                orig_ts = tf.convert_to_tensor(orig_ts, dtype=tf.float32)

                # take first trajectory for visualization
                z0 = z0[0:1]

                ts_pos = np.linspace(0., 10.24, num=1025)
                ts_neg = np.linspace(-10.24, 0., num=1025)[::-1].copy()
                ts_pos = tf.convert_to_tensor(ts_pos, dtype=tf.float32)
                ts_neg = tf.convert_to_tensor(ts_neg, dtype=tf.float32)

                zs_pos = odeint(func, z0, ts_pos, rtol=1.5e-8)
                zs_neg = odeint(func, z0, ts_neg, rtol=1.5e-8)
                # xs_pos = zs_pos
                # xs_neg = zs_neg
                xs_pos = dec(zs_pos)
                xs_neg = tf.reverse(dec(zs_neg), axis=[0])

                xs_pos = xs_pos.numpy().squeeze(1)
                xs_neg = xs_neg.numpy().squeeze(1)
                orig_traj = x_val[0].numpy()
                samp_traj = x_train[0].numpy()

                # xs_neg = np.clip(xs_neg, xs_pos.min(), xs_pos.max())
                plt.figure()
                plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                         'g', label='true trajectory')
                plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                         label='learned trajectory (t>0)')
                plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
                         label='learned trajectory (t<0)')
                plt.scatter(samp_traj[:, 0], samp_traj[
                            :, 1], label='sampled data', s=3)
                plt.legend(loc='upper left')
                plt.savefig('./plots/latent_single_pendulum/vis{}.png'.format(itr), dpi=250)
                plt.close()
                print('Saved visualization figure at {}'.format('./plots/latent_single_pendulum/vis{}.png'.format(itr)))
                # xp = np.linspace(-10., 10., 60)
                # yp = np.linspace(-10., 10., 60)
                # xpv, ypv = np.meshgrid(xp, yp)
                # xpv = np.reshape(xpv, (-1))
                # ypv = np.reshape(ypv, (-1))
                # inp = np.vstack([xpv, ypv]).T
                # inp = np.reshape(inp, (-1, 2))
                #
                # preds = func(0., inp)
                # preds = np.reshape(preds, (-1, 2))
                # u_pred = preds[:, 0]
                # v_pred = preds[:, 1]
                # plt.quiver(xpv, ypv, u_pred, v_pred)
                # plt.xlabel('x')
                # plt.ylabel('x_dt')
                # plt.savefig('./plots/latent_single_pendulum/quiver_plot{}.png'.format(itr), dpi=250)
                # plt.close()
        print('Training complete after {} iters.'.format(itr))
