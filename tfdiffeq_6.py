"""
Definition of ODENet and Augmented ODENet.
Ported from https://github.com/EmilienDupont/augmented-neural-odes/blob/master/anode/models.py
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environments.DoublePendulum import DoublePendulum
from environments.SinglePendulum import SinglePendulum
import datetime

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver
tf.keras.backend.set_floatx('float64')


def visualize_single_pendulum(model):
    x_t = np.stack([[[1.01, 10.]]])
    for i in range(500):
        x_t = np.concatenate([x_t, model.predict(x_t[:,-1:])[:1]], axis=1)
    x_t = x_t[0]
    for i in range(500):
        print(x_t[i])
    plt.scatter(x_t[:,0], x_t[:,1], c=tf.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.savefig('plots/phase_plot_theta_single_pendulum.png')
    plt.close()
    xp = np.linspace(-6, 6, 60)
    yp = np.zeros((60,))
    inp = np.vstack([xp, yp]).T
    inp = np.reshape(inp, (-1, 1, 2))
    preds = model.predict(inp)
    preds = np.reshape(preds[:,:,1], (-1))
    plt.scatter(xp, preds)
    plt.savefig('plots/dttheta_overtheta.png')

    x1 = np.sin(x_t[:, 0])
    y1 = -np.cos(x_t[:, 0])

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i]]
        thisy = [0, y1[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*0.01))
        return line, time_text
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    ani = animation.FuncAnimation(fig, animate, range(1, len(x_t)),
                                  interval=0.01*1000, blit=True, init_func=init)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2400)
    ani.save('plots/sp.mp4', writer=writer)

def visualize_double_pendulum(model):
    x_t = np.stack([[[1.01, 4., 0., 0.]]])
    for i in range(1000):
        x_t = np.concatenate([x_t, model.predict(x_t[:,-1:])[:1]], axis=1)
        print(x_t[0][i])
    x_t = x_t[0]
    plt.scatter(x_t[:,0], x_t[:,1], c=tf.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.savefig('plots/phase_plot_theta.png')
    plt.close()
    plt.scatter(x_t[:,2], x_t[:,3], c=tf.linspace(0., 255., x_t.shape[0]), cmap='magma')
    plt.savefig('plots/phase_plot_theta_dt.png')
    plt.close()

    x1 = np.sin(x_t[:, 0])
    y1 = -np.cos(x_t[:, 0])

    x2 = np.sin(x_t[:, 1]) + x1
    y2 = -np.cos(x_t[:, 1]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*0.01))
        return line, time_text
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    ani = animation.FuncAnimation(fig, animate, range(1, len(x_t)),
                                  interval=0.01*1000, blit=True, init_func=init)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('plots/dp.mp4', writer=writer)

def load_dataset_single_pendulum():
    x_train = np.load('single_pendulum_x_train.npy')
    x_train = np.reshape(x_train, (-1, 32, 2))
    y_train = np.load('single_pendulum_y_train.npy')
    y_train = np.reshape(y_train, (-1, 32, 2))
    x_val = np.load('single_pendulum_x_val.npy')
    y_val = np.load('single_pendulum_y_val.npy')
    return x_train, y_train, x_val, y_val

def load_dataset_double_pendulum():
    x_train = np.load('double_pendulum_x_train2.npy')
    x_train = np.reshape(x_train, (-1, 32, 4))
    y_train = np.load('double_pendulum_y_train2.npy')
    y_train = np.reshape(y_train, (-1, 32, 4))
    x_val = np.load('double_pendulum_x_val.npy')
    y_val = np.load('double_pendulum_y_val.npy')
    return x_train, y_train, x_val, y_val

def create_dataset_single_pendulum(N, samples_per_series=1024):
    pendulum = SinglePendulum(theta=1.01)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
    y_t = x_t[1:]
    x_t = x_t[:-1]
    x_t = np.expand_dims(x_t, axis=0)
    y_t = np.expand_dims(y_t, axis=0)
    x_t = np.reshape(x_t, (-1, 16, x_t.shape[-1]))
    y_t = np.reshape(y_t, (-1, 16, y_t.shape[-1]))
    for i in range(100):
        #1.01, 0.5
        pendulum = SinglePendulum(theta=np.random.random()*np.pi)
        delta_t = 0.01
        with tf.device('/gpu:0'):
            x_t_i = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
        y_t_i = x_t_i[1:]
        x_t_i = x_t_i[:-1]
        x_t_i = np.expand_dims(x_t_i, axis=0)
        y_t_i = np.expand_dims(y_t_i, axis=0)
        x_t_i = np.reshape(x_t_i, (-1, 16, x_t_i.shape[-1]))
        y_t_i = np.reshape(y_t_i, (-1, 16, y_t_i.shape[-1]))
        print(x_t.shape, x_t_i.shape)
        x_t = np.concatenate([x_t_i, x_t], axis=0)
        y_t = np.concatenate([y_t_i, y_t], axis=0)


    pendulum = SinglePendulum(theta=-0.8, theta_dt=0.5)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t_val = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
    y_t_val = x_t_val[1:]
    x_t_val = x_t_val[:-1]
    x_t_val = np.expand_dims(x_t_val, axis=0)
    y_t_val = np.expand_dims(y_t_val, axis=0)

    np.save('single_pendulum_x_train.npy', x_t)
    np.save('single_pendulum_y_train.npy', y_t)
    np.save('single_pendulum_x_val.npy', x_t_val)
    np.save('single_pendulum_y_val.npy', y_t_val)
    return x_t, y_t, x_t_val, y_t_val

def create_dataset_double_pendulum(N, samples_per_series=1000):
    pendulum = DoublePendulum(theta1=1.01, theta2=0.5)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
    y_t = x_t[1:]
    x_t = x_t[:-1]
    x_t = np.expand_dims(x_t, axis=0)
    y_t = np.expand_dims(y_t, axis=0)
    x_t = np.reshape(x_t, (-1, 16, 4))
    y_t = np.reshape(y_t, (-1, 16, 4))
    for i in range(100):
        #1.01, 0.5
        pendulum = DoublePendulum(theta1=np.random.random()*5, theta2=np.random.random()*5)
        delta_t = 0.01
        with tf.device('/gpu:0'):
            x_t_i = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
        y_t_i = x_t_i[1:]
        x_t_i = x_t_i[:-1]
        x_t_i = np.expand_dims(x_t_i, axis=0)
        y_t_i = np.expand_dims(y_t_i, axis=0)
        x_t_i = np.reshape(x_t_i, (-1, 16, 4))
        y_t_i = np.reshape(y_t_i, (-1, 16, 4))
        print(x_t.shape, x_t_i.shape)
        x_t = np.concatenate([x_t_i, x_t], axis=0)
        y_t = np.concatenate([y_t_i, y_t], axis=0)


    pendulum = DoublePendulum(theta1=-0.8, theta2_dt=0.5)
    delta_t = 0.01
    with tf.device('/gpu:0'):
        x_t_val = pendulum.step(dt=samples_per_series*delta_t, n_steps=samples_per_series+1)
    y_t_val = x_t_val[1:]
    x_t_val = x_t_val[:-1]
    x_t_val = np.expand_dims(x_t_val, axis=0)
    y_t_val = np.expand_dims(y_t_val, axis=0)
    np.save('double_pendulum_x_train2.npy', x_t)
    np.save('double_pendulum_y_train2.npy', y_t)
    np.save('double_pendulum_x_val.npy', x_t_val)
    np.save('double_pendulum_y_val.npy', y_t_val)
    return x_t, y_t, x_t_val, y_t_val


N = 1200
x_train, y_train, x_val, y_val = create_dataset_single_pendulum(N)
# x_train, y_train, x_val, y_val = load_dataset_double_pendulum()


model = Sequential()
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.00001), input_shape=(None, x_train.shape[-1])))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.00001)))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.00001)))
model.add(Dense(y_train.shape[-1], activation=None, kernel_regularizer=l2(0.00001)))

adam = Adam(lr=3e-4, clipnorm=1)

model.compile(optimizer=adam, loss='mse', metrics=['mae'])
print(model.summary())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

print(x_train.shape)
for epoch in range(50):
    model.fit(x_train, y_train,
              epochs=10*(epoch+1),
              validation_data=(x_val, y_val),
              callbacks=[tensorboard_callback],
              initial_epoch=10*epoch)
model.save('models/sp2.h5')
# model = load_model('models/dp.h5')
# visualize_single_pendulum(model)
visualize_double_pendulum(model)
