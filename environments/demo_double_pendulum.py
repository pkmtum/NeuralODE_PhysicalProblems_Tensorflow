"""
Creates a demo for the DoublePendulum environment.
Two phase-plots are generated.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from DoublePendulum import DoublePendulum

dp = DoublePendulum(theta1=1.01, theta2=0.5)
with tf.device('/gpu:0'):
    xt = dp.step(dt=10, n_steps=1001)

plt.scatter(xt[:,0], xt[:,1], c=tf.linspace(0., 255., xt.shape[0]), cmap='magma')
plt.savefig('plots/phase_plot_theta.png')
plt.close()
plt.scatter(xt[:,2], xt[:,3], c=tf.linspace(0., 255., xt.shape[0]), cmap='magma')
plt.savefig('plots/phase_plot_theta_dt.png')
plt.close()

x1 = tf.math.sin(xt[:, 0])
y1 = -tf.math.cos(xt[:, 0])

x2 = tf.math.sin(xt[:, 1]) + x1
y2 = -tf.math.cos(xt[:, 1]) + y1

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


ani = animation.FuncAnimation(fig, animate, range(1, len(xt)),
                              interval=0.01*1000, blit=True, init_func=init)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani.save('plots/double_pendulum.mp4', writer=writer)
