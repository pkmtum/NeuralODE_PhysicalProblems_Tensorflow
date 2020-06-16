import numpy as np
import matplotlib.pyplot as plt
x_train = np.load('experiments/datasets/airplane_x_train.npy')
x_train = np.reshape(x_train, (-1, 4))
#
# c = np.arange(len(x_train))
# np.random.shuffle(c)
# x_train = x_train[c[::int(100/1)]]
plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.1, s=3)
plt.savefig('train_long_period.png')
plt.close()
plt.scatter(x_train[:, 2], x_train[:, 3], alpha=0.1, s=3)
plt.savefig('train_short_period.png')
plt.close()
x_val = np.load('experiments/datasets/airplane_x_val.npy')
x_val = np.reshape(x_val, (-1, 4))

plt.scatter(x_val[:, 0], x_val[:, 1], s=3)
plt.savefig('val_lp.png')
plt.close()
plt.scatter(x_val[:, 2], x_val[:, 3], s=3)
plt.savefig('val_sp.png')
plt.close()
plt.scatter(x_train[:, 0], x_train[: ,1], s=4, alpha=0.8, label='Training')
plt.scatter(x_val[:, 0], x_val[:, 1], s=4, label='Testing')
# plt.xlim([-6.3, 6.3])
# plt.ylim([-6.3, 6.3])
plt.xlabel('V')
plt.ylabel('gamma')
plt.legend(loc="upper left")
plt.savefig('trainval.png')
plt.close()
