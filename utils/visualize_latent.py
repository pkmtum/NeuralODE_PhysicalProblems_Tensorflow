import numpy as np
import matplotlib.pyplot as plt
orig_trajs = np.load('experiments/latentpendulum/orig_trajs.npy')
orig_ts = np.load('experiments/latentpendulum/orig_ts.npy')
print(orig_trajs.shape)
print(orig_ts.shape)
print(orig_ts)
plt.scatter(orig_trajs[:,:,0], orig_trajs[:,:,1], alpha=0.01, s=1)
plt.show()
plt.savefig('latent_orig_traj.png')
plt.close()
samp_trajs = np.load('experiments/latentpendulum/samp_trajs.npy')
samp_ts = np.load('experiments/latentpendulum/samp_ts.npy')
print(samp_trajs.shape)
print(samp_ts.shape)
print(samp_ts)
plt.scatter(samp_trajs[:,:,0], samp_trajs[:,:,1], alpha=0.01, s=1)
plt.show()
plt.savefig('latent_orig_traj_samp.png')
plt.close()
