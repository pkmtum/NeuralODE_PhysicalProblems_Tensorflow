"""Generate all plots from the adjoint_float_test.py and adjoint_float_test_nn.py data.
These two files have to be run before this script can be executed.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Generate plots for the simple example
csv = pd.read_csv('plots/dtype/adjoint_float_test.csv', header=0)
# Pick out correct rows
f32_adj = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'adjoint')]
f32_bp = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'backprop')]
f64_adj = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'adjoint')]
f64_bp = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'backprop')]
# Plot the relative error over the requested tolerance
plt.plot(f32_adj['rtol'], f32_adj['error'], 'o-', label='f32, adjoint')
plt.plot(f32_bp['rtol'], f32_bp['error'], 'o-', label='f32, backprop')
plt.plot(f64_adj['rtol'], f64_adj['error'], 'o-', label='f64, adjoint')
plt.plot(f64_bp['rtol'], f64_bp['error'], 'o-', label='f64, backprop')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xlabel('rtol')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('plots/dtype/adjoint_float_test_rtol_error.png')
plt.close()
# Plot the relative error over wall time
plt.plot(f32_adj['fwd_pass']+f32_adj['bwd_pass'], f32_adj['error'], 'o-', label='f32, adjoint')
plt.plot(f32_bp['fwd_pass']+f32_bp['bwd_pass'], f32_bp['error'], 'o-', label='f32, backprop')
plt.plot(f64_adj['fwd_pass']+f64_adj['bwd_pass'], f64_adj['error'], 'o-', label='f64, adjoint')
plt.plot(f64_bp['fwd_pass']+f64_bp['bwd_pass'], f64_bp['error'], 'o-', label='f64, backprop')
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel('wall time')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('plots/dtype/adjoint_float_test_walltime_error.png')
plt.close()

# Generate plots for the neural network example
csv = pd.read_csv('plots/dtype/adjoint_float_test_nn.csv', header=0)
# Pick correct rows
f32_adj = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'adjoint')]
f32_bp = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'backprop')]
f64_adj = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'adjoint')]
f64_bp = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'backprop')]
# Plot relative error vs requested tolerance
plt.plot(f32_adj['rtol'], f32_adj['error'], 'o-', label='f32, adjoint')
plt.plot(f32_bp['rtol'], f32_bp['error'], 'o-', label='f32, backprop')
plt.plot(f64_adj['rtol'], f64_adj['error'], 'o-', label='f64, adjoint')
plt.plot(f64_bp['rtol'], f64_bp['error'], 'o-', label='f64, backprop')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xlabel('rtol')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('plots/dtype/adjoint_float_test_nn_rtol_error.png')
plt.close()
# Plot relative error over wall time
plt.plot(f32_adj['fwd_pass']+f32_adj['bwd_pass'], f32_adj['error'], 'o-', label='f32, adjoint')
plt.plot(f32_bp['fwd_pass']+f32_bp['bwd_pass'], f32_bp['error'], 'o-', label='f32, backprop')
plt.plot(f64_adj['fwd_pass']+f64_adj['bwd_pass'], f64_adj['error'], 'o-', label='f64, adjoint')
plt.plot(f64_bp['fwd_pass']+f64_bp['bwd_pass'], f64_bp['error'], 'o-', label='f64, backprop')
plt.legend(loc="upper left")
ax = plt.gca()
ax.set_xlabel('wall time')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('plots/dtype/adjoint_float_test_nn_walltime_error.png')
plt.close()
