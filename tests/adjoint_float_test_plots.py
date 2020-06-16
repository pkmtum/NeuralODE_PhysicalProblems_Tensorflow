import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('tests/adjoint_float_test.csv', header=0)
plt.scatter(csv['rtol'][::4], csv['error'][::4], label='f32, adjoint')
plt.scatter(csv['rtol'][1::4], csv['error'][1::4], label='f32, backprop')
plt.scatter(csv['rtol'][2::4], csv['error'][2::4], label='f64, adjoint')
plt.scatter(csv['rtol'][3::4], csv['error'][3::4], label='f64, backprop')
plt.legend(loc="upper left")

ax = plt.gca()
ax.set_xlabel('rtol')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('tests/adjoint_float_test_rtol_error.png')
plt.close()
plt.scatter(csv['fwd_pass'][::4]+csv['bwd_pass'][::4], csv['error'][::4], label='f32, adjoint')
plt.scatter(csv['fwd_pass'][1::4]+csv['bwd_pass'][1::4], csv['error'][1::4], label='f32, backprop')
plt.scatter(csv['fwd_pass'][2::4]+csv['bwd_pass'][2::4], csv['error'][2::4], label='f64, adjoint')
plt.scatter(csv['fwd_pass'][3::4]+csv['bwd_pass'][3::4], csv['error'][3::4], label='f64, backprop')
plt.legend(loc="upper right")

ax = plt.gca()
ax.set_xlabel('wall time')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('tests/adjoint_float_test_walltime_error.png')
plt.close()





csv = pd.read_csv('tests/adjoint_float_test_nn.csv', header=0)

f32_adj = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'adjoint')]
f32_bp = csv.loc[(csv['dtype'] == '<dtype: \'float32\'>') & (csv['method'] == 'backprop')]
f64_adj = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'adjoint')]
f64_bp = csv.loc[(csv['dtype'] == '<dtype: \'float64\'>') & (csv['method'] == 'backprop')]

plt.scatter(f32_adj['rtol'], f32_adj['error'], label='f32, adjoint')
plt.scatter(f32_bp['rtol'], f32_bp['error'], label='f32, backprop')
plt.scatter(f64_adj['rtol'], f64_adj['error'], label='f64, adjoint')
plt.scatter(f64_bp['rtol'], f64_bp['error'], label='f64, backprop')
plt.legend(loc="upper left")

ax = plt.gca()
ax.set_xlabel('rtol')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('tests/adjoint_float_test_nn_rtol_error.png')
plt.close()

plt.scatter(f32_adj['fwd_pass']+f32_adj['bwd_pass'], f32_adj['error'], label='f32, adjoint')
plt.scatter(f32_bp['fwd_pass']+f32_bp['bwd_pass'], f32_bp['error'], label='f32, backprop')
plt.scatter(f64_adj['fwd_pass']+f64_adj['bwd_pass'], f64_adj['error'], label='f64, adjoint')
plt.scatter(f64_bp['fwd_pass']+f64_bp['bwd_pass'], f64_bp['error'], label='f64, backprop')

plt.legend(loc="upper left")

ax = plt.gca()
ax.set_xlabel('wall time')
ax.set_ylabel('rel. error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.grid(True)
plt.savefig('tests/adjoint_float_test_nn_walltime_error.png')
plt.close()
