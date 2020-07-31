# NeuralODE

This is the Code for the Bachelor's Thesis "Neural ODE for physical problems" by Tim Beyer.

All scripts should be run from the main trajectory, e.g.;
```
python3 experiments/single_pendulum/densenet.py
```

## Requirements
* TensorFlow 2.2
* tfdiffeq
* matplotlib
* (keras-mdn-layer) if you want to try mixture-density-layers on top of an lstm