# NeuralODE

This is the Code for the Bachelor's Thesis "Neural ODE for physical problems" by Tim Beyer.

All scripts should be run from the main trajectory, e.g.;
```
python3 experiments/densenet.py --system=airplane_lat_long
```

# Add your own systems

If you would like to test models on your own system, simply create a file in the
'environments' folder and add your configuration to the 'environments.json' file.

## Requirements
* TensorFlow 2.2
* tfdiffeq
* matplotlib
* (keras-mdn-layer) if you want to try mixture-density-layers on top of an lstm