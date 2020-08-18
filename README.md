# NeuralODE

This is the Code for the Bachelor's Thesis "Neural ODE for physical problems" by Tim Beyer.

All scripts should be run from the main trajectory, e.g.;
```
python3 experiments/densenet.py --system=airplane_lat_long
```

# Add your own systems

If you would like to test models on your own physical systems, simply create a file in the
'experiments/environments/' folder and add your configuration details to the 'environments.json'.
The file should implement a subclass of PhysicalSystem with the method ```call``` and optionally a staticmethod ```visualize```.
To see how these methods should work, see any of the built-in systems.

## Requirements
* TensorFlow 2.2
* tfdiffeq
* matplotlib
* (keras-mdn-layer) if you want to try mixture-density-layers on top of an lstm