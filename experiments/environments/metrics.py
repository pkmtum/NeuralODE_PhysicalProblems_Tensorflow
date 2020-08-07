"""Functions to compute the physics-based metrics.
"""
import numpy as np


def relative_energy_drift(energy_pred, energy_true):
    """Computes the relative energy drift of x_pred w.r.t. x_true
    # Arguments:
        energy_pred: float - predicted energy
        energy_true: float - reference energy
    """
    return (energy_pred-energy_true) / energy_true


def relative_phase_error(x_pred, x_val, check=True):
    """Computes the relative phase error of x_pred w.r.t. x_true.
    Finds the locations of the zero crossings in both signals,
    then compares corresponding crossings to each other.
    # Arguments:
        x_pred: numpy.ndarray shape=(n_datapoints) - predicted time series
        x_true: numpy.ndarray shape=(n_datapoints) - reference time series
        check: bool - set phase error to NaN if the crossings are too different
    """
    ref_crossings = zero_crossings(x_val)
    pred_crossings = zero_crossings(x_pred)
    t_ref = np.mean(np.diff(ref_crossings)) * 2
    t_pred = np.mean(np.diff(pred_crossings)) * 2
    phase_error = t_ref/t_pred - 1
    if check and len(pred_crossings) < len(ref_crossings) - 2:
        phase_error = np.nan
    return phase_error


def trajectory_error(x_pred, x_val):
    return np.mean(np.abs(x_pred - x_val))


def zero_crossings(x):
    """Find indices of zeros crossings"""
    return np.array(np.where(np.diff(np.sign(x)))[0])
