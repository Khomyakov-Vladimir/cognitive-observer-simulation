# analysis_utils.py

import numpy as np
import scipy.signal
from scipy.ndimage import uniform_filter1d


def compute_derivative(entropy, epsilon):
    """
    Computes the derivative of entropy with respect to log(epsilon),
    i.e., dS / d(log epsilon), which is often used in scale-space analysis.

    Parameters:
        entropy (np.ndarray): Array of entropy values S(ε).
        epsilon (np.ndarray): Array of scale parameters ε.

    Returns:
        np.ndarray: The derivative dS/dlog(ε), same shape as entropy.
    """
    log_eps = np.log(epsilon)
    dS_dlogeps = np.gradient(entropy, log_eps)
    return dS_dlogeps


def smooth(values, window=3):
    """
    Applies a uniform moving average filter for smoothing input values.

    Parameters:
        values (np.ndarray): Input array to smooth.
        window (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed array of the same shape.
    """
    return uniform_filter1d(values, size=window)


def find_local_maxima(y_vals, order=3):
    """
    Identifies local maxima in a 1D array using neighborhood comparison.

    Parameters:
        y_vals (np.ndarray): Input array.
        order (int): Number of points to compare on each side.

    Returns:
        np.ndarray: Indices of local maxima.
    """
    return scipy.signal.argrelextrema(np.array(y_vals), np.greater, order=order)[0]


def extract_extrema(entropy, epsilon, window=3, order=3):
    """
    Identifies significant scale extrema by computing the smoothed derivative
    of entropy over logarithmic scale and finding its local maxima.

    Parameters:
        entropy (np.ndarray): Entropy values S(ε).
        epsilon (np.ndarray): Scale parameters ε.
        window (int): Window size for smoothing.
        order (int): Neighborhood order for extremum detection.

    Returns:
        tuple:
            max_eps (np.ndarray): Epsilon values at local maxima of dS/dlog(ε).
            max_S (np.ndarray): Entropy values at those scales.
            dS_smooth (np.ndarray): Smoothed derivative curve.
    """
    dS = compute_derivative(entropy, epsilon)
    dS_smooth = smooth(dS, window=window)
    maxima_indices = find_local_maxima(dS_smooth, order=order)
    max_eps = epsilon[maxima_indices]
    max_S = entropy[maxima_indices]
    return max_eps, max_S, dS_smooth
