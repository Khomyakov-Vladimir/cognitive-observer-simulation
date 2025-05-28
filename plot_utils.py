# plot_utils.py

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.signal import argrelextrema
from matplotlib.ticker import MaxNLocator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from config import (
    NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR,
    PLOT_DPI, PLOT_FIGSIZE, PLOT_ALPHA, EXTREMUM_PROMINENCE, SAVE_DERIVATIVE_DATA
)
import networkx as nx

# === ANALYTICAL UTILITIES ===

def compute_derivative(entropy, epsilon):
    """
    Computes the numerical derivative of the entropy S with respect to log(ε).
    This reflects the rate of change of cognitive complexity as a function of perceptual resolution.
    """
    log_eps = np.log(epsilon)
    dS_dlogeps = np.gradient(entropy, log_eps)
    return dS_dlogeps

def find_local_maxima(y_vals, order=2):
    """
    Identifies local maxima in the input array.
    Commonly used to detect phase transitions or scale shifts.
    """
    return argrelextrema(y_vals, np.greater, order=order)[0]

# === VISUALIZATION FUNCTIONS ===

def plot_entropy_graph(entropy_list, save_path, show=True):
    """
    Plots the entropy function S(ε) as a function of ε in logarithmic scale.
    Typically used to analyze the scale-dependent complexity of cognitive representations.
    """
    eps_values = [x[0] for x in entropy_list]
    entropies = [x[2] for x in entropy_list]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_path, _ = os.path.splitext(save_path)

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(eps_values, entropies, 'b-', marker='o', markersize=3, linewidth=1.5, label='Entropy S(ε)')
    plt.xscale('log')
    plt.xticks([1e-3, 1e-2, 1e-1], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.xlabel(r'$\epsilon$ (Perceptual resolution, log scale)', fontsize=12)
    plt.ylabel(r'Cognitive Entropy $S_\epsilon$', fontsize=12)
    plt.title('Cognitive Entropy vs. Perceptual Resolution (log scale)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(f"{base_path}.png", format='png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')

    if show and SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_2d_projection(Y_proj, title="Projection", save_path=None, show=True):
    """
    Plots a 2D projection of the cognitive representation space.
    This is typically used after dimensionality reduction (e.g., t-SNE or UMAP).
    """
    Y_proj = np.array(Y_proj)
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_proj[:, 0], Y_proj[:, 1], c='dodgerblue', s=10, alpha=0.7)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base_path, _ = os.path.splitext(save_path)
        plt.savefig(f"{base_path}.png", format='png', dpi=PLOT_DPI, bbox_inches='tight')
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')

    if show and SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_entropy_and_derivative(
    entropy_values,
    epsilon_values,
    save_path,
    smoothing_window=3,
    save_json=True
):
    """
    Plots both the entropy curve S(ε) and its derivative dS/dlog(ε).
    Highlights local extrema in the derivative, which may correspond to cognitive transitions or critical points.
    Optionally saves detected extrema data to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_path, _ = os.path.splitext(save_path)

    # Compute and smooth derivative
    dS = compute_derivative(entropy_values, epsilon_values)
    dS_smooth = np.convolve(dS, np.ones(smoothing_window) / smoothing_window, mode='same')

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)

    # Plot entropy
    ax1.plot(epsilon_values, entropy_values, 'b-', label='Entropy S(ε)', alpha=PLOT_ALPHA)
    ax1.set_xlabel(r'$\epsilon$ (Perceptual resolution, log scale)', fontsize=12)
    ax1.set_ylabel(r'$S_\epsilon$', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xscale("log")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot derivative on a secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epsilon_values, dS_smooth, 'r--', label=r"$\frac{dS}{d\log(\epsilon)}$", alpha=PLOT_ALPHA)
    ax2.set_ylabel(r"$\frac{dS}{d\log(\epsilon)}$", color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')

    # Identify and mark local extrema
    maxima_idx = find_local_maxima(dS_smooth, order=3)
    max_eps = epsilon_values[maxima_idx]
    max_S = entropy_values[maxima_idx]
    ax1.scatter(max_eps, max_S, color='green', marker='D', label='Extrema')

    fig.legend(loc='upper right')
    plt.title("Entropy $S(ε)$ and Its Derivative (log scale)", fontsize=14)
    plt.tight_layout()

    # Save plots in multiple formats
    plt.savefig(f"{base_path}.png", format='png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight')

    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Save extrema data to JSON
    if save_json:
        extrema_data = [
            {
                "epsilon": float(eps),
                "entropy": float(S),
                "dS_dlogeps": float(dS_val)
            }
            for eps, S, dS_val in zip(max_eps, max_S, dS_smooth[maxima_idx])
        ]
        json_path = base_path + ".json"
        with open(json_path, "w") as f:
            json.dump(extrema_data, f, indent=4)

    return max_eps, maxima_idx, dS_smooth
