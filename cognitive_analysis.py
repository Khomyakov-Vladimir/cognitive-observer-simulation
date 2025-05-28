# cognitive_analysis.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
import networkx as nx

from config import NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR


# === Core Computational Utilities ===

def compute_derivative(entropy, epsilon):
    """
    Computes the derivative of the entropy curve with respect to log-scaled ε.
    """
    log_eps = np.log(epsilon)
    return np.gradient(entropy, log_eps)


def find_local_maxima(y_vals, order=2):
    """
    Identifies local maxima in a given sequence using the specified neighborhood order.
    """
    return argrelextrema(y_vals, np.greater, order=order)[0]


# === Entropy Curve Visualization and Extremum Analysis ===

def plot_entropy_graph(entropy_values, epsilon_values, save_path_base, smoothing_window=3, show: bool = SHOW_PLOTS):
    """
    Plots the entropy S(ε) alongside its logarithmic derivative dS/dlog(ε),
    highlights local extrema, and stores them in a JSON file.
    """
    base_dir = os.path.dirname(save_path_base)

    dS = compute_derivative(entropy_values, epsilon_values)
    dS_smooth = np.convolve(dS, np.ones(smoothing_window) / smoothing_window, mode='same')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epsilon_values, entropy_values, 'b-', label='Entropy S(ε)')
    ax1.set_xlabel("ε (scale)")
    ax1.set_ylabel("Entropy S", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xscale("log")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(epsilon_values, dS_smooth, 'r--', label="dS/dlog(ε)")
    ax2.set_ylabel("dS/dlog(ε)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    maxima_idx = find_local_maxima(dS_smooth, order=3)
    max_eps = epsilon_values[maxima_idx]
    max_S = entropy_values[maxima_idx]

    extrema_data = [
        {"epsilon": float(eps), "entropy": float(S), "dS_dlogeps": float(dS)}
        for eps, S, dS in zip(max_eps, max_S, dS_smooth[maxima_idx])
    ]
    with open(os.path.join(base_dir, "entropy_maxima.json"), "w") as f:
        json.dump(extrema_data, f, indent=4)

    ax1.scatter(max_eps, max_S, color='green', marker='D', label='Detected Extrema')

    fig.legend(loc='upper right')
    plt.title("Entropy S(ε) and Its Logarithmic Derivative")
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300)
    plt.savefig(save_path_base + ".pdf", dpi=300)
    if show:
        plt.show()
    plt.close()

    return max_eps, maxima_idx, dS_smooth


def plot_extremum_heatmap(projections_list, extrema_indices, save_path_base, show: bool = SHOW_PLOTS):
    """
    Plots a heatmap of entropy extrema points in the t-SNE projection space.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    all_points = []

    for idx in extrema_indices:
        if 0 <= idx < len(projections_list):
            proj = projections_list[idx]
            all_points.append(proj)

    all_points = np.vstack(all_points)
    ax.scatter(all_points[:, 0], all_points[:, 1], c='red', s=60, alpha=0.8, label='Entropy Extremum')
    ax.legend()
    plt.title('Entropy Extremum Points in Projection Space')
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300)
    plt.savefig(save_path_base + ".pdf", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_tsne_colored_by_derivative(projections_list, dS_list, save_path_base, show: bool = SHOW_PLOTS):
    """
    Visualizes t-SNE projections colored by normalized entropy derivative values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    dS_norm = (dS_list - np.min(dS_list)) / (np.max(dS_list) - np.min(dS_list))
    projections = np.vstack(projections_list)

    scatter = ax.scatter(
        projections[:, 0], projections[:, 1],
        c=np.repeat(dS_norm, projections_list[0].shape[0]),
        cmap='viridis', s=40, alpha=0.7
    )

    plt.colorbar(scatter, ax=ax, label='Normalized dS/dlog(ε)')
    plt.title('t-SNE Projection Colored by Entropy Derivative')
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300)
    plt.savefig(save_path_base + ".pdf", dpi=300)
    if show:
        plt.show()
    plt.close()


def build_extremum_graph(projections_list, extrema_indices, save_path_base, show: bool = SHOW_PLOTS):
    """
    Constructs a graph connecting entropy extrema based on Euclidean distances in projection space.
    """
    nodes = []

    for idx in extrema_indices:
        if 0 <= idx < len(projections_list):
            coord = projections_list[idx].mean(axis=0)
            nodes.append(coord)

    nodes = np.array(nodes)
    dist_matrix = euclidean_distances(nodes)
    G = nx.Graph()

    for i in range(len(nodes)):
        G.add_node(i, pos=(nodes[i][0], nodes[i][1]))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            G.add_edge(i, j, weight=dist_matrix[i, j])

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='orange', edge_color='gray', node_size=300)
    plt.title("Graph of Entropy Extrema in Projection Space")
    plt.tight_layout()
    plt.savefig(save_path_base + ".png", dpi=300)
    plt.savefig(save_path_base + ".pdf", dpi=300)
    if show:
        plt.show()
    plt.close()


def get_projection_list(all_obs, reducer=None):
    """
    Applies dimensionality reduction (default: t-SNE) to a list of observation matrices.
    """
    if reducer is None:
        reducer = TSNE(n_components=2, perplexity=30)
    return [reducer.fit_transform(obs) for obs in all_obs]


# === Public API ===

__all__ = [
    'NUM_EPSILON_STEPS',
    'compute_derivative',
    'plot_entropy_graph',
    'plot_extremum_heatmap',
    'plot_tsne_colored_by_derivative',
    'build_extremum_graph',
    'get_projection_list'
]
