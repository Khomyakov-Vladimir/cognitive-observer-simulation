"""
validate_pcs_tds_tsne.py

Validation module for Projected Cognitive Space (PCS), Trajectory Discrimination Score (TDS),
and t-SNE-based visualization of cognitive trajectories. This script is part of the
v3_entropy_validation/metrics pipeline, designed to empirically assess the stability and
discriminability of observer representations under variable cognitive precision.

Key Objectives:
---------------
1. Simulate cognitive observer trajectories under different levels of projection noise (epsilon).
2. Evaluate the inter-trajectory vs. intra-trajectory separability using the Trajectory Discrimination Score (TDS).
3. Visualize the embedding structure of projected trajectories in 2D using t-SNE.
4. Store and analyze metric values (TDS vs epsilon) for inclusion in empirical entropy validation.
5. Ensure reproducibility and deterministic behavior under fixed random seeds.

Usage:
------
This script can be run as a standalone module for batch validation:

    $ python validate_pcs_tds_tsne.py --timesteps 100 --n_runs 10 --seed 42 --csv tds_vs_epsilon.csv

Output:
-------
- PDF files with t-SNE plots for each epsilon value (saved in `verification_plots/`).
- CSV file (`tds_vs_epsilon.csv`) with TDS metrics for different projection granularities.
- Summary plot (`tds_vs_epsilon.pdf`) showing how TDS varies with epsilon.

Scientific Context:
-------------------
The validation procedure captures how cognitive rounding (quantified by ε) affects the representational
separability of observer states. A high TDS implies robust discriminability despite projection-induced
noise, while low TDS indicates representational collapse or excessive overlap. t-SNE plots complement
this by revealing structural changes in the low-dimensional embedding space.

Author: Vladimir Khomyakov
Project: cognitive-observer-simulation / v3_entropy_validation
Date: 2025-06
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import os
import logging
import argparse
import csv
import sys
from pathlib import Path

# Validate required module
if not os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "observer_simulator_decoherence.py")):
    logging.critical("Missing required module: observer_simulator_decoherence.py")
    sys.exit(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from observer_simulator_decoherence import simulate_observer_trajectory

# Setup logging for reproducibility and traceability
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Directory to save verification plots
VERIFICATION_DIR = Path(__file__).resolve().parent / "verification_plots"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def simulate_projected_trajectories(epsilon: float, n_runs: int = 10, timesteps: int = 100, seed: int = 42) -> np.ndarray:
    """
    Simulate multiple projected observer trajectories under a given projection granularity epsilon.

    Parameters:
        epsilon (float): Projection granularity parameter (cognitive precision).
        n_runs (int): Number of independent observer simulations.
        timesteps (int): Number of time steps per trajectory.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (n_runs, timesteps, 2) with 2D projected trajectories.
    """
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(n_runs):
        latent_states = simulate_observer_trajectory(timesteps=timesteps)
        noise = rng.normal(scale=epsilon, size=latent_states.shape)
        projected = np.round(latent_states + noise)
        trajectories.append(projected)
    return np.array(trajectories)


def compute_tds(trajectories: np.ndarray, delta: float = 1e-8) -> float:
    """
    Compute the Trajectory Discrimination Score (TDS), which quantifies inter- vs. intra-trajectory separation.

    Parameters:
        trajectories (np.ndarray): Array of shape (n_runs, timesteps, 2).
        delta (float): Stability constant to avoid division by zero.

    Returns:
        float: The TDS value.
    """
    n_runs, timesteps, _ = trajectories.shape
    flattened = trajectories.reshape(n_runs, -1)
    dists = pairwise_distances(flattened)
    intra = np.mean([dists[i, i] for i in range(n_runs)])
    inter = np.mean([dists[i, j] for i in range(n_runs) for j in range(n_runs) if i != j])
    return inter / (intra + delta)


def plot_tsne(trajectories: np.ndarray, epsilon: float):
    """
    Visualize projected trajectories using t-SNE embedding.

    Parameters:
        trajectories (np.ndarray): Array of shape (n_runs, timesteps, 2).
        epsilon (float): Projection granularity parameter (used in plot title).
    """
    n_runs, timesteps, _ = trajectories.shape
    flattened = trajectories.reshape(n_runs, -1)
    X_embedded = TSNE(n_components=2, perplexity=5, learning_rate='auto', init='random').fit_transform(flattened)

    plt.figure(figsize=(6, 5))
    for i in range(n_runs):
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], label=f'Traj {i}', alpha=0.7)
    plt.title(r"$\mathrm{t}$-SNE of Projected Trajectories ($\varepsilon = %.3f$)" % epsilon)
    plt.xlabel(r"$\mathrm{tSNE}_1$")
    plt.ylabel(r"$\mathrm{tSNE}_2$")
    plt.grid(True)
    plt.tight_layout()
    filename = f"tsne_epsilon_{epsilon:.3f}.pdf"
    plt.savefig(VERIFICATION_DIR / filename, format='pdf')
    plt.close()
    logging.info(f"Saved t-SNE plot: {filename}")


def save_tds_to_csv(epsilons, tds_scores, filepath):
    """
    Save TDS scores and corresponding epsilons to CSV.

    Parameters:
        epsilons (list): List of epsilon values.
        tds_scores (list): Corresponding TDS values.
        filepath (str): Path to output CSV file.
    """
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epsilon", "tds"])
        for eps, tds in zip(epsilons, tds_scores):
            writer.writerow([eps, tds])
    logging.info(f"Saved TDS metrics to CSV: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Validate PCS, TDS and t-SNE under varying projection precision.")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of timesteps per trajectory")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of observer trajectories to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--csv", type=str, default="tds_vs_epsilon.csv", help="Output CSV file for TDS values")
    args = parser.parse_args()

    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    tds_scores = []

    for eps in epsilons:
        logging.info(f"Simulating for epsilon = {eps}...")
        trajs = simulate_projected_trajectories(epsilon=eps, n_runs=args.n_runs, timesteps=args.timesteps, seed=args.seed)
        tds = compute_tds(trajs)
        tds_scores.append(tds)
        logging.info(f"TDS (epsilon={eps}): {tds:.4f}")
        plot_tsne(trajs, epsilon=eps)

    # Save CSV and plot summary
    csv_path = VERIFICATION_DIR / args.csv
    save_tds_to_csv(epsilons, tds_scores, csv_path)

    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, tds_scores, marker='o', color='navy')
    plt.title(r"$\mathrm{Trajectory\ Discrimination\ Score\ vs.\ Projection\ Granularity}$")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\mathrm{TDS}$")
    plt.grid(True)
    plt.tight_layout()
    summary_path = VERIFICATION_DIR / "tds_vs_epsilon.pdf"
    plt.savefig(summary_path, format='pdf')
    plt.close()
    logging.info(f"Saved TDS vs epsilon plot: {summary_path}")


# Pytest-compatible test: verify that TDS is stable for fixed seed

def test_tds_reproducibility():
    seed = 123
    eps = 0.1
    t1 = compute_tds(simulate_projected_trajectories(eps, seed=seed))
    t2 = compute_tds(simulate_projected_trajectories(eps, seed=seed))
    assert abs(t1 - t2) < 1e-6, f"TDS values not consistent: {t1} vs {t2}"


if __name__ == "__main__":
    main()
