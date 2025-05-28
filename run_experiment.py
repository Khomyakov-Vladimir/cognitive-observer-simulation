# run_experiment.py

import torch
import matplotlib.pyplot as plt
from cognitive_functor import CognitiveObserver
from trajectory_simulation import generate_trajectories
from metrics import compute_projection_scores
from plot_utils import plot_entropy_graph, plot_2d_projection
from config import NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR
import os
import numpy as np
import time


def generate_log_scale_epsilons(n=1000, x_min=0.001, x_max=0.1, save_path="log_scale_values.txt"):
    """
    Generates a list of ε (epsilon) values logarithmically spaced between x_min and x_max.
    These values serve as perceptual resolution levels for cognitive entropy evaluation.

    Args:
        n (int): Number of samples.
        x_min (float): Minimum value of ε.
        x_max (float): Maximum value of ε.
        save_path (str): Path to save the resulting list.

    Returns:
        List[float]: A list of logarithmically spaced ε values.
    """
    values = x_min * (x_max / x_min) ** (np.arange(n) / (n - 1))
    values_rounded = np.round(values, 12)
    result = ", ".join([f"{x:.12f}" for x in values_rounded])
    with open(save_path, "w") as file:
        file.write(result)
    return values_rounded.tolist()


def plot_kernel_size_vs_epsilon(entropy_list, save_path_base, show=True):
    """
    Plots the ε-kernel neighborhood size as a function of perceptual resolution ε.

    Args:
        entropy_list (List[Tuple[float, int, float]]): List containing (ε, kernel_size, entropy).
        save_path_base (str): Path base for saving plots.
        show (bool): Whether to display the plot interactively.
    """
    eps_values = [x[0] for x in entropy_list]
    kernel_sizes = [x[1] for x in entropy_list]

    plt.figure(figsize=(10, 6))
    plt.plot(eps_values, kernel_sizes, 'r-', marker='s', markersize=4, linewidth=1.5, label='Kernel Size')
    plt.xscale('log')
    plt.xticks([1e-3, 1e-2, 1e-1], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.xlabel(r'$\epsilon$ (Perceptual resolution, log scale)', fontsize=12)
    plt.ylabel(r'Kernel size $|K_\epsilon(x)|$', fontsize=12)
    plt.title('Kernel Size vs. Perceptual Resolution', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path_base}.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path_base}.png", format='png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_entropy_derivative(entropy_list, save_path_base, show=True):
    """
    Plots the derivative of cognitive entropy with respect to log(ε).

    Args:
        entropy_list (List[Tuple[float, int, float]]): List containing (ε, kernel_size, entropy).
        save_path_base (str): Path base for saving plots.
        show (bool): Whether to display the plot interactively.
    """
    eps_values = np.array([x[0] for x in entropy_list])
    entropies = np.array([x[2] for x in entropy_list])

    log_eps = np.log(eps_values)
    dS_dlogeps = np.gradient(entropies, log_eps)

    plt.figure(figsize=(10, 6))
    plt.plot(eps_values, dS_dlogeps, 'b-', linewidth=2.0, label=r'$\frac{dS}{d \log \epsilon}$')
    plt.xscale('log')
    plt.xlabel(r'$\epsilon$ (log scale)', fontsize=12)
    plt.ylabel(r'Derivative of Entropy', fontsize=12)
    plt.title('Derivative of Cognitive Entropy vs. ε', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path_base}.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path_base}.png", format='png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def main():
    # === Experiment configuration ===
    in_dim = 5
    hidden_dim = 36
    out_dim = 2
    num_points = 200
    noise_level = 0.03
    rounding_precision = 0.02
    seed = 42
    show_plots = SHOW_PLOTS

    save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # === Generate ε values (logarithmic scale) ===
    epsilons = generate_log_scale_epsilons(
        n=NUM_EPSILON_STEPS, x_min=EPSILON_MIN, x_max=EPSILON_MAX,
        save_path=os.path.join(save_dir, "log_scale_values.txt")
    )

    torch.manual_seed(seed)

    # === Initialize Cognitive Observer ===
    observer = CognitiveObserver(in_dim, hidden_dim, out_dim)

    # === Generate synthetic trajectories in input space ===
    print("[~] Generating input trajectories...")
    X = generate_trajectories(num_points=num_points, in_dim=in_dim)

    # === Perform observation (projection into cognitive space) ===
    print("[~] Computing observed projections...")
    Y_proj = observer.observe_batch(X, noise_level=noise_level, rounding_precision=rounding_precision)

    scores = compute_projection_scores(X, Y_proj)
    print("=== Projection Scores ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    # === Save 2D projection visualization ===
    proj_dir = os.path.join(save_dir, "projections")
    os.makedirs(proj_dir, exist_ok=True)
    proj_path = os.path.join(proj_dir, "projection.png")
    plot_2d_projection(Y_proj, title="Projected Observations", save_path=proj_path, show=show_plots)

    # === Evaluate cognitive entropy over the ε range ===
    print("[~] Evaluating cognitive entropy over ε range...")
    entropy_list = []
    for eps in epsilons:
        result = observer.compute_kernel_entropy(
            X,
            epsilon=eps,
            noise_level=noise_level,
            rounding_precision=rounding_precision
        )
        entropy_list.append((eps, result['kernel_size'], result['entropy']))
        print(f"ε = {eps:.4f} | Kernel size: {result['kernel_size']} | Entropy: {result['entropy']:.4f}")

    # === Create directory for entropy visualizations ===
    entropy_dir = os.path.join(save_dir, "entropy")
    os.makedirs(entropy_dir, exist_ok=True)

    # === Plot entropy vs ε ===
    entropy_plot_base = os.path.join(entropy_dir, "entropy_vs_epsilon")
    plot_entropy_graph(entropy_list, save_path=f"{entropy_plot_base}.png", show=show_plots)
    plt.close()

    # === Plot ε-kernel size vs ε ===
    kernel_plot_base = os.path.join(entropy_dir, "kernel_vs_epsilon")
    plot_kernel_size_vs_epsilon(entropy_list, save_path_base=kernel_plot_base, show=show_plots)

    # === Plot entropy derivative w.r.t. log(ε) ===
    derivative_plot_base = os.path.join(entropy_dir, "derivative_entropy_vs_logeps")
    plot_entropy_derivative(entropy_list, save_path_base=derivative_plot_base, show=show_plots)

    print("[✓] Experiment completed successfully.")


if __name__ == "__main__":
    main()
