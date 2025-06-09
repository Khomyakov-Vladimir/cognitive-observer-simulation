# visualization_decoherence.py

import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(x, y, xlabel, ylabel, title, filename, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    """
    Save a single plot as both PDF and PNG.

    Parameters:
        x (np.ndarray): Values for the x-axis.
        y (np.ndarray): Values for the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        filename (str): Base filename for saving (no extension).
        results_dir (str): Directory to save plots into.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    pdf_path = os.path.join(results_dir, filename + ".pdf")
    png_path = os.path.join(results_dir, filename + ".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_collapse(Lambda_values, collapse_probs, results_dir):
    """
    Plot the cognitive collapse probability curve.

    Parameters:
        Lambda_values (np.ndarray): Array of distinguishability scales.
        collapse_probs (np.ndarray): Corresponding collapse probabilities.
        results_dir (str): Directory to save the plot.
    """
    save_plot(
        Lambda_values,
        collapse_probs,
        xlabel="Cognitive distinguishability Λ",
        ylabel="Collapse probability p(|0⟩)",
        title="Phase transition in observer collapse probability",
        filename="plot_collapse",
        results_dir=results_dir
    )


def plot_entropy(Lambda_values, entropy_values, results_dir):
    """
    Plot the entropy curve under decoherence.

    Parameters:
        Lambda_values (np.ndarray): Array of distinguishability scales.
        entropy_values (np.ndarray): Corresponding entropy values.
        results_dir (str): Directory to save the plot.
    """
    save_plot(
        Lambda_values,
        entropy_values,
        xlabel="Cognitive distinguishability Λ",
        ylabel="Cognitive entropy S(Λ)",
        title="Entropy under decoherence",
        filename="plot_entropy",
        results_dir=results_dir
    )


def plot_decoherence_effect(Lambda_values, new_entropy, old_entropy, results_dir):
    """
    Plot the difference in entropy between decoherence and original models.

    Parameters:
        Lambda_values (np.ndarray): Array of distinguishability scales.
        new_entropy (np.ndarray): Entropy with decoherence.
        old_entropy (np.ndarray): Entropy without decoherence.
        results_dir (str): Directory to save the plot.
    """
    delta = new_entropy - old_entropy
    save_plot(
        Lambda_values,
        delta,
        xlabel="Cognitive distinguishability Λ",
        ylabel="ΔS (decoherence effect)",
        title="Decoherence effect ΔS = S_new - S_old",
        filename="plot_decoherence",
        results_dir=results_dir
    )


def plot_entropy_comparison(Lambda_values, new_entropy, old_entropy, results_dir):
    """
    Plot both entropy curves for direct comparison.

    Parameters:
        Lambda_values (np.ndarray): Array of distinguishability scales.
        new_entropy (np.ndarray): Entropy with decoherence.
        old_entropy (np.ndarray): Entropy without decoherence.
        results_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(Lambda_values, new_entropy, label="New entropy", lw=2)
    ax.plot(Lambda_values, old_entropy, label="Old entropy", lw=2, linestyle="--")
    ax.set_xlabel("Cognitive distinguishability Λ")
    ax.set_ylabel("Entropy S(Λ)")
    ax.set_title("Comparison of entropy models")
    ax.legend()
    ax.grid(True)
    pdf_path = os.path.join(results_dir, "plot_entropy_comparison.pdf")
    png_path = os.path.join(results_dir, "plot_entropy_comparison.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    x = np.linspace(0.5, 2.5, 50)
    y = np.sin(x)  # dummy data
    save_plot(x, y, "λ", "Entropy", "Sample Entropy Plot", "test_plot", "results")
    print("Test plot saved to results/test_plot.pdf and .png")
