# plot_discrimination_vs_landauer_v3_annotated.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === Constants ===
k_B = 1.380649e-23  # [J/K]
ln2 = np.log(2)

# === Logging function ===
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# === Main function ===
def main():
    data_path = os.path.join("results", "entropy_landauer_data.npz")

    if not os.path.exists(data_path):
        log(f"[ERROR] File not found: {data_path}")
        return

    log(f"Loading data from {data_path}...")
    data = np.load(data_path)

    required_keys = {"epsilon", "T", "entropy", "energy"}
    if not required_keys.issubset(set(data.keys())):
        log(f"[ERROR] Missing required keys in NPZ: {required_keys}")
        return

    epsilon = data["epsilon"]
    T = data["T"]
    entropy = data["entropy"]
    energy = data["energy"]

    log("Data loaded. Preparing plot...")

    log_eps = np.log10(epsilon)
    E_aJ = k_B * T * ln2 * 1e18

    # Critical points (from previous analysis)
    T_star = 914.63
    eps_star = T_star / 1e7
    log_eps_star = np.log10(eps_star)
    S_star = np.interp(log_eps_star, log_eps, entropy)

    intersection_log_eps = -2.676
    intersection_T = 10 ** intersection_log_eps * 1e7
    intersection_E = k_B * intersection_T * ln2 * 1e18
    intersection_S = np.interp(intersection_log_eps, log_eps, entropy)

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(log_eps, entropy, label="Normalized Cognitive Entropy", color="blue")
    ax1.set_xlabel(r"$\log_{10}(\varepsilon)$", fontsize=12)
    ax1.set_ylabel("Normalized Entropy", color="blue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Vertical lines
    ax1.axvline(log_eps_star, color="dimgray", linestyle="--", linewidth=1.5, label="$T^*$ (Threshold)")
    ax1.axvline(intersection_log_eps, color="darkgreen", linestyle=":", linewidth=2, label="Landauer Intersection")

    # Annotations
    ax1.annotate(r"$T^*$", xy=(log_eps_star, S_star), xytext=(log_eps_star+0.3, S_star+0.1),
                 arrowprops=dict(arrowstyle="->", color="dimgray"), fontsize=10, color="dimgray")
    ax1.annotate("Landauer", xy=(intersection_log_eps, intersection_S), 
                 xytext=(intersection_log_eps-0.6, intersection_S+0.1),
                 arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=10, color="darkgreen")

    # Right axis for energy in aJ
    ax2 = ax1.twinx()
    ax2.plot(log_eps, E_aJ, color="forestgreen", linewidth=2.0, label="Landauer Energy", alpha=0.7)
    ax2.set_ylabel("Energy per bit [aJ]", color="forestgreen", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="forestgreen")

    # Layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ax1.legend(
        loc='upper left',
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.05, 0.98)
    )
    plt.title("Cognitive Entropy and Landauer Limit with Annotations", fontsize=14, color="black")

    output_pdf = os.path.join("results", "plot_discrimination_vs_landauer_v3_annotated.pdf")
    output_png = os.path.join("results", "plot_discrimination_vs_landauer_v3_annotated.png")
    plt.savefig(output_pdf)
    plt.savefig(output_png, dpi=600)
    plt.close()

    log(f"Annotated plot saved to {output_pdf}")

if __name__ == "__main__":
    main()
