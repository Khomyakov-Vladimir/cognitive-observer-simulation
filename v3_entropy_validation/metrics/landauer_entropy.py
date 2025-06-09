#!/usr/bin/env python3

"""
landauer_entropy.py

This script computes the Landauer bound energy based on probabilistic outputs
of the observer simulation under different noise levels ε (epsilon). The input
is expected in the form of a CSV file, where each row corresponds to a fixed ε
and contains softmax probabilities of cognitive state clustering.

The resulting Landauer bound energy, proportional to the information entropy
S(ε), is saved as both a CSV and a PDF plot.

⚠️ Note:
This script is independent from `visualization_decoherence.py`
because it operates on the noise parameter ε instead of cognitive
distinguishability Λ, and performs metric validation post-simulation.

Expected input file:
    simulation_outputs/probabilities_vs_epsilon.csv
    Columns: [epsilon, p₀, p₁, ..., pₖ]

Generated outputs:
    metrics/verification_plots/landauer_vs_epsilon.csv
    metrics/verification_plots/landauer_vs_epsilon.pdf

Author: Vladimir Khomyakov
Project: cognitive-observer-simulation / v3_entropy_validation
Date: 2025-06
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constant
kT_log2 = 1.0  # normalized units: k * T * ln(2)

# Resolve paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "../simulation_outputs/probabilities_vs_epsilon.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "verification_plots")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "landauer_vs_epsilon.csv")
PDF_OUTPUT = os.path.join(OUTPUT_DIR, "landauer_vs_epsilon.pdf")

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy H(p) = -Σ p log₂ p for each row of probabilities.

    Parameters:
        probs (np.ndarray): 2D array of shape (n_samples, n_classes)

    Returns:
        np.ndarray: entropy values of shape (n_samples,)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_probs, axis=1)
    return entropy

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read input probabilities
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    if "epsilon" not in df.columns:
        print("❌ Missing 'epsilon' column in input CSV.")
        return

    epsilon_values = df["epsilon"].values
    prob_columns = df.drop(columns=["epsilon"]).values

    if not np.allclose(prob_columns.sum(axis=1), 1.0, atol=1e-3):
        print("⚠️ Warning: Some rows do not sum to 1.0 (softmax output expected).")

    entropy_values = compute_entropy(prob_columns)
    landauer_energy = entropy_values * kT_log2

    # Save CSV
    out_df = pd.DataFrame({
        "epsilon": epsilon_values,
        "entropy": entropy_values,
        "landauer_energy": landauer_energy
    })
    out_df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Saved entropy data to {CSV_OUTPUT}")

    # Plot
    plt.figure()
    plt.plot(epsilon_values, landauer_energy, lw=2)
    plt.xlabel("Noise level ε")
    plt.ylabel("Landauer bound energy (arb. units)")
    plt.title("Landauer bound vs noise ε")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PDF_OUTPUT)
    print(f"✅ Saved plot to {PDF_OUTPUT}")
    plt.close()

if __name__ == "__main__":
    main()
