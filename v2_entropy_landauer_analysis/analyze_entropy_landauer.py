# analyze_entropy_landauer.py

import pandas as pd
import numpy as np
import os
import shutil

def prepare_results_dir(path="results"):
    """
    Clears the output directory before use.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[INFO] Cleared previous contents of '{path}/'")
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Created fresh directory '{path}/'")

# === Constants ===
BOLTZMANN_CONSTANT = 1.380649e-23  # [J/K]
LN2 = np.log(2)
SCALE_ATTOJOULE = 1e18  # Conversion from joules to attojoules

def landauer_energy(T_kelvin: float) -> float:
    """
    Compute Landauer bound (energy per bit) in attojoules for given temperature in Kelvin.
    """
    return BOLTZMANN_CONSTANT * T_kelvin * LN2 * SCALE_ATTOJOULE

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # === Analyze the relation between cognitive entropy and Landauer's energy bound ===

    # Load and sort data
    df = pd.read_csv("entropy_kernel_vs_epsilon.csv")
    df = df.sort_values("epsilon")

    # Convert epsilon to temperature
    df["T"] = 1.0 / df["epsilon"]

    # Normalize entropy
    S_max = df["entropy"].max()
    df["entropy_normalized"] = df["entropy"] / S_max

    # Compute Landauer energy
    df["energy_aJ"] = landauer_energy(df["T"])
    E_max = df["energy_aJ"].max()
    df["energy_normalized"] = df["energy_aJ"] / E_max

    # Find intersection point between normalized entropy and energy
    diff = np.abs(df["entropy_normalized"] - df["energy_normalized"])
    idx_intersection = diff.idxmin()
    row_intersection = df.loc[idx_intersection]

    # Discrimination threshold: T* where dS/dT is maximal
    dS_dT = np.gradient(df["entropy_normalized"], df["T"])
    idx_T_star = np.argmax(np.abs(dS_dT))
    T_star = df["T"].iloc[idx_T_star]
    S_star = df["entropy_normalized"].iloc[idx_T_star]

    # Display results
    print(f"[INFO] Discrimination threshold: T* = {T_star:.2f} K, S*(normalized) = {S_star:.3f}")
    print(f"[INFO] Intersection with Landauer bound: T = {row_intersection['T']:.2f} K, log10(epsilon) = {np.log10(row_intersection['epsilon']):.3f}")
    print(f"[INFO] Energy at intersection: E ~= {row_intersection['energy_aJ']:.3f} aJ")

    # Save CSV
    file_csv = "results/entropy_landauer_analysis.csv"
    df.to_csv(file_csv, index=False)
    print(f"[INFO] Saved CSV: {file_csv}")

    # Save to .npz for plotting
    np.savez("results/entropy_landauer_data.npz",
             epsilon=df["epsilon"].values,
             T=df["T"].values,
             entropy=df["entropy_normalized"].values,
             energy=df["energy_normalized"].values)
    print(f"[INFO] Saved NPZ: results/entropy_landauer_data.npz")

if __name__ == "__main__":
    main()
