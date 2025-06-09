# observer_simulator_decoherence.py
# Observer simulation with and without decoherence
# Part of the Cognitive Observer framework

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from scipy.special import softmax


def simulate_decoherence_observer(lambda_values, steps=100, seed=42):
    """
    Simulates the observer's entropy under a decoherence-based probabilistic model.
    """
    np.random.seed(seed)
    entropy_values = []

    for lam in tqdm(lambda_values, desc="Simulating Decoherence"):
        prob = 0.5 + 0.5 * np.tanh(lam - 1.0)
        samples = np.random.binomial(1, prob, size=steps)
        p = np.mean(samples)
        entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
        entropy_values.append(entropy)

    return np.array(entropy_values)


def simulate_original_observer(lambda_values, steps=100, seed=123):
    """
    Simulates the observer's entropy using the original sigmoid-based model.
    """
    np.random.seed(seed)
    entropy_values = []

    for lam in tqdm(lambda_values, desc="Simulating Original Observer"):
        prob = 1.0 / (1.0 + np.exp(-lam))
        samples = np.random.binomial(1, prob, size=steps)
        p = np.mean(samples)
        entropy = -p * np.log(p + 1e-9) - (1 - p) * np.log(1 - p + 1e-9)
        entropy_values.append(entropy)

    return np.array(entropy_values)


def compute_collapse_probability(lambda_values):
    """
    Computes the phenomenological collapse probability.
    """
    return 0.5 + 0.5 * np.tanh(lambda_values - 1.0)


def sample_observer_model(epsilon, num_clusters=4, seed=42):
    """
    Generates softmax probabilities for an artificial observer model given ε.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=1.0, scale=1.0 / (epsilon + 1e-6), size=num_clusters)
    return softmax(logits)


def generate_softmax_probabilities(epsilon_values, model, save_path, num_clusters=4):
    """
    Generates softmax probabilities for a range of ε values and saves them to CSV.
    """
    all_probs = []

    for eps in tqdm(epsilon_values, desc="Generating Softmax Probabilities"):
        probs = model(eps, num_clusters=num_clusters)
        all_probs.append(probs)

    df = pd.DataFrame(all_probs, columns=[f"p_{i}" for i in range(num_clusters)])
    df.insert(0, "epsilon", epsilon_values)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


def simulate_observer_trajectory(timesteps=100, seed=None):
    """
    Simulates a single 2D latent trajectory of an observer.

    Parameters:
        timesteps (int): Number of time steps in the trajectory.
        seed (int or None): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (timesteps, 2) representing latent cognitive states.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, size=timesteps))  # 1D Brownian motion
    y = np.cumsum(rng.normal(0, 1, size=timesteps))
    return np.stack([x, y], axis=1)


if __name__ == "__main__":
    lambdas = np.linspace(0.5, 2.5, 10)
    entropies_dec = simulate_decoherence_observer(lambdas)
    entropies_orig = simulate_original_observer(lambdas)
    print("Decoherence entropies:", entropies_dec)
    print("Original entropies:", entropies_orig)
