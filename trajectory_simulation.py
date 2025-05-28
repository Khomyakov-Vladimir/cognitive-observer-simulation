# trajectory_simulation.py

import torch
import numpy as np
from config import NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR


def generate_trajectories(num_points=300, in_dim=5, noise_std=0.05, mode='spiral'):
    """
    Generates synthetic trajectories in an `in_dim`-dimensional input space.

    Parameters:
        num_points (int): Number of points in the trajectory.
        in_dim (int): Dimensionality of the input space.
        noise_std (float): Standard deviation of Gaussian noise added to the trajectory.
        mode (str): Type of trajectory to generate. Options are:
            - 'spiral': Multidimensional spiral with increasing amplitude.
            - 'random': Random walk (cumulative Gaussian noise).
            - 'linear': Linearly increasing components.
            - 'chaotic': Chaotic-like sinusoidal interactions.

    Returns:
        torch.Tensor: A tensor of shape [num_points, in_dim] representing the generated trajectory.
    """
    t = torch.linspace(0, 4 * np.pi, steps=num_points)

    def spiral():
        return torch.stack([
            torch.sin(t + i * np.pi / in_dim) * (1 + 0.1 * i * t)
            for i in range(in_dim)
        ], dim=1)

    def random():
        return torch.randn(num_points, in_dim).cumsum(dim=0)

    def linear():
        return torch.stack([t + i for i in range(in_dim)], dim=1)

    def chaotic():
        return torch.stack([
            torch.sin(t * (i + 1)) * torch.cos(t * (i + 2))
            for i in range(in_dim)
        ], dim=1)

    generators = {
        'spiral': spiral,
        'random': random,
        'linear': linear,
        'chaotic': chaotic,
    }

    if mode not in generators:
        raise ValueError(f"Unknown mode: {mode}")

    X = generators[mode]()

    noise = noise_std * torch.randn_like(X)
    return X + noise
