import torch
import torch.nn as nn
import math
from config import NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR


class CognitiveObserver(nn.Module):
    """
    Cognitive Functor F: H_ont → C_obs.
    Models subjective perception: projection, noise, quantization, and entropy estimation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def add_noise(self, x: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """
        Adds Gaussian noise to the input tensor.
        """
        if noise_level > 0.0:
            noise = torch.randn_like(x, device=x.device) * noise_level
            return x + noise
        return x

    def apply_rounding(self, x: torch.Tensor, rounding_precision: float = 0.0) -> torch.Tensor:
        """
        Applies element-wise quantization (cognitive rounding) to the tensor.
        """
        if rounding_precision > 0.0:
            return torch.round(x / rounding_precision) * rounding_precision
        return x

    def observe_batch(
        self,
        X: torch.Tensor,
        noise_level: float = 0.0,
        rounding_precision: float = 0.0
    ) -> torch.Tensor:
        """
        Simulates the perception process on a batch of inputs:
        projection → noise → cognitive rounding.
        """
        with torch.no_grad():
            X_noisy = self.add_noise(X, noise_level=noise_level)
            Y = self.forward(X_noisy)
            Y_rounded = self.apply_rounding(Y, rounding_precision=rounding_precision)
            return Y_rounded

    def compute_kernel_entropy(
        self,
        X: torch.Tensor,
        epsilon: float = 1e-2,
        noise_level: float = 0.0,
        rounding_precision: float = 0.0
    ) -> dict:
        """
        Estimates cognitive entropy based on ε-neighborhoods.
        Counts the number of pairs (i,j) such that ||Y_i - Y_j|| < ε
        and returns the log-count as a surrogate for entropy.
        """
        Y = self.observe_batch(X, noise_level=noise_level, rounding_precision=rounding_precision)
        dists = torch.cdist(Y, Y, p=2)
        N = Y.shape[0]

        # Ignore self-distances (diagonal)
        mask = ~torch.eye(N, dtype=torch.bool, device=Y.device)
        close_pairs = (dists < epsilon)[mask].sum().item() / 2  # divide by 2 to avoid double counting

        entropy = math.log1p(close_pairs)
        return {
            'kernel_size': int(close_pairs),
            'entropy': entropy
        }

    def summary(self):
        return f"CognitiveObserver: {self.in_dim} → {self.out_dim}"
