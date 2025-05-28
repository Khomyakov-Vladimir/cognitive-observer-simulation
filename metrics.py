import torch
from config import NUM_EPSILON_STEPS, EPSILON_MIN, EPSILON_MAX, SHOW_PLOTS, RESULTS_DIR


def compute_projection_scores(X: torch.Tensor, Y: torch.Tensor, epsilon: float = 1e-2) -> dict:
    """
    Distortion metrics for cognitive projection:
    
    PCS — Perceptual Collapsing Score (loss of discriminability);
    LCS — Local Coherence Score (preservation of local neighborhoods);
    TDS — Total Distance Distortion (average absolute distortion).
    
    Parameters:
        X (torch.Tensor): Original high-dimensional representations (H_ont).
        Y (torch.Tensor): Projected representations (C_obs).
        epsilon (float): Perceptual resolution threshold.
    
    Returns:
        dict: Dictionary containing PCS, LCS, and TDS metrics.
    """

    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    N = X.shape[0]

    D_X = torch.cdist(X, X, p=2)
    D_Y = torch.cdist(Y, Y, p=2)

    # Consider only upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones_like(D_X), diagonal=1).bool()

    D_X_flat = D_X[mask]
    D_Y_flat = D_Y[mask]

    collapse = (D_X_flat > epsilon) & (D_Y_flat < epsilon)
    local = (D_X_flat < epsilon) & (D_Y_flat < epsilon)
    distortion = torch.abs(D_X_flat - D_Y_flat)

    total_pairs = D_X_flat.numel()
    pcs = collapse.sum().item() / total_pairs
    lcs = local.sum().item() / total_pairs
    tds = distortion.mean().item()

    return {
        "PCS": round(pcs, 4),
        "LCS": round(lcs, 4),
        "TDS": round(tds, 4)
    }
