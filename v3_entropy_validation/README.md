# Cognitive Observer Simulation – v3 Entropy Validation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15627738.svg)](https://doi.org/10.5281/zenodo.15627738)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📌 **Note:** A newer version of the related article is available:  
> **Version 7.0 – June 2025**: [https://doi.org/10.5281/zenodo.15713858](https://doi.org/10.5281/zenodo.15713858)  
> Please consider referring to this updated version for the most recent results and clarifications.

This repository provides a modular validation suite for cognitive observer simulation and entropy dynamics.
It includes tools for evaluating projected cognitive structure (PCS), trajectory discriminability (TDS),
t-SNE-based embeddings, and Landauer entropy estimates.

## Directory Structure

```
v3_entropy_validation/
│
├── metrics/
│   ├── validate_pcs_tds_tsne.py         # PCS, TDS and t-SNE validation script
│   ├── landauer_entropy.py              # Landauer entropy estimator
│   ├── test_validate_pcs.py             # Unit test for PCS/TDS/t-SNE metric verification
│   └── verification_plots/              # All publication-ready plots and CSV data
│       ├── landauer_vs_epsilon.csv
│       ├── landauer_vs_epsilon.pdf
│       ├── tds_vs_epsilon.csv
│       ├── tds_vs_epsilon.pdf
│       ├── tsne_epsilon_0.010.pdf
│       ├── tsne_epsilon_0.050.pdf
│       ├── tsne_epsilon_0.100.pdf
│       ├── tsne_epsilon_0.200.pdf
│       └── tsne_epsilon_0.500.pdf
│
├── observer_simulator_decoherence.py    # Simulator of observer latent trajectory
├── experiment_renkel_decoherence.py     # Main reproducible experiment (Renkel-type decoherence)
├── visualization_decoherence.py         # Graph plotting and export module
└── test_visualization.py                # Unit test for visualization output
```

## PCS, TDS and t-SNE Validation

The file `metrics/validate_pcs_tds_tsne.py` performs validation of the Projected Cognitive Space (PCS),
evaluates the Trajectory Discrimination Score (TDS), and visualizes t-SNE embeddings under varying projection precision (ε).

### Usage (as Python module)

```python
from metrics.validate_pcs_tds_tsne import simulate_projected_trajectories, compute_tds

trajectories = simulate_projected_trajectories(epsilon=0.1, n_runs=10, timesteps=100, seed=42)
tds_value = compute_tds(trajectories)
print(f"TDS(ε=0.1): {tds_value:.4f}")
```

### Verification Outputs

- `tds_vs_epsilon.pdf`: TDS vs ε summary plot
- `tsne_epsilon_*.pdf`: t-SNE 2D projections for each ε value
- `tds_vs_epsilon.csv`: Raw data for LaTeX plotting

### Verification Metrics

Automated tests are provided to ensure reproducibility and correctness of the computed metrics.

- `test_validate_pcs.py`: Pytest script to check:
  - Existence and integrity of `tds_vs_epsilon.csv`
  - Validity of TDS values (positive, finite)
  - Monotonic increase of ε values in the dataset

Run the tests via:

```bash
pytest test_validate_pcs.py -v
```

## Landauer Entropy Estimation

The script `metrics/landauer_entropy.py` estimates minimum entropy cost in simulated observer transitions.
It quantifies thermodynamic constraints in the information-processing behavior of the model.

### Example (from Python)

```python
from metrics.landauer_entropy import estimate_entropy

entropy_series = estimate_entropy(n_steps=200, seed=1)
```

Output: Time-series of computed entropy bounds for each trajectory segment.

## Renkel-Type Decoherence Experiment

`experiment_renkel_decoherence.py` reproduces an abstract version of the Renkel decoherence scenario using latent observer dynamics.
This experiment builds on a minimal simulator `observer_simulator_decoherence.py`.

**Note:** This script reproduces a minimal version of the Renkel scenario without saving advanced plots.  
Full visualizations are part of a separate article under preparation.

### Execution

```bash
python experiment_renkel_decoherence.py
```

All figures are saved into the root directory or `verification_plots/` depending on the module invoked.

## Visualization Module

The `visualization_decoherence.py` module provides export-ready plotting utilities (PDF, PNG) for:

- Latent observer trajectories
- Projected cognitive structure
- t-SNE comparison views
- Entropy bounds

Each function is documented and reproducible from any simulation pipeline.

---

## Reproducibility

All components use `numpy.random.default_rng(seed)` for deterministic behavior.
Tests are available via `pytest` for validating visualization and metric stability.

---

## 📄 License

MIT License. See `LICENSE` file for details.

## 📖 Cite this Work

If you use this codebase in your research, please cite:

```bibtex
@software{vladimir_khomyakov_2025_cognitive_observer,
  author = {Vladimir Khomyakov},
  title = {Cognitive Observer Simulation: Entropy Scaling and Extremum Structure},
  year = 2025,
  url = {https://github.com/Khomyakov-Vladimir/cognitive-observer-simulation}
}
```

> 📌 **Note:** This code is associated with Version 3 of the article.  
> For the most recent version (v4.0, June 2025) with updated discussion and extended results, please cite:  
> [https://doi.org/10.5281/zenodo.15661536](https://doi.org/10.5281/zenodo.15661536)
