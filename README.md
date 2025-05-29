# Cognitive Observer Simulation: Entropy Scaling and Extremum Structure

This repository contains the code and tools used in the simulation and analysis of cognitive representations under scale-dependent entropy, as described in the [Subjective Physics](https://doi.org/10.5281/zenodo.15544618) framework.

## 🧠 Project Overview

This project simulates the behavior of a cognitive observer when interacting with a high-dimensional ontological space \(\mathcal{H}_{\text{ont}} \subseteq \mathbb{R}^N\), projected into a cognitive manifold \(\mathcal{C}_{\text{obs}} \subseteq \mathbb{R}^d\) via a lossy functor \(F\). The central object of study is the entropy function \(S(\epsilon)\), evaluated at various scales \(\epsilon\), and its derivative \(dS/d\log(\epsilon)\), which reveals cognitive phase transitions and distinct perceptual regimes.

## 🔬 Research Goals

- Analyze scale-dependent entropy \(S(\epsilon)\) of cognitive representations.
- Detect local extrema of \(dS/d\log(\epsilon)\) as indicators of representational phase transitions.
- Visualize entropy landscapes and their associated cognitive geometries using t-SNE embeddings.
- Construct extremum graphs to explore transitions between distinct observer states.

## 📂 Repository Structure

├── cognitive_analysis.py # Entropy plotting, extremum detection, t-SNE, graph construction  
├── run_experiment.py # Main entry point for data generation and analysis pipeline  
├── plot_utils.py # General-purpose plotting utilities  
├── config.py # Global configuration parameters  
├── results/ # Output directory (entropy curves, JSON, plots)  
├── README.md # This file  
└── requirements.txt # Python dependencies

## 🚀 Quick Start

### 1. Clone the repository

git clone https://github.com/Khomyakov-Vladimir/cognitive-observer-simulation.git  
cd cognitive-observer-simulation

2. Set up a virtual environment

python -m venv venv  
source venv/bin/activate  # or venv\Scripts\activate on Windows  
pip install -r requirements.txt

3. Run a full experiment

python run_experiment.py  
Output will be saved to the results/ directory, including:

entropy_curve.png/.pdf: Plot of entropy \(S(\epsilon)\) and its derivative.

entropy_maxima.json: List of extremal points with metadata.

projection_heatmap.png: t-SNE view of entropy extrema.

extremum_graph.png: Graph connecting extrema in projection space.

📊 Reproducibility Checklist

Requirement	Status  
Random seed control	✅  
Deterministic t-SNE embeddings	⚠️*  
Save/load experiment configurations	✅  
Export of key outputs (JSON, PNG)	✅  
No reliance on external APIs	✅  

⚠️ *t-SNE is stochastic by default. For consistent results, set a global seed in TSNE(random_state=42) or use PCA instead.

🔧 Configuration Parameters

All hyperparameters can be adjusted in config.py:

NUM_EPSILON_STEPS = 20  
EPSILON_MIN = 0.01  
EPSILON_MAX = 1.0  
SHOW_PLOTS = True  
RESULTS_DIR = "./results/"

🧠 Theoretical Context

This project is part of an ongoing investigation into subjective physics, where physical regularities are interpreted as cognitive invariants. Entropy and its scale derivative serve as proxies for an observer's internal differentiation of stimuli under limited resolution.

See related theoretical work:

Kaminsky, A. (2025). Subjective foundations of quantum mechanics. Zenodo. https://doi.org/10.5281/zenodo.15098840

Vanchurin, V. (2025). Neural Relativity. ResearchGate. DOI:10.13140/RG.2.2.36422.79689

📎 Sample Output

Entropy \(S(\epsilon)\) and its derivative showing critical points.

Graph of entropy extrema in projection space.

📄 License

MIT License. See LICENSE file for details.

📫 Contact

For questions, contributions, or collaboration proposals, please contact:

Vladimir Khomyakov – khomyakov.vladimir.ru@gmail.com

GitHub – @Khomyakov-Vladimir

## 📖 Cite this Work

If you use this codebase in your research, please cite:

@software{vladimir_khomyakov_2025_cognitive_observer,  
author = {Vladimir Khomyakov},  
title = {Cognitive Observer Simulation: Entropy Scaling and Extremum Structure},  
year = 2025,  
url = {https://github.com/Khomyakov-Vladimir/cognitive-observer-simulation}  
}
