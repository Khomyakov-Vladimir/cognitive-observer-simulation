
# Cognitive Entropy vs Landauer Bound: Annotated Analysis (v2)

This folder contains the reproducible analysis and visualization of the comparison between cognitive entropy and the Landauer thermodynamic bound, as described in the **v2 version** of the article:

> **Cognitive Simulator and Subjective Physics: Entropy as a Cognitive Projection**  
> Vladimir Khomyakov, 2025  
> https://doi.org/10.5281/zenodo.15544618

---

## 📌 Purpose

This analysis quantifies the relation between cognitive entropy \( S \), computed in simulations of a categorical observer model, and the physical minimum energy cost per bit defined by the **Landauer principle**.

We identify:
- The **cognitive discrimination threshold** \( T^* \), determined by the peak of \( \frac{dS}{dT} \);
- The **intersection point** where normalized entropy matches the normalized Landauer energy \( k_B T \ln 2 \);
- The appearance of **entropy plateaus** (discrimination steps) suggesting quantized cognitive phases.

---

## 📁 Files

| File | Description |
|------|-------------|
| `entropy_landauer_analysis.z01`              | Part 1 of split archive (15 MB) |
| `entropy_landauer_analysis.zip`              | Part 2 of split archive (11.8 MB) — open this file to extract |
| `entropy_landauer_data.zip`                  | Zipped NumPy `.npz` file used for plotting (original: 30.5MB) |
| `entropy_kernel_vs_epsilon.zip`              | Zipped raw CSV data used for entropy simulation |
| `plot_discrimination_vs_landauer_v3_annotated.pdf` | Final annotated figure included in the publication |
| `plot_discrimination_vs_landauer_v3_annotated.py`  | Annotated plot generator |
| `analyze_entropy_landauer.py`                | Core analysis script (entropy vs. Landauer) |
| `gradient.py`                                | Optional script to visualize higher-order derivatives of entropy |
| `runner.py`                                  | Main controller script (automated execution + logging) |
| `log.txt`                                    | Full execution log for reproducibility |
| `README.md`                                  | This file |

> 💡 Note: To extract `entropy_landauer_analysis.zip`, ensure that `entropy_landauer_analysis.z01` is in the same directory. Use any archive manager that supports multi-volume ZIP (e.g., 7-Zip, WinRAR).

---

## ⚙️ How to Run

Make sure your environment has the required Python packages:
```bash
pip install numpy pandas matplotlib
```

To run the full analysis:
```bash
python runner.py
```

This will:
- Clean the `results/` folder;
- Run the entropy–Landauer analysis;
- Generate and save the annotated figure;
- Log all steps in `log.txt`.

---

## 📈 Figure Description

The figure `plot_discrimination_vs_landauer_v3_annotated.pdf` shows:

- **Blue curve**: Normalized cognitive entropy \( S/S_{\max} \);
- **Green curve (right axis)**: Landauer energy in attojoules;
- **Dashed line**: Cognitive discrimination threshold \( T^* \approx 915\,\mathrm{K} \);
- **Dotted line**: Intersection with Landauer bound at \( T \approx 474\,\mathrm{K} \);
- **Step-like descent**: Emergent cognitive phase structure.

---

## 📚 Background

This repository is part of a broader investigation of Subjective Physics and cognitive foundations of quantum observation. It builds on:

- **Rolf Landauer** (1961): Thermodynamic irreversibility of information processing;
- **Cognitive observer model**: Entropy as a categorical projection;
- **Distinction phases**: Interpreted as cognitive transitions in representational scale.

Recommended reading:

- Kaminsky, A. (2025). *Subjective Foundations of Quantum Mechanics*. Zenodo.  
- Landauer, R. (1961). *Irreversibility and Heat Generation in the Computing Process*. IBM JRD.

---

## 🧠 Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{vladimir_khomyakov_2025_cognitive_observer,
  author       = {Vladimir Khomyakov},
  title        = {Cognitive Observer Simulation: Entropy Scaling and Extremum Structure},
  year         = {2025},
  howpublished = {\url{https://github.com/Khomyakov-Vladimir/cognitive-observer-simulation}}
}
```

---

© 2025 Vladimir Khomyakov. MIT License.
