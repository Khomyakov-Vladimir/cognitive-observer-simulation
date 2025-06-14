import matplotlib.pyplot as plt
import numpy as np

# Simulate data for demonstration of entropy steps and Landauer line
epsilons = np.logspace(-2, 0.5, 200)
entropy = np.log2(1 + np.exp(-4 * (np.log10(epsilons) - 0.2)**2)) * 6
landauer_energy = np.log(2) / epsilons  # E = kT ln 2, with T ~ 1/ε, normalized

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epsilons, entropy, label='Cognitive Entropy $S(\\varepsilon)$', color='blue')
plt.plot(epsilons, landauer_energy, label='Landauer Limit $E = k_B T \\ln 2$', color='green', linestyle='--')
plt.axvline(x=0.1, color='red', linestyle=':', label='Phase Threshold $\\varepsilon^*$')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('$\\varepsilon$ (Cognitive resolution)')
plt.ylabel('Entropy / Energy (a.u.)')
plt.title('Cognitive Entropy vs. Landauer Bound')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cognitive_entropy_vs_landauer.pdf")
plt.show()
