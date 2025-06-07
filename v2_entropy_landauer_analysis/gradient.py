# gradient.py 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("entropy_kernel_vs_epsilon.csv")

grouped = df.groupby("epsilon").agg({"entropy": "mean"}).reset_index()
eps = grouped["epsilon"].values
S = grouped["entropy"].values
log_eps = np.log(eps)

d1 = np.gradient(S, log_eps)
d2 = np.gradient(d1, log_eps)
d3 = np.gradient(d2, log_eps)
d4 = np.gradient(d3, log_eps)
d5 = np.gradient(d4, log_eps)

plt.figure(figsize=(12, 8))
plt.plot(eps, d1, label="1st", linewidth=2)
plt.plot(eps, d2, label="2nd", linewidth=2)
plt.plot(eps, d3, label="3rd", linewidth=2)
plt.plot(eps, d4, label="4th", linewidth=2)
plt.plot(eps, d5, label="5th", linewidth=2)
plt.xscale("log")
plt.xlabel("ε")
plt.ylabel("Derivative of S(ε)")
plt.title("Higher-Order Derivatives of Cognitive Entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
