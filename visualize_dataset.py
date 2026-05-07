import numpy as np
import matplotlib.pyplot as plt

from dataset_generation import generate_dataset
from add_noise import add_noise

# 1. Parameters setup
N_SAMPLES = 100
N_FEATURES = 20
SEED_DATA = 1   # X and w* generation seed
SEED_NOISE = 1 # noise generation seed
SNR_TARGET = 2  # SNR used in the paper

# 2. Clean dataset (Noiseless)
X, y_clean, true_w = generate_dataset(N_SAMPLES, N_FEATURES, distribution='normal', seed=SEED_DATA)

# 3. Gaussian noise addition
# y_noisy = f*(x) + epsilon
y_noisy = add_noise(y_clean, snr=SNR_TARGET, seed=SEED_NOISE)

# 4. Computing actual noise
epsilon = y_noisy - y_clean

# 5. Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Comparison between Signal (Bayes Predictor) e noisy observation
plt.subplot(1, 2, 1)
plt.plot(y_clean[:100], 'g-', label='True Signal $f^*(x)$ (Bayes)', alpha=0.7)
plt.scatter(range(100), y_noisy[:100], color='red', marker='x', label=f'Noisy Labels (SNR={SNR_TARGET})')
plt.title(f"Dataset: n={N_SAMPLES}, d={N_FEATURES} (Seed: {SEED_DATA})")
plt.xlabel("Sample index")
plt.ylabel("Label value")
plt.legend()
plt.grid(True)

# Plot 2: Histogram of added noise
plt.subplot(1, 2, 2)
plt.hist(epsilon, bins=15, color='orange', edgecolor='black', alpha=0.8)
plt.title(f"Noise distribution $\\epsilon$ (SNR={SNR_TARGET})")
plt.xlabel("Error amplitude")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Printing the reproducibility parameters
print(f"--- Reproducibility parameters ---")
print(f"Seed Dataset: {SEED_DATA} | Seed Noise: {SEED_NOISE}")
print(f"Signal variance: {np.var(y_clean):.4f}")
print(f"Noise variance: {np.var(epsilon):.4f}")
print(f"Actual SNR: {np.var(y_clean)/np.var(epsilon):.2f}")