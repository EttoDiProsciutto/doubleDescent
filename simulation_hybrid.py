import numpy as np
import matplotlib.pyplot as plt
import dataset_generation  
import add_noise           
import training

# --- 1. SETUP & CONFIGURATION ---
n_samples = 100              
test_samples = 500           
d_max = 500                  # maximum number of parameters (d)
num_trials = 20              # average over 20 trials for stability
snr = 2                      # Signal-to-Noise Ratio
feature_decay=True

# --- 2. SIMULATION LOOP ---
d_range = np.arange(1, d_max + 1, 2)
results = {'d': [], 'ls_train': [], 'ls_test': []}

# Initialize counters for trials average
total_train_errs = np.zeros(len(d_range))
total_test_errs = np.zeros(len(d_range))

# Sets generation
X_test_full, y_clean_test, w_star = dataset_generation.generate_dataset(test_samples, d_max, decay=feature_decay)
print(f"X_test_full shape:{X_test_full.shape}, y_clean_test: {y_clean_test.shape}, w_star: {w_star.shape}")

for t in range(num_trials):
    
    # X_train has d_max feature, y_clean_train is generated with w_star and from same distribution N(0,1)
    X_train_full, y_clean_train, _ = dataset_generation.generate_dataset(n_samples, d_max, w_star=w_star)
    
    # Adding noise only to training set using dedicated module
    # The noise introduces variance, exploding at d=n
    random_noise_seed=t
    y_train_noisy = add_noise.add_noise(y_clean_train, snr, random_noise_seed)
    
    for idx, d in enumerate(d_range):
        # Slicing: using only first 'd' columns
        X_train_d = X_train_full[:, :d]
        X_test_d = X_test_full[:, :d]
        if d<n_samples:
            w_hat_d=training.train_least_squares(X_train_d,y_train_noisy)
        else:
            alpha=1e-06
            w_hat_d =training.train_ridge_regression(X_train_d,y_train_noisy,alpha)
        # adding the errors
        total_train_errs[idx] += training.compute_risk(X_train_d, y_train_noisy, w_hat_d)
        total_test_errs[idx] += training.compute_risk(X_test_d, y_clean_test, w_hat_d)

# Final averaging over the trials
results['d'] = d_range
results['ls_train'] = total_train_errs / num_trials
results['ls_test'] = total_test_errs / num_trials


# --- 3. PLOTTING ---
plt.figure(figsize=(12, 7))

# Training Error
plt.plot(results['d'], results['ls_train'], 
         label='Empirical Risk (Train Error)', color='royalblue', linestyle='--', alpha=0.7)

# Test Error
plt.plot(results['d'], results['ls_test'], 
         label='Statistical Risk (Test Error)', color='crimson', linewidth=2.5)

# Interpolation Threshold
plt.axvline(x=n_samples, color='black', linestyle=':', 
            label=f'Interpolation Threshold (d=n={n_samples})')

plt.ylim(1e-3, 1e2)
plt.yscale('log')
plt.xlabel('Model Complexity (Number of Features d)', fontsize=12)
plt.ylabel('Risk (Squared Loss)', fontsize=12)
plt.title(f'Double Descent in OLS and RR (Trials={num_trials}, SNR={snr})', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle='-', alpha=0.2)

plt.show()