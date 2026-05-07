import numpy as np

def add_noise(y_clean, snr, seed=1):
    """
    Adds gaussian noise based on target SNR.
    """

    if snr == float('inf'):
        return y_clean #noiseless case
    
    rng = np.random.default_rng(seed)
    
    # 1. Computing signal power (y_clean variance)
    # E[h*(x)^2] 
    signal_power = np.mean(y_clean**2)
    
    # 2. Computing noise variance (sigma^2) according to given snr
    # Formula: sigma^2 = Signal_Power / SNR
    noise_variance = signal_power / snr
    sigma = np.sqrt(noise_variance)
    
    # 3. Generating noise epsilon ~ N(0, sigma^2)
    noise = rng.normal(0, sigma, size=y_clean.shape)
    
    # 4. Adding noise
    y_noisy = y_clean + noise
    
    return y_noisy
