import numpy as np
def generate_dataset(n, d, distribution='normal', seed=1, w_star=None, decay=False):
    
    #random number generator based on seed
    rng = np.random.default_rng(seed)

    # Generating X
    if distribution == 'normal':
        # Mean 0, Variance 1
        X = rng.standard_normal(size=(n, d))
    else:
        # Compact cube [6]^d (used for approximation theorem)
        X = rng.uniform(0, 1, size=(n, d))
    
    
    # Generating w_star
    if w_star is None:
        if decay==True:

            w_star = np.zeros((d, 1))

            k = 20  # desired sweet spot

            # ===== FIRST 20 FEATURE IMPORTANT =====
            head = np.ones(k)

            # ===== TAIL: other features matter, but less =====
            tail_size = d - k
            decay_rate = 0.05  # smaller = slower descent

            #tail = e^-(n/20) with n that goes from 1 to tail_size+1
            tail = np.exp(-decay_rate * np.arange(1, tail_size + 1))

            weights = np.concatenate([head, tail])

            # normalization
            weights = weights / np.linalg.norm(weights)

            w_star[:, 0] = weights
        else:
            w_star = rng.standard_normal(size=(d, 1)) 

    y_clean = X @ w_star


    return X, y_clean, w_star