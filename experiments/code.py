import numpy as np

def relationship_generation(num_samples, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, num_samples)
    y = 2 * x + rng.normal(0, 0.1, num_samples)
    return x, y