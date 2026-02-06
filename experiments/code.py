
import numpy as np

def relationship_generation(num_samples, seed):
    rng = np.random.default_rng(seed)
    
    # Generate x values from uniform distribution
    x = rng.uniform(0, 1, num_samples)
    
    # Generate y values from linear function of x with noise
    slope = 2
    noise_std = 1
    y = slope * x + rng.normal(scale=noise_std, size=num_samples)
    
    return x, y