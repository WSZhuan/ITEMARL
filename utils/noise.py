# utils/noise.py
import numpy as np

def gaussian_noise(shape, scale=1/20.0, rng=None):
    """Return Gaussian noise array of given shape."""
    if rng is None: rng=np.random
    return rng.normal(loc=0.0, scale=scale, size=shape)