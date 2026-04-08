import numpy as np


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    """The complex function we're trying to minimize."""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
