import numpy as np

if np.__version__ >= "2.0.0":
    """Required as the `b` input array shape is treated differently.

    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    """

    def linalg_solve(a, b):
        x = np.linalg.solve(a, b[..., np.newaxis])
        return np.squeeze(x, axis=2)
else:
    linalg_solve = np.linalg.solve
