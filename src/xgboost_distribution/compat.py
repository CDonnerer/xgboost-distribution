import numpy as np

if np.__version__ >= "2.0.0":

    def linalg_solve(a, b):
        b_padded = b[..., np.newaxis]
        res = np.linalg.solve(a, b_padded)
        return np.squeeze(res, axis=2)
else:

    def linalg_solve(a, b):
        return np.linalg.solve(a, b)
