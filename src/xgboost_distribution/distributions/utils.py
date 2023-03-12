"""Utility functions for distributions
"""
import numpy as np

MIN_EXPONENT = np.log(np.float32(1e-32))
MAX_EXPONENT = np.log(np.finfo("float32").max) - 1


def safe_exp(x):
    """Like np.exp, but clipped to prevent overflow (in float32 world)

    Ensures that
        1. large numbers cannot hit infinity
        2. small numbers cannot hit precisely zero

    NB: The limits are chosen such that we have some stability in subsequent
    computations. E.g the minimum returned value should be safe in a division
    with a numerator of size up to ~1e6.
    """
    # TODO: Empirically refine these limits using different datasets.
    # Can we clip more without losing accuracy?
    x_clipped = np.clip(x, a_min=MIN_EXPONENT, a_max=MAX_EXPONENT)
    return np.exp(x_clipped)


def check_all_integer(x):
    if not all(x == x.astype(int)):
        raise ValueError("All values of target must be integers!")


def check_all_ge_zero(x):
    if not all(x >= 0):
        raise ValueError("All values of target must be >=0!")


def check_all_gt_zero(x):
    if not all(x > 0):
        raise ValueError("All values of target must be > 0!")
