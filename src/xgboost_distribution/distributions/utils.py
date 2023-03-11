"""Utility functions for distributions
"""
import numpy as np

MIN_EXPONENT = np.log(np.finfo("float32").tiny) + 1
MAX_EXPONENT = np.log(np.finfo("float32").max) - 1


def check_all_integer(x):
    if not all(x == x.astype(int)):
        raise ValueError("All values of target must be integers!")


def check_all_ge_zero(x):
    if not all(x >= 0):
        raise ValueError("All values of target must be >=0!")


def check_all_gt_zero(x):
    if not all(x > 0):
        raise ValueError("All values of target must be > 0!")


def safe_exp(x):
    """Like np.exp, but protects against overflow"""
    x_clipped = np.clip(x, a_min=MIN_EXPONENT, a_max=MAX_EXPONENT)
    return np.exp(x_clipped)
