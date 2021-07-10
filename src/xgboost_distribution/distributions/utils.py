"""Utility functions for distributions
"""


def check_is_integer(x):
    if not all(x == x.astype(int)):
        raise ValueError("All values of target must be integers!")


def check_is_positive(x):
    if not all(x >= 0):
        raise ValueError("All values of target must be positive!")


def check_is_gt_zero(x):
    if not all(x > 0):
        raise ValueError("All values of target must be > 0!")
