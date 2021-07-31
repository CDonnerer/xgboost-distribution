"""Utility functions for distributions
"""


def check_is_integer(x):
    if not all(x == x.astype(int)):
        raise ValueError("All values of target must be integers!")


def check_is_ge_zero(x):
    if not all(x >= 0):
        raise ValueError("All values of target must be >=0!")


def check_is_gt_zero(x):
    if not all(x > 0):
        raise ValueError("All values of target must be > 0!")
