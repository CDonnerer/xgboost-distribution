from functools import singledispatch

import numpy as np


@singledispatch
def to_serializable(val):
    return str(val)


@to_serializable.register
def _(val: np.float32):
    return np.float64(val)
