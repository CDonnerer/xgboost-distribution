from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    dist_name = __name__  # assuming project and package name are equal
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from xgboost_distribution.model import XGBDistribution  # noqa
