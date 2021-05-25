import numpy as np
from sklearn.model_selection import train_test_split

from xgb_dist.model import XGBDistribution


def test_XGBDistribution(small_X_y_data):
    """Dummy test that everything runs"""
    X, y = small_X_y_data

    X_train, X_test, y_train, y_test = train_test_split(
        X[..., np.newaxis], y, test_size=0.2
    )
    model = XGBDistribution(distribution="normal", max_depth=3, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2)

    assert model is not None
