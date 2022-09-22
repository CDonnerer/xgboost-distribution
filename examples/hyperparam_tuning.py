"""Hyperparameter tuning example
"""
from sklearn.datasets import load_boston
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from xgboost_distribution import XGBDistribution
from xgboost_distribution.metrics import (
    get_ll_score_func,
    wrap_point_estimate_score_func,
)


def main():
    data = load_boston()
    X, y = data.data, data.target

    distribution = "normal"

    param_grid = {"n_estimators": [5, 10, 20], "max_depth": [1, 2, 3]}

    xgb_cv = GridSearchCV(
        XGBDistribution(distribution=distribution),
        param_grid,
        cv=5,
        scoring={
            f"{distribution}_ll": make_scorer(get_ll_score_func(distribution)),
            "mae": make_scorer(
                wrap_point_estimate_score_func(mean_absolute_error, distribution)
            ),
        },
        refit=False,
    )

    xgb_cv.fit(X, y)
    print(f"CV results: {xgb_cv.cv_results_}")


if __name__ == "__main__":
    main()
