"""Hyperparameter tuning example
"""
import scipy
from sklearn.datasets import load_boston
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from xgboost_distribution import XGBDistribution


def ll_score_func(distribution):
    dists = {
        "exponential": scipy.stats.expon.logpdf,
        "laplace": scipy.stats.laplace.logpdf,
        "log-normal": scipy.stats.lognorm.logpdf,
        "negative-binomial": scipy.stats.nbinom.logpmf,
        "normal": scipy.stats.norm.logpdf,
        "poisson": scipy.stats.poisson.logpmf,
    }

    def score_func(y, y_pred):
        return dists[distribution](y, *y_pred).mean()

    return score_func


def wrap_sklearn_metric(score_func, distribution):
    dists_mean = {
        "normal": lambda x: x.loc,
        # TODO: What about the others?
    }

    def new_score_func(y, y_pred, **kwargs):
        y_mean = dists_mean[distribution](y_pred)
        return score_func(y, y_mean, **kwargs)

    return new_score_func


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
            f"{distribution}_ll": make_scorer(ll_score_func(distribution)),
            "mae": make_scorer(wrap_sklearn_metric(mean_absolute_error, distribution)),
        },
        refit=False,
    )

    xgb_cv.fit(X, y)
    print(f"CV results: {xgb_cv.cv_results_}")


if __name__ == "__main__":
    main()
