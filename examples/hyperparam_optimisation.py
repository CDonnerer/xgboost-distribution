"""Example for hyperparameter optimsation
"""
import scipy
from sklearn.datasets import load_boston
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from xgboost_distribution import XGBDistribution


def ll_score_func(distribution):
    # TODO: Why isn't this working?
    dists = {"normal": scipy.stats.norm}

    def score_func(y, y_pred):
        return dists[distribution](y, *y_pred).mean()

    return score_func


def normal_ll_score(y, y_pred):
    return scipy.stats.norm.logpdf(y, *y_pred).mean()


def main():
    data = load_boston()

    X, y = data.data, data.target

    param_grid = {"n_estimators": [1, 5, 10, 20], "max_depth": [1, 2, 3]}

    # normal_ll_score = ll_score_func("normal")
    # breakpoint()

    xgb_cv = GridSearchCV(
        XGBDistribution(),
        param_grid,
        cv=5,
        scoring=make_scorer(normal_ll_score),
    )

    xgb_cv.fit(X, y)
    print(f"Best CV score: {xgb_cv.best_score_}")
    print(f"Best params: {xgb_cv.best_params_}")


if __name__ == "__main__":
    main()
