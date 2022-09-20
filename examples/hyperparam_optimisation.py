"""Example for hyperparameter optimsation
"""
from scipy.stats import norm
from sklearn.datasets import load_boston
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from xgboost_distribution import XGBDistribution


def normal_distribution_log_loss_score(y, y_pred):
    loc, scale = y_pred.loc, y_pred.scale
    return norm.logpdf(y, loc=loc, scale=scale).mean()


def main():
    data = load_boston()

    X, y = data.data, data.target

    param_grid = {"n_estimators": [10, 20, 50, 100], "max_depth": [1, 3, 6]}

    xgb_cv = GridSearchCV(
        XGBDistribution(),
        param_grid,
        cv=3,
        scoring=make_scorer(normal_distribution_log_loss_score),
    )

    xgb_cv.fit(X, y)
    print(f"Best CV score: {xgb_cv.best_score_}")
    print(f"Best params: {xgb_cv.best_params_}")


if __name__ == "__main__":
    main()
