"""Minimal example of XGBDistribution on Boston Housing dataset
"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from xgb_dist import XGBDistribution


def plot_residuals(y_true, y_pred, y_err):
    fig, ax = plt.subplots()
    ax.errorbar(
        y_true,
        y_true - y_pred,
        yerr=y_err,
        marker="o",
        linestyle="None",
        c="k",
        markersize=2.5,
        linewidth=0.5,
    )
    ax.axhline(0, c="k", linestyle="--")
    ax.set_xlabel("y_test")
    ax.set_ylabel("y_test - y_pred")
    plt.show()


def main():
    data = load_boston()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = XGBDistribution(
        distribution="normal",
        natural_gradient=True,
        max_depth=2,
        n_estimators=500,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False,
    )
    preds = model.predict(X_test)

    plot_residuals(y_true=y_test, y_pred=preds.mean, y_err=preds.scale)


if __name__ == "__main__":
    main()
