"""Example of count data modelled with Poisson
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson
from sklearn.model_selection import train_test_split

from xgboost_distribution import XGBDistribution


def generate_count_data(n_samples=10_000):
    X = np.random.uniform(-2, 0, n_samples)
    n = 66 * np.abs(np.cos(X))
    p = 0.5 * np.abs(np.cos(X / 3))

    y = np.random.negative_binomial(n=n, p=p, size=n_samples)
    return X[..., np.newaxis], y


def plot_distribution_heatmap(model, x_range=(-2, 0), y_range=(0, 100)):
    xx = np.linspace(x_range[0], x_range[1], 100)
    yy = np.linspace(y_range[0], y_range[1], y_range[1] - y_range[0] + 1)

    preds = model.predict(xx[..., np.newaxis])

    ym, xm = np.meshgrid(xx, yy)

    z = np.array([poisson.pmf(yy, mu=preds.mu[ii]) for ii, x in enumerate(xx)])
    for val in z:
        val /= val.max()
    z = z.transpose()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.pcolormesh(
        ym, xm, z, cmap="Oranges", vmin=0, vmax=1.608, alpha=1.0, shading="auto"
    )
    plt.show()


def main():
    X, y = generate_count_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = XGBDistribution(
        distribution="poisson",
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
    plot_distribution_heatmap(model)


if __name__ == "__main__":
    main()
