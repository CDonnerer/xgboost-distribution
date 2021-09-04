"""Example of count data sampled from negative-binomial distribution
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

from xgboost_distribution import XGBDistribution


def generate_count_data(n_samples=10_000):
    X = np.random.uniform(-2, 0, n_samples)
    n = 66 * np.abs(np.cos(X))
    p = 0.5 * np.abs(np.cos(X / 3))

    y = np.random.negative_binomial(n=n, p=p, size=n_samples)
    return X[..., np.newaxis], y


def predict_distribution(model, X, y):
    """Predict a distribution for a given X, and evaluate over y"""

    distribution_func = {
        "normal": getattr(stats, "norm").pdf,
        "laplace": getattr(stats, "laplace").pdf,
        "poisson": getattr(stats, "poisson").pmf,
        "negative-binomial": getattr(stats, "nbinom").pmf,
    }
    preds = model.predict(X[..., np.newaxis])

    dists = np.zeros(shape=(len(X), len(y)))
    for ii, x in enumerate(X):
        params = {field: param[ii] for (field, param) in zip(preds._fields, preds)}
        dists[ii] = distribution_func[model.distribution](y, **params)

    return dists


def create_distribution_heatmap(
    model, x_range=(-2, 0), x_steps=100, y_range=(0, 100), normalize=True
):
    xx = np.linspace(x_range[0], x_range[1], x_steps)
    yy = np.linspace(y_range[0], y_range[1], y_range[1] - y_range[0] + 1)
    ym, xm = np.meshgrid(xx, yy)

    z = predict_distribution(model, xx, yy)

    if normalize:
        z = z / z.max(axis=0)

    return ym, xm, z.transpose()


def main():
    random_state = 10
    np.random.seed(random_state)

    X, y = generate_count_data(n_samples=10_000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = XGBDistribution(
        distribution="negative-binomial",  # try changing the distribution here
        natural_gradient=True,
        max_depth=3,
        n_estimators=500,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False,
    )

    xm, ym, z = create_distribution_heatmap(model)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.pcolormesh(
        xm, ym, z, cmap="Oranges", vmin=0, vmax=1.608, alpha=1.0, shading="auto"
    )
    ax.scatter(X_test, y_test, s=0.75, alpha=0.25, c="k", label="data")
    plt.show()


if __name__ == "__main__":
    main()
