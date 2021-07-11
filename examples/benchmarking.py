import time
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.learners import default_linear_learner, default_tree_learner
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from xgboost_distribution import XGBDistribution

np.random.seed(1)

dataset_name_to_loader = {
    "housing": lambda: pd.read_csv(
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "housing/housing.data"
        ),
        header=None,
        delim_whitespace=True,
    ),
    "concrete": lambda: pd.read_excel(
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "concrete/compressive/Concrete_Data.xls"
        )
    ),
    "wine": lambda: pd.read_csv(
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "wine-quality/winequality-red.csv"
        ),
        delimiter=";",
    ),
    "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
    "naval": lambda: pd.read_csv(
        "data/uci/naval-propulsion.txt", delim_whitespace=True, header=None
    ).iloc[:, :-1],
    "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
    "energy": lambda: pd.read_excel(
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00242/ENB2012_data.xlsx"
        )
    ).iloc[:, :-1],
    "protein": lambda: pd.read_csv("data/uci/protein.csv")[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ],
    "yacht": lambda: pd.read_csv(
        (
            "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/"
            "yacht_hydrodynamics.data"
        ),
        header=None,
        delim_whitespace=True,
    ),
    "msd": lambda: pd.read_csv("data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def rmse(preds, y_test):
    return np.sqrt(mean_squared_error(preds.loc, y_test))


def nll(preds, y_test):
    return -norm.logpdf(y_test.flatten(), loc=preds.loc, scale=preds.scale).mean()


@dataclass
class EvalResults:
    total_time: float
    rmse: float
    nll: float


def summarize_results(results):
    records = [(result.rmse, result.nll, result.total_time) for result in results]
    df = pd.DataFrame.from_records(records, columns=["rmse", "nll", "total_time"])
    return df.agg(["mean", "std"], axis=0)


def fit_predict_ngb(data):

    start_time = time.time()

    ngb = NGBRegressor(verbose=False)
    # Base=base_name_to_learner[args.base],
    # Dist=eval(args.distn),
    # Score=eval(args.score),
    # n_estimators=args.n_est,
    # learning_rate=args.lr,
    # natural_gradient=args.natural,
    # minibatch_frac=args.minibatch_frac,
    # verbose=args.verbose,

    ngb.fit(
        data.X_train,
        data.y_train,
        X_val=data.X_val,
        Y_val=data.y_val,
        early_stopping_rounds=10,
    )
    preds = ngb.pred_dist(data.X_test)

    elapsed_time = time.time() - start_time
    return EvalResults(
        rmse=rmse(preds, data.y_test),
        nll=nll(preds, data.y_test),
        total_time=elapsed_time,
    )


def fit_predict_xgbd(data):
    start_time = time.time()

    xgbd = XGBDistribution(n_estimators=500)
    xgbd.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_test, data.y_test)],
        early_stopping_rounds=10,
        verbose=False,
    )
    preds = xgbd.predict(data.X_test)

    elapsed_time = time.time() - start_time
    return EvalResults(
        rmse=rmse(preds, data.y_test),
        nll=nll(preds, data.y_test),
        total_time=elapsed_time,
    )


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="concrete")
    argparser.add_argument("--reps", type=int, default=5)
    argparser.add_argument("--n-est", type=int, default=2000)
    argparser.add_argument("--n-splits", type=int, default=10)
    argparser.add_argument("--distn", type=str, default="Normal")
    argparser.add_argument("--lr", type=float, default=0.01)
    argparser.add_argument("--natural", action="store_true")
    argparser.add_argument("--score", type=str, default="MLE")
    argparser.add_argument("--base", type=str, default="tree")
    argparser.add_argument("--minibatch-frac", type=float, default=None)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    # load dataset -- use last column as label
    data = dataset_name_to_loader[args.dataset]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    if not args.minibatch_frac:
        args.minibatch_frac = 1.0

    print(f"Dataset={args.dataset} X.shape={X.shape}")

    if args.dataset == "msd":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
        args.minibatch_frac = 0.1
    else:
        kf = KFold(n_splits=args.n_splits)
        folds = kf.split(X)

        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(args.n_splits):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))

    ngb_res, xgbd_res = list(), list()

    for itr, (train_index, test_index) in enumerate(folds):
        print(f"Fold {itr+1}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2
        )

        split_data = SplitData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        ngb_res.append(fit_predict_ngb(split_data))
        xgbd_res.append(fit_predict_xgbd(split_data))

    print(summarize_results(ngb_res))
    print(summarize_results(xgbd_res))
