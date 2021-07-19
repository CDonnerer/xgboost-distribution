import glob
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from ngboost import NGBRegressor
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from xgboost_distribution import XGBDistribution

_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Datasets for benchmarking
# -----------------------------------------------------------------------------

DATASETS = {}  # each instantiated dataset will be stored here, keyed by name
DATA_DIR = Path(__file__).parent.parent.absolute().joinpath("data")


@dataclass
class Dataset:
    name: str
    url: str
    unpack: str = None
    load_func: callable = pd.read_csv
    processing_func: callable = None

    def __post_init__(self):
        DATASETS.update({self.name: self})


Dataset(
    name="housing",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",  # noqa: E501
    load_func=partial(pd.read_csv, header=None, delim_whitespace=True),
)

Dataset(
    name="wine",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",  # noqa: E501
    load_func=partial(pd.read_csv, delimiter=";"),
)

Dataset(
    name="naval",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",  # noqa: E501
    unpack="data.txt",
    load_func=partial(pd.read_csv, header=None, delim_whitespace=True),
    processing_func=lambda x: x.iloc[:, :-1],
)

Dataset(
    name="energy",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",  # noqa: E501
    load_func=pd.read_excel,
    processing_func=lambda x: x.iloc[:, :-1],
)

Dataset(
    name="yacht",
    url="http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",  # noqa: E501
    load_func=partial(pd.read_csv, header=None, delim_whitespace=True),
)

Dataset(
    name="concrete",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",  # noqa: E501
    load_func=pd.read_excel,
)


def load_dataset(name, data_dir=DATA_DIR):
    dataset = DATASETS[name]

    os.makedirs(data_dir, exist_ok=True)
    local_path = f"{DATA_DIR}/{name}.data"

    if os.path.exists(local_path):
        df = dataset.load_func(local_path)
        if dataset.processing_func:
            df = dataset.processing_func(df)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values
    else:
        _logger.info("Dataset not locally cached, downloading from url...")
        r = requests.get(dataset.url)

        if dataset.unpack is not None:
            with tempfile.TemporaryFile() as fp:
                fp.write(r.content)
                unpack_file_from_zip(fp, dataset.unpack, local_path)
        else:
            with open(local_path, "wb") as f:
                f.write(r.content)

        _logger.info(f"Downloaded file to {local_path}")
        return load_dataset(name)


def unpack_file_from_zip(zip_file, to_unpack, path):
    with zipfile.ZipFile(zip_file, "r") as inzipfile:
        with tempfile.TemporaryDirectory() as temp_dir:
            inzipfile.extractall(path=temp_dir)
            data_file = glob.glob(f"{temp_dir}/**/{to_unpack}", recursive=True)[0]
            shutil.copyfile(data_file, path)


# dataset_name_to_loader = {
#     "kin8nm": lambda: pd.read_csv("data/uci/kin8nm.csv"),
#     "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
#     "protein": lambda: pd.read_csv("data/uci/protein.csv")[
#         ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
#     ],
#     "msd": lambda: pd.read_csv("data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
# }


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def rmse(y_pred, y_test):
    return np.sqrt(mean_squared_error(y_pred, y_test))


def normal_nll(loc, scale, y_test):
    return -norm.logpdf(y_test.flatten(), loc=loc, scale=scale).mean()


@dataclass
class EvalResults:
    name: str
    total_time: float
    rmse: float
    nll: float


def summarize_results(results):
    # records = [(result.rmse, result.nll, result.total_time) for result in results]
    df = pd.DataFrame(results)
    # df = df.set_index("name")
    # records, columns=["rmse", "nll", "total_time"])
    return df.agg(["mean", "std"], axis=0)


evaluations = []


def evaluate(evaluation_func):
    def measured(data):
        t0 = time.perf_counter()
        preds = evaluation_func(data)
        elapsed = time.perf_counter() - t0
        return EvalResults(
            name=evaluation_func.__name__,
            rmse=rmse(preds.loc, data.y_test),
            nll=normal_nll(preds.loc, preds.scale, data.y_test),
            total_time=elapsed,
        )

    evaluations.append(measured)
    # evaluate.update({evaluation_func.__name__: measured})
    return measured


@evaluate
def ngb_regressor(data):
    ngb = NGBRegressor(verbose=False)
    ngb.fit(
        data.X_train,
        data.y_train,
        X_val=data.X_val,
        Y_val=data.y_val,
        early_stopping_rounds=10,
    )
    return ngb.pred_dist(data.X_test, max_iter=ngb.best_val_loss_itr)


@evaluate
def xgb_distribution(data):
    xgbd = XGBDistribution(max_depth=3, natural_gradient=True, n_estimators=500)
    xgbd.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_test, data.y_test)],
        early_stopping_rounds=10,
        verbose=False,
    )
    return xgbd.predict(data.X_test)


# @evaluate
def xgb_regressor(data):
    xgb = XGBRegressor(max_depth=3, n_estimators=500)
    xgb.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_test, data.y_test)],
        early_stopping_rounds=10,
        verbose=False,
    )
    return xgb.predict(data.X_test)


def setup_logging(loglevel=logging.INFO):
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="concrete")
    argparser.add_argument("--random-seed", type=int, default=42)
    argparser.add_argument("--n-folds", type=int, default=10)
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging()

    np.random.seed(args.random_seed)

    X, y = load_dataset(args.dataset)
    _logger.info(f"Loaded dataset: `{args.dataset}`, X.shape={X.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
    _logger.info(f"Cross-validation with {args.n_folds} folds...")

    results = [list() for eval in evaluations]

    for ii, (train_index, test_index) in enumerate(kf.split(X)):
        _logger.info(f"Fold {ii+1} / {args.n_folds}...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=args.random_seed
        )
        split_data = SplitData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        for eval_func, result in zip(evaluations, results):
            result.append(eval_func(split_data))

    for result in results:
        _logger.info(f"{result[0].name}\n{summarize_results(result)}")
