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
from datetime import datetime
from functools import partial, wraps
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sqlalchemy as sa
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

Dataset(
    name="protein",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    processing_func=lambda x: x.loc[
        :, ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ],
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


# -----------------------------------------------------------------------------
# Metrics to evaluate
# -----------------------------------------------------------------------------


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def root_mean_squared_error(y_pred, y_test):
    return np.sqrt(mean_squared_error(y_pred, y_test))


def normal_nll(loc, scale, y_test):
    return -norm.logpdf(y_test.flatten(), loc=loc, scale=scale).mean()


@dataclass
class EvalResult:
    total_time: float
    rmse: float
    nll: float


def summarize_results(results):
    return (
        pd.DataFrame(results)
        .agg(["mean", "std"], axis=0)
        .melt(ignore_index=False, var_name="metric")
        .rename_axis("agg_func")
        .reset_index()
    )


evaluations = []


def evaluate(evaluation_func):
    @wraps(evaluation_func)
    def measured(data):
        t0 = time.perf_counter()
        preds = evaluation_func(data)
        elapsed = time.perf_counter() - t0

        if isinstance(preds, np.ndarray):
            rmse = root_mean_squared_error(preds, data.y_test)
            nll = None
        else:
            rmse = root_mean_squared_error(preds.loc, data.y_test)
            nll = normal_nll(preds.loc, preds.scale, data.y_test)
        return EvalResult(rmse=rmse, nll=nll, total_time=elapsed)

    evaluations.append(measured)
    return measured


# -----------------------------------------------------------------------------
# Model functions, everything decorated by @evaluate will be evaluated
# -----------------------------------------------------------------------------


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
        eval_set=[(data.X_val, data.y_val)],
        early_stopping_rounds=10,
        verbose=False,
    )
    return xgbd.predict(data.X_test)


@evaluate
def xgb_regressor(data):
    xgb = XGBRegressor(max_depth=3, n_estimators=500)
    xgb.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_val, data.y_val)],
        early_stopping_rounds=10,
        verbose=False,
    )
    return xgb.predict(data.X_test)


# -----------------------------------------------------------------------------
# Main method for running cross-validation becnhmark experiment
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    setup_logging()
    db = DataBase()

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

    for eval_func, result in zip(evaluations, results):
        df_summary = summarize_results(result)
        df_summary["dataset"] = args.dataset
        df_summary["model"] = eval_func.__name__

        _logger.info(f"{eval_func.__name__}\n{df_summary}")

        db.insert_metrics(df_summary.to_dict("records"))

    # query = (
    #     sa.select(
    #         [
    #             db.metrics.c.dataset,
    #             db.metrics.c.model,
    #             db.metrics.c.metric,
    #             db.metrics.c.agg_func,
    #             func.coalesce(db.metrics.c.value, None).label("value"),
    #         ]
    #     )
    #     .order_by(db.metrics.c.created_on.desc())
    #     .group_by(
    #         db.metrics.c.dataset,
    #         db.metrics.c.model,
    #         db.metrics.c.metric,
    #         db.metrics.c.agg_func,
    #     )
    # )
    # print(pd.read_sql(query, db.connection))

    df = db.get_metrics_pdf()
    df_pivot = df.pivot_table(
        values=["value"],
        index=["dataset", "agg_func"],
        columns=["model", "metric"],
    )
    # df_pivot.columns = df_pivot.columns.get_level_values(1)
    #
    # def number_with_error(val, err):
    #     dist = -int(np.floor(np.log10(err)))
    #     if dist > 0:
    #         return f"{str(round(val, dist))}({str(round(err, dist))[-1]})"
    #     else:
    #         return f"{str(round(val, 0))}({str(round(err, 0))})"
    #
    # df_pivot["col_3"] = df_pivot.apply(
    #     lambda x: number_with_error(x["mean"], x["std"]), axis=1
    # )
    print(df_pivot)
    # print(df_pivot.transpose())


class DataBase:
    def __init__(self, data_dir=DATA_DIR, db_name="results.db"):
        self.metadata = sa.MetaData()
        self.engine = sa.create_engine(f"sqlite:///{data_dir}/{db_name}")
        self.connection = self.engine.connect()
        self._create_tables()

    def _create_tables(self):
        self.metrics = sa.Table(
            "metrics",
            self.metadata,
            sa.Column("dataset", sa.String(), nullable=False),
            sa.Column("model", sa.String(), nullable=False),
            sa.Column("metric", sa.String(), nullable=False),
            sa.Column("agg_func", sa.String(), nullable=False),
            sa.Column("value", sa.Float(), nullable=True),
            sa.Column("created_on", sa.DateTime(), default=datetime.now),
        )
        self.metadata.create_all(self.engine)

    def insert_metrics(self, records):
        ins = self.metrics.insert()
        self.connection.execute(ins, records)

    def get_metrics_pdf(self):
        return pd.read_sql(
            sa.select(self.metrics).order_by(self.metrics.c.created_on.desc()),
            self.connection,
        )


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
    main()
