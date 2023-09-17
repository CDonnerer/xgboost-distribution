"""Performance tests on different datasets
"""
import time
from pathlib import Path

import pytest

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from xgboost_distribution import XGBDistribution
from xgboost_distribution.metrics import get_ll_score_func

RANDOM_STATE = 12
DATA_DIR = Path(__file__).parent.parent / "data"


def get_protein_data():
    df = pd.read_csv(
        DATA_DIR / "protein.data",
        usecols=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"],
    )
    y = df.pop("F7")
    X = df
    return X, y


def get_msd_data():
    df = pd.read_csv(DATA_DIR / "msd.data")
    df = df.iloc[:, ::-1]
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def get_concrete_data():
    df = pd.read_excel(DATA_DIR / "concrete.data")
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def get_naval_data():
    df = pd.read_csv(DATA_DIR / "naval.data", header=None, delim_whitespace=True)
    df = df.iloc[:, :-1]
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


@pytest.mark.skip
def test_performance():
    X, y = fetch_california_housing(return_X_y=True)

    normal_ll = get_ll_score_func("normal")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=0.2
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, random_state=RANDOM_STATE
    )

    t0 = time.perf_counter()

    xgbd = XGBDistribution(max_depth=3, n_estimators=500, early_stopping_rounds=10)
    xgbd.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    pred = xgbd.predict(X_test)

    elapsed = time.perf_counter() - t0

    score = normal_ll(y_test, pred)

    print(f"Score: {score}, time {elapsed}")
    # Score: -2.802896707617748, time 6.97 to 7.1026417638 (main)
    # Score: -2.802896707617748, time 5.744270787


if __name__ == "__main__":
    test_performance()
