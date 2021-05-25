"""Dist-xgboost estimator
"""
import numpy as np
import xgboost as xgb

from dist_xgboost.distributions import get_distributions

from xgb_dist import XGBDistribution

from xgb_distribution import XGBDistribution

from xgbdistribution import XGBDistribution


available_distributions = get_distributions()


class XGBDistribution(xgb.XGBModel):
    def __init__(self, distribution="normal", **kwargs):
        self.distribution = available_distributions[distribution]()
        self._booster = None
        self._xgb_params = {}
        self._xgb_params.update(kwargs)
        self._xgb_params["num_class"] = self.distribution.n_params

    def fit(self, X, y, *, eval_set=None, early_stopping_rounds=None, verbose=True):
        self._xgb_params["disable_default_eval_metric"] = True
        evals = None

        m_train = xgb.DMatrix(X, y)
        evals = [(m_train, "train")]

        if eval_set is not None:
            for ii, eval in enumerate(eval_set):
                evals.append((xgb.DMatrix(eval[0], eval[1]), str(ii)))

        self._booster = xgb.train(
            self._xgb_params,
            m_train,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            obj=self._objective_func(),
            feval=self._evaluation_func(),
            verbose_eval=verbose,
        )
        return self

    def _objective_func(self):
        def obj(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            grad, hess = self.distribution.gradient_and_hessian(y, params)

            grad = grad.reshape((len(y) * self.distribution.n_params, 1))
            hess = hess.reshape((len(y) * self.distribution.n_params, 1))
            return grad, hess

        return obj

    def _evaluation_func(self):
        def feval(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            return self.distribution.loss(y, params)

        return feval

    def predict_dist(self, X):
        params = self._booster.predict(xgb.DMatrix(X), output_margin=True)
        return self.distribution.predict(params)
