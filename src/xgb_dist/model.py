"""Dist-xgboost estimator
"""
import numpy as np
import xgboost as xgb
from xgboost.sklearn import _wrap_evaluation_matrices, xgboost_model_doc

from xgb_dist.distributions import get_distributions, get_distribution_doc

available_distributions = get_distributions()


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost distribution.",
    ["model"],
    extra_parameters=get_distribution_doc(),
)
class XGBDistribution(xgb.XGBModel):
    def __init__(self, distribution="normal", **kwargs):
        self.distribution = available_distributions[distribution]()
        super().__init__(objective=None, **kwargs)

    def fit(self, X, y, *, eval_set=None, early_stopping_rounds=None, verbose=True):
        evals = None

        params = self.get_xgb_params()
        params["disable_default_eval_metric"] = True
        params["num_class"] = len(self.distribution.params)

        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=None,
            base_margin=None,
            feature_weights=None,
            eval_set=eval_set,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            eval_group=None,
            eval_qid=None,
            create_dmatrix=lambda **kwargs: xgb.DMatrix(nthread=self.n_jobs, **kwargs),
            label_transform=lambda x: x,
        )

        self._Booster = xgb.train(
            params,
            train_dmatrix,
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

            grad = grad.reshape((len(y) * len(self.distribution.params), 1))
            hess = hess.reshape((len(y) * len(self.distribution.params), 1))
            return grad, hess

        return obj

    def _evaluation_func(self):
        def feval(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            return self.distribution.loss(y, params)

        return feval

    def predict_dist(self, X):
        params = self._Booster.predict(xgb.DMatrix(X), output_margin=True)
        return self.distribution.predict(params)
