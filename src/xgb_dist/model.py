"""XGBDistribution model
"""
import numpy as np
import xgboost as xgb
from xgboost.sklearn import _wrap_evaluation_matrices, xgboost_model_doc

from xgb_dist.distributions import get_distribution, get_distribution_doc


@xgboost_model_doc(
    "Implementation of XGBoost to estimate distributions (scikit-learn API).",
    ["model"],
    extra_parameters=get_distribution_doc(),
)
class XGBDistribution(xgb.XGBModel):
    def __init__(self, distribution=None, **kwargs):
        self.distribution = distribution or "normal"
        super().__init__(objective=None, **kwargs)

    def fit(self, X, y, *, eval_set=None, early_stopping_rounds=None, verbose=True):

        self._distribution = get_distribution(self.distribution)

        params = self.get_xgb_params()
        params["disable_default_eval_metric"] = True
        params["num_class"] = len(self._distribution.params)

        # TODO: setting base_score is crucial to get right as it's the starting
        # point for all params of the distribution
        params["base_score"] = 0.0
        # np.log(np.std(y))

        self._starting_params = self._distribution.starting_params(y)

        self._mean = np.mean(y)
        self._log_scale = np.log(np.std(y))

        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=None,
            base_margin=self._get_base_margins(len(y)),
            feature_weights=None,
            eval_set=eval_set,
            sample_weight_eval_set=None,
            base_margin_eval_set=[self._get_base_margins(len(eval_set[0][1]))],
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

    def predict_dist(self, X):
        """Predict all params of the distribution"""

        params = self._Booster.predict(
            xgb.DMatrix(X, base_margin=self._get_base_margins(X.shape[0])),
            output_margin=True,
        )
        return self._distribution.predict(params)

    def predict(self, X):
        """Predict the first param of the distribution, typically the mean"""
        return self.predict_dist(X)[0]

    def distribution_params(self):
        """Get the names of the paramaters of the distribution"""
        return self._distribution.params

    def _objective_func(self):
        def obj(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            grad, hess = self._distribution.gradient_and_hessian(y, params)

            flattened_len = len(y) * len(self._distribution.params)
            grad = grad.flatten()  # .reshape((flattened_len, 1))
            hess = hess.reshape((flattened_len, 1))
            return grad, hess

        return obj

    def _evaluation_func(self):
        def feval(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            return self._distribution.loss(y, params)

        return feval

    def _get_base_margins(self, n_samples):
        return (
            np.array(
                [
                    self._starting_params[0] * np.ones(shape=(n_samples,)),
                    self._starting_params[1] * np.ones(shape=(n_samples,)),
                ]
            )
            .transpose()
            .flatten()
        )
