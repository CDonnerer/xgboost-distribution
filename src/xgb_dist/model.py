"""XGBDistribution model
"""
import numpy as np
import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from xgboost.sklearn import XGBModel, _wrap_evaluation_matrices, xgboost_model_doc

from xgb_dist.distributions import get_distribution, get_distribution_doc


@xgboost_model_doc(
    "Implementation of XGBoost to estimate distributions (scikit-learn API).",
    ["model"],
    extra_parameters=get_distribution_doc()
    + """
    natural_gradient : bool, default=True
        Whether or not natural gradients should be used.""",
)
class XGBDistribution(XGBModel, RegressorMixin):
    def __init__(self, distribution=None, natural_gradient=True, **kwargs):
        self.distribution = distribution or "normal"
        self.natural_gradient = natural_gradient
        super().__init__(objective=None, **kwargs)

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        eval_set=None,
        early_stopping_rounds=None,
        verbose=False,
        xgb_model=None,
        sample_weight_eval_set=None,
        feature_weights=None,
        callbacks=None,
    ):
        self._distribution = get_distribution(self.distribution)

        params = self.get_xgb_params()
        params["disable_default_eval_metric"] = True
        params["num_class"] = len(self._distribution.params)

        # we set base score to zero to instead use base_margin in dmatrices
        # this allows different starting values for the distribution params
        params["base_score"] = 0.0
        self._starting_params = self._distribution.starting_params(y)

        base_margin = self._get_base_margins(len(y))
        if eval_set is not None:
            base_margin_eval_set = [
                self._get_base_margins(len(evals[1])) for evals in eval_set
            ]
        else:
            base_margin_eval_set = None

        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
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

    fit.__doc__ = XGBModel.fit.__doc__.replace(
        "Fit gradient boosting model", "Fit gradient boosting distribution", 1
    )

    def predict(
        self,
        X,
        ntree_limit=None,
        validate_features=False,
        iteration_range=None,
    ):
        """Predict all params of distribution of each `X` example.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        ntree_limit : int
            Deprecated, use `iteration_range` instead.
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

        Returns
        -------
        predictions : namedtuple
            A namedtuple of the distribution parameters, each of which is a
            numpy array of shape (n_samples), for each data example.
        """
        check_is_fitted(self, attributes=("_distribution", "_starting_params"))

        base_margin = self._get_base_margins(X.shape[0])

        params = super().predict(
            X=X,
            output_margin=True,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        return self._distribution.predict(params)

    def save_model(self, fname) -> None:
        super().save_model(fname)

    def load_model(self, fname) -> None:
        super().load_model(fname)

        # self._distribution does not get saved in self.save_model(): reinstantiate
        # Note: This is safe, as the distributions are always stateless
        self._distribution = get_distribution(self.distribution)

    def _objective_func(self):
        def obj(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            grad, hess = self._distribution.gradient_and_hessian(
                y, params, self.natural_gradient
            )
            return grad.flatten(), hess.flatten()

        return obj

    def _evaluation_func(self):
        def feval(params: np.ndarray, data: xgb.DMatrix):
            y = data.get_label()
            return self._distribution.loss(y, params)

        return feval

    def _get_base_margins(self, n_samples):
        return (
            np.array(
                [param * np.ones(shape=(n_samples,)) for param in self._starting_params]
            )
            .transpose()
            .flatten()
        )
