"""XGBDistribution model
"""
from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from xgboost import config_context
from xgboost.core import DMatrix
from xgboost.sklearn import XGBModel, _wrap_evaluation_matrices, xgboost_model_doc
from xgboost.training import train

from xgboost_distribution.distributions import get_distribution, get_distribution_doc


@xgboost_model_doc(
    "Implementation of XGBoost to estimate distributions (in scikit-learn API).",
    ["estimators", "model"],
    extra_parameters=get_distribution_doc()
    + """
    natural_gradient : bool, default=True
        Whether or not natural gradients should be used.""",
)
class XGBDistribution(XGBModel, RegressorMixin):
    def __init__(
        self, distribution=None, natural_gradient=True, objective=None, **kwargs
    ):
        self.distribution = distribution or "normal"
        self.natural_gradient = natural_gradient

        if objective is not None:
            raise ValueError(
                "Please do not set object directly! Use the `distribution` kwarg"
            )

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
        """Fit gradient boosting distribution model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        eval_set : list
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once
            in every **early_stopping_rounds** round(s) to continue training.
            Requires at least one item in **eval_set**.

            The method returns the model from the last iteration (not the best one).
            If there's more than one item in **eval_set**, the last entry will be used
            for early stopping.

            If early stopping occurs, the model will have three additional fields:
            ``clf.best_score``, ``clf.best_iteration``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model : `xgboost.core.Booster`, `xgboost.sklearn.XGBModel`
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set : array_like
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        feature_weights : array_like
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.  Only available for `hist`, `gpu_hist`
            and `exact` tree methods.
        callbacks : list
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using :ref:`callback_api`.
            Example:

            .. code-block:: python

                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                        save_best=True)]

        """
        self._distribution = get_distribution(self.distribution)
        self._distribution.check_target(y)

        params = self.get_xgb_params()
        params["objective"] = None
        params["disable_default_eval_metric"] = True
        params["num_class"] = len(self._distribution.params)

        # we set `base_score` to zero and instead use base_margin in dmatrices
        # -> this allows different starting values for each distribution parameter
        params["base_score"] = 0.0
        self._starting_params = self._distribution.starting_params(y)

        base_margin = self._get_base_margin(len(y))
        if eval_set is not None:
            base_margin_eval_set = [
                self._get_base_margin(len(evals[1])) for evals in eval_set
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
            create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
            label_transform=lambda x: x,
        )

        evals_result = {}
        model, _, params = self._configure_fit(xgb_model, None, params)

        # Suppress warnings from unexpected distribution & natural_gradient params
        with config_context(verbosity=0):
            self._Booster = train(
                params,
                train_dmatrix,
                num_boost_round=self.get_num_boosting_rounds(),
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                obj=self._objective_func(),
                feval=self._evaluation_func(),
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=callbacks,
            )

        self._set_evaluation_result(evals_result)
        self.objective = f"distribution:{self.distribution}"

        return self

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
            A namedtuple of the distribution parameters. Each parameter is a
            numpy array of shape (n_samples,).
        """
        check_is_fitted(self, attributes=("_distribution", "_starting_params"))

        base_margin = self._get_base_margin(X.shape[0])

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
        # self._distribution class cannot be saved by `super().save_model`, as it
        # attempts to call `json.dumps({"_distribution": self._distribution})`
        # Hence we delete, and then reinstantiate
        # (this is safe as distributions are by definition stateless)
        del self._distribution
        super().save_model(fname)
        self._distribution = get_distribution(self.distribution)

    def load_model(self, fname) -> None:
        super().load_model(fname)
        # See above: Currently need to reinstantiate distribution post loading
        self._distribution = get_distribution(self.distribution)

    def _objective_func(self) -> Callable:
        def obj(params: np.ndarray, data: DMatrix):
            y = data.get_label()
            grad, hess = self._distribution.gradient_and_hessian(
                y=y, params=params, natural_gradient=self.natural_gradient
            )
            return grad.flatten(), hess.flatten()

        return obj

    def _evaluation_func(self) -> Callable:
        def feval(params: np.ndarray, data: DMatrix):
            y = data.get_label()
            return self._distribution.loss(y=y, params=params)

        return feval

    def _get_base_margin(self, n_samples):
        return (
            np.ones(shape=(n_samples, 1)) * np.array(self._starting_params)
        ).flatten()
