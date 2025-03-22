"""XGBDistribution model"""

import importlib
import json
import os
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    no_type_check,
)

import numpy as np
from sklearn.utils.validation import check_is_fitted
from xgboost._typing import ArrayLike
from xgboost.callback import TrainingCallback
from xgboost.config import config_context
from xgboost.core import Booster, DMatrix, _deprecate_positional_args
from xgboost.sklearn import (
    XGBModel,
    XGBRegressor,
    _wrap_evaluation_matrices,
    xgboost_model_doc,
)
from xgboost.training import train

from xgboost_distribution.distributions import get_distribution, get_distribution_doc
from xgboost_distribution.utils import to_serializable


@xgboost_model_doc(
    "Implementation of XGBoost to estimate distributions (in scikit-learn API).",
    ["estimators", "model"],
    extra_parameters=get_distribution_doc()
    + """
    natural_gradient : bool, default=True
        Whether or not natural gradients should be used.""",
)
class XGBDistribution(XGBRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        distribution: Optional[str] = None,
        natural_gradient: bool = True,
        objective: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.distribution = distribution or "normal"
        self.natural_gradient = natural_gradient

        if objective is not None:
            raise ValueError(
                "Please do not set objective directly! Use the `distribution` kwarg"
            )
        else:
            super().__init__(objective=None, **kwargs)

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBDistribution":
        with config_context(verbosity=self.verbosity):
            evals_result: TrainingCallback.EvalsLog = {}

            self._distribution = get_distribution(self.distribution)
            self._distribution.check_target(y)

            params = self.get_xgb_params()

            # we remove unexpected (i.e. not xgb native) params before fitting
            for param in ["distribution", "natural_gradient"]:
                params.pop(param)

            params["objective"] = None
            params["disable_default_eval_metric"] = True
            params["num_class"] = len(self._distribution.params)

            # we set `base_score` to zero and instead use base_margin in dmatrices
            # -> this allows different starting values for each distribution parameter
            params["base_score"] = 0.0
            self._starting_params = self._distribution.starting_params(y)

            base_margin = self._get_base_margin(len(y))
            if eval_set is not None:
                base_margin_eval_set: Optional[List[np.ndarray]] = [
                    self._get_base_margin(len(evals[1])) for evals in eval_set
                ]
            else:
                base_margin_eval_set = None

            model, _, params, feature_weights = self._configure_fit(
                booster=xgb_model,
                params=params,
                feature_weights=feature_weights,
            )

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
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=self.feature_types,
            )

            self._Booster = train(
                params,
                train_dmatrix,
                num_boost_round=self.get_num_boosting_rounds(),
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                obj=self._objective_func(),
                custom_metric=self._evaluation_func(),
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=self.callbacks,
            )

            self._set_evaluation_result(evals_result)
            self.objective = f"distribution:{self.distribution}"

            # we set additional params needed for XGBDistribution on the Booster object,
            # in order to make use of the Booster's serialisation methods
            self._Booster.set_attr(distribution=self.distribution)
            self._Booster.set_attr(
                starting_params=json.dumps(
                    self._starting_params, default=to_serializable
                )
            )
            return self

    assert XGBModel.fit.__doc__ is not None
    fit.__doc__ = XGBModel.fit.__doc__.replace(
        "Fit gradient boosting model", "Fit gradient boosting distribution model", 1
    )

    @no_type_check
    def predict(
        self,
        X: ArrayLike,
        validate_features: bool = True,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray]:
        """Predict all params of distribution of each `X` example.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
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
        with config_context(verbosity=self.verbosity):
            check_is_fitted(self, attributes=("_distribution", "_starting_params"))

            base_margin = self._get_base_margin(X.shape[0])

            params = super().predict(
                X=X,
                output_margin=True,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
            )
            return self._distribution.predict(params)

    def save_model(self, fname: Union[str, os.PathLike]) -> None:
        # self._distribution class cannot be saved by `super().save_model`, as it
        # attempts to call `json.dumps({"_distribution": self._distribution})`
        # Hence we delete, and then reinstantiate
        # (this is safe as distributions are by definition stateless)
        del self._distribution
        super().save_model(fname)
        self._distribution = get_distribution(self.distribution)

    def load_model(self, fname: Union[str, bytearray, os.PathLike]) -> None:
        super().load_model(fname)

        # See above: Currently need to reinstantiate distribution post loading
        self.distribution = self._Booster.attr("distribution")
        self._distribution = get_distribution(self.distribution)

        distribution_module = importlib.import_module(self._distribution.__module__)
        self._starting_params = distribution_module.Params(
            *json.loads(self._Booster.attr("starting_params"))
        )
        del distribution_module

    def _objective_func(
        self,
    ) -> Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]:
        def obj(params: np.ndarray, data: DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            y = data.get_label()
            grad, hess = self._distribution.gradient_and_hessian(
                y=y, params=params, natural_gradient=self.natural_gradient
            )

            weights = data.get_weight()
            if weights.size != 0:
                weights = weights.reshape(-1, 1)
                grad *= weights
                hess *= weights
            return grad, hess

        return obj

    def _evaluation_func(self) -> Callable[[np.ndarray, DMatrix], Tuple[str, float]]:
        def feval(params: np.ndarray, data: DMatrix) -> Tuple[str, float]:
            y = data.get_label()
            weights = data.get_weight()
            if weights.size == 0:
                weights = None

            loss_name, loss = self._distribution.loss(y=y, params=params)
            return loss_name, np.average(loss, weights=weights)

        return feval

    def _get_base_margin(self, n_samples: int) -> np.ndarray:
        return np.ones(shape=(n_samples, 1)) * np.array(self._starting_params)
