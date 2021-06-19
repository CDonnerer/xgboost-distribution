"""XGBDistribution model
"""
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from xgboost.core import DMatrix
from xgboost.sklearn import XGBModel, _wrap_evaluation_matrices, xgboost_model_doc
from xgboost.training import train

from xgb_dist.distributions import get_distribution, get_distribution_doc


@xgboost_model_doc(
    "Implementation of XGBoost to estimate distributions (in scikit-learn API).",
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
            create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
            label_transform=lambda x: x,
        )

        evals_result = {}
        model, _, params = self._configure_fit(xgb_model, None, params)
        if len(X.shape) != 2:
            # Simply raise an error here since there might be many
            # different ways of reshaping
            raise ValueError(
                "Please reshape the input data X into 2-dimensional matrix."
            )

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
        def obj(params: np.ndarray, data: DMatrix):
            y = data.get_label()
            grad, hess = self._distribution.gradient_and_hessian(
                y, params, self.natural_gradient
            )
            return grad.flatten(), hess.flatten()

        return obj

    def _evaluation_func(self):
        def feval(params: np.ndarray, data: DMatrix):
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
