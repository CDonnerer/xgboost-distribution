======================
Experiments
======================

We performed experiments on ``XGBDistribution`` across various datasets for probabilistic
regression tasks. Comparison were made with both `NGBoost`_'s ``NGBRegressor``, as well
as a standard `xgboost`_ ``XGBRegressor`` (point estimate only).

Probabilistic regression
========================

For probabilistic regression, within errorbars, ``XGBDistribution`` performs essentially
identically to ``NGBRegressor`` (measured on the negative log likelihood [NLL] of a normal
distribution).

However, ``XGBDistribution`` is **substantially faster, typically at least an order of
magnitude**. For example, for the MSD dataset, the fit and predict steps took 18 minutes
for ``XGBDistribution`` vs a full 6.7 hours for ``NGBRegressor``:

+-----------------+---------------------------+---------------------------+
|                 | XGBDistribution           | NGBRegressor              |
+---------+-------+-----------+---------------+-----------+---------------+
| Dataset | N     | NLL       | Time          | NLL       | Time          |
+=========+=======+===========+===============+===========+===============+
| Boston  |506    | 2.62(26)  | 0.067(1) (s)  | 2.55(24)  | 2.68(45) (s)  |
+---------+-------+-----------+---------------+-----------+---------------+
| Concrete|1,030  | 3.14(21)  | 0.13(3) (s)   | 3.09(13)  | 5.79(59) (s)  |
+---------+-------+-----------+---------------+-----------+---------------+
| Energy  |768    | 0.58(41)  | 0.15(3) (s)   | 0.62(28)  | 5.33(35) (s)  |
+---------+-------+-----------+---------------+-----------+---------------+
| Naval   |11,934 | -5.11(6)  | 5.8(8) (s)    | -3.91(2)  | 43.6(5) (s)   |
+---------+-------+-----------+---------------+-----------+---------------+
| Power   |9,568  | 2.77(11)  | 1.21(52) (s)  | 2.77(7)   | 14.9(3.1) (s) |
+---------+-------+-----------+---------------+-----------+---------------+
| Protein |45,730 | 2.81(4)   | 12.2(4.0) (s) | 2.91(1)   | 146.5(1.8) (s)|
+---------+-------+-----------+---------------+-----------+---------------+
| Wine    |1,588  | 0.98(15)  | 0.11(3) (s)   | 0.93(7)   | 4.85(99) (s)  |
+---------+-------+-----------+---------------+-----------+---------------+
| Yacht   |308    | 0.89(1.1) | 0.093(25) (s) | 0.75(64)  | 4.95(50) (s)  |
+---------+-------+-----------+---------------+-----------+---------------+
| MSD     |515,345| 3.450(4)  | 18.9(1.5) (m) | 3.526(4)  | 6.70(7) (h)   |
+---------+-------+-----------+---------------+-----------+---------------+


Point estimation
========================

For point estimates, we compared ``XGBDistribution`` to both the ``NGBRegressor`` and the
``XGBRegressor`` (measured on the RMSE). Generally, the ``XGBRegressor`` will offer the
best performance for this task. However, compared with ``XGBRegressor``,
``XGBDistribution`` only incurs small penalties on both performance and speed, thus
making ``XGBDistribution`` a viable "drop-in" replacement to obtain probabilistic predictions.

+---------+---------------------------+---------------------------+---------------------------+
|         | XGBDistribution           | NGBRegressor              | XGBRegressor              |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Dataset | RMSE      | Time          | RMSE      | Time          | RMSE      | Time          |
+=========+===========+===============+===========+===============+===========+===============+
| Boston  | 3.41(69)  | 0.067(1) (s)  | 3.25(66)  | 2.68(45) (s)  | 3.27(65)  | 0.035(1) (s)  |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Concrete| 5.41(74)  | 0.13(3) (s)   | 5.62(69)  | 5.79(59) (s)  | 4.38(70)  | 0.09(2) (s)   |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Energy  | 0.45(7)   | 0.15(3) (s)   | 0.49(7)   | 5.33(35) (s)  | 0.40(6)   | 0.05(2) (s)   |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Naval   | 0.0014(1) | 5.8(8) (s)    | 0.0059(1) | 43.6(5) (s)   | 0.00123(5)| 1.93(7) (s)   |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Power   | 3.79(24)  | 1.21(52) (s)  | 3.93(19)  | 14.9(3.1) (s) | 3.31(22)  | 0.59(19) (s)  |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Protein | 4.35(12)  | 12.2(4.0) (s) | 4.78(5)   | 146.5(1.8) (s)| 4.09(7)   | 8.26(1.4) (s) |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Wine    | 0.63(4)   | 0.11(3) (s)   | 0.62(3)   | 4.85(99) (s)  | 0.62(3)   | 0.035(13) (s) |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| Yacht   | 0.76(29)  | 0.093(25) (s) | 0.74(28)  | 4.95(50) (s)  | 0.74(37)  | 0.047(35) (s) |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+
| MSD     | 9.03(4)   | 18.9(1.5) (m) | 9.47(4)   | 6.70(7) (h)   | 8.97(3)   | 16.3(1.7) (m) |
+---------+-----------+---------------+-----------+---------------+-----------+---------------+


Methodology
========================

We used 10-fold cross-validation. In each training fold, 10% of the data were used as a
validation set for early stopping. This process was repeated over 5 random seeds. For
the MSD dataset, we used a single 5-fold cross-validation.

The negative log-likelihood (NLL) and root mean squared error (RMSE) were estimated
for each test fold, the above are the mean and standard deviation of these metrics
(across folds and random seeds).

For all estimators, we used default hyperparameters, with the exception of setting
``max_depth=3`` in ``XGBDistribution`` and ``XGBRegressor``, since this is the default
value of ``NGBRegressor``. For all experiments, ``XGBDistribution`` and ``NGBRegressor``
estimated normal distributions, with natural gradients.

Please see the `experiments script`_ for the full details.


.. _ngboost: https://github.com/stanfordmlgroup/ngboost
.. _xgboost: https://xgboost.readthedocs.io/en/latest/
.. _experiments script: https://github.com/CDonnerer/xgboost-distribution/blob/main/examples/experiments.py
