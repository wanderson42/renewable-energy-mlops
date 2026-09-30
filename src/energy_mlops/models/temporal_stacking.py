import numpy as np
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    clone,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


def _take_rows(data, indices):
    """Seleciona linhas preservando suporte a DataFrame ou ndarray."""

    if hasattr(data, "iloc"):
        return data.iloc[indices]

    return data[indices]


class TemporalStackingRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """
    Stacking causal para séries temporais.

    As previsões usadas pelo meta-learner são produzidas
    exclusivamente por modelos treinados em observações
    anteriores às respectivas observações de validação.
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        cv=None,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        if self.cv is None:
            raise ValueError(
                "TemporalStackingRegressor exige uma "
                "estratégia temporal de validação em 'cv'."
            )

        if not self.estimators:
            raise ValueError(
                "Nenhum estimador base foi fornecido."
            )

        y_array = np.asarray(y)

        n_samples = len(X)
        n_estimators = len(self.estimators)

        # NaN identifica observações que ainda não possuem
        # previsão OOF temporal.
        oof_predictions = np.full(
            shape=(n_samples, n_estimators),
            fill_value=np.nan,
            dtype=float,
        )

        # --------------------------------------------------
        # 1. Geração das previsões OOF de forma temporal
        # --------------------------------------------------
        for train_idx, val_idx in self.cv.split(X):
            X_train_fold = _take_rows(
                X,
                train_idx,
            )

            X_val_fold = _take_rows(
                X,
                val_idx,
            )

            y_train_fold = y_array[train_idx]

            for estimator_idx, (_, estimator) in enumerate(
                self.estimators
            ):
                fold_estimator = clone(estimator)

                fold_estimator.fit(
                    X_train_fold,
                    y_train_fold,
                )

                fold_predictions = (
                    fold_estimator.predict(
                        X_val_fold
                    )
                )

                oof_predictions[
                    val_idx,
                    estimator_idx,
                ] = fold_predictions

        # TimeSeriesSplit não gera OOF para o primeiro
        # bloco porque não existe passado suficiente.
        oof_mask = ~np.isnan(
            oof_predictions
        ).any(axis=1)

        if not np.any(oof_mask):
            raise RuntimeError(
                "Nenhuma previsão OOF temporal foi gerada."
            )

        # Guardamos para auditoria e testes.
        self.oof_predictions_ = oof_predictions
        self.oof_mask_ = oof_mask
        self.meta_training_indices_ = np.flatnonzero(
            oof_mask
        )

        # --------------------------------------------------
        # 2. Treinamento do meta-learner SOMENTE com OOF
        # --------------------------------------------------
        if self.final_estimator is None:
            final_estimator = LinearRegression()
        else:
            final_estimator = clone(
                self.final_estimator
            )

        self.final_estimator_ = final_estimator

        self.final_estimator_.fit(
            oof_predictions[oof_mask],
            y_array[oof_mask],
        )

        # --------------------------------------------------
        # 3. Refit dos modelos base com TODO o treino
        #
        # Estes são os modelos usados em produção.
        # --------------------------------------------------
        self.estimators_ = []
        self.named_estimators_ = {}

        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator)

            fitted_estimator.fit(
                X,
                y_array,
            )

            self.estimators_.append(
                fitted_estimator
            )

            self.named_estimators_[
                name
            ] = fitted_estimator

        self.n_features_in_ = X.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(
                X.columns,
                dtype=object,
            )

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "final_estimator_",
                "named_estimators_",
            ],
        )

        base_predictions = np.column_stack(
            [
                self.named_estimators_[
                    name
                ].predict(X)
                for name, _ in self.estimators
            ]
        )

        return self.final_estimator_.predict(
            base_predictions
        )