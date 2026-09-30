import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from energy_mlops.models.temporal_stacking import (
    TemporalStackingRegressor,
)


class LastSeenTimeRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """
    Estimador de teste que revela qual foi o maior
    instante temporal observado durante fit().
    """

    def fit(self, X, y):
        self.max_time_seen_ = float(
            X["time"].max()
        )

        return self

    def predict(self, X):
        return np.full(
            len(X),
            self.max_time_seen_,
        )


def test_temporal_stacking_oof_never_uses_future_data():
    X = pd.DataFrame(
        {
            "time": np.arange(
                12,
                dtype=float,
            ),
            "feature": np.arange(
                12,
                dtype=float,
            ),
        }
    )

    y = pd.Series(
        np.arange(
            12,
            dtype=float,
        )
    )

    ensemble = TemporalStackingRegressor(
        estimators=[
            (
                "model_a",
                LastSeenTimeRegressor(),
            ),
            (
                "model_b",
                LastSeenTimeRegressor(),
            ),
        ],
        final_estimator=LinearRegression(),
        cv=TimeSeriesSplit(
            n_splits=3,
        ),
    )

    ensemble.fit(
        X,
        y,
    )

    valid_indices = (
        ensemble.meta_training_indices_
    )

    oof_predictions = (
        ensemble.oof_predictions_[
            valid_indices
        ]
    )

    validation_times = (
        X.iloc[valid_indices]["time"]
        .to_numpy()
    )

    # Cada previsão informa o maior timestamp
    # visto pelo modelo que a produziu.
    #
    # Esse timestamp deve ser SEMPRE anterior
    # ao timestamp da observação validada.
    for estimator_idx in range(
        oof_predictions.shape[1]
    ):
        assert np.all(
            oof_predictions[
                :,
                estimator_idx,
            ]
            < validation_times
        )


def test_temporal_stacking_excludes_initial_block_from_meta_training():
    X = pd.DataFrame(
        {
            "time": np.arange(
                12,
                dtype=float,
            ),
            "feature": np.arange(
                12,
                dtype=float,
            ),
        }
    )

    y = pd.Series(
        np.arange(
            12,
            dtype=float,
        )
    )

    ensemble = TemporalStackingRegressor(
        estimators=[
            (
                "model",
                LastSeenTimeRegressor(),
            ),
        ],
        final_estimator=LinearRegression(),
        cv=TimeSeriesSplit(
            n_splits=3,
        ),
    )

    ensemble.fit(
        X,
        y,
    )

    missing_oof = (
        ~ensemble.oof_mask_
    )

    assert missing_oof.any()

    assert np.isnan(
        ensemble.oof_predictions_[
            missing_oof
        ]
    ).all()


def test_temporal_stacking_refits_base_models_on_full_training_data():
    X = pd.DataFrame(
        {
            "time": np.arange(
                12,
                dtype=float,
            ),
            "feature": np.arange(
                12,
                dtype=float,
            ),
        }
    )

    y = pd.Series(
        np.arange(
            12,
            dtype=float,
        )
    )

    ensemble = TemporalStackingRegressor(
        estimators=[
            (
                "model",
                LastSeenTimeRegressor(),
            ),
        ],
        final_estimator=LinearRegression(),
        cv=TimeSeriesSplit(
            n_splits=3,
        ),
    )

    ensemble.fit(
        X,
        y,
    )

    fitted_model = (
        ensemble.named_estimators_[
            "model"
        ]
    )

    assert (
        fitted_model.max_time_seen_
        == 11.0
    )

    predictions = ensemble.predict(
        X.tail(2)
    )

    assert predictions.shape == (2,)