from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from energy_mlops.models.evaluation import (
    align_features_to_model,
    evaluate_model_on_oot,
)


def test_evaluate_model_on_oot_uses_model_feature_order_and_calculates_metrics():
    model = SimpleNamespace(
        feature_names_in_=np.array(
            [
                "wind_speed_100m",
                "temperature_2m",
            ],
            dtype=object,
        )
    )

    received = {}

    def predict(X):
        received["columns"] = list(X.columns)
        return np.array([0.50, 0.50])

    model.predict = predict

    X = pd.DataFrame(
        {
            "temperature_2m": [25.0, 26.0],
            "wind_speed_100m": [8.0, 9.0],
        }
    )

    df_oot = pd.DataFrame(
        {
            "wind_generation_mw": [60.0, 40.0],
            "capacidade_mw": [100.0, 100.0],
        }
    )

    result = evaluate_model_on_oot(
        model,
        X,
        df_oot,
    )

    assert received["columns"] == [
        "wind_speed_100m",
        "temperature_2m",
    ]
    assert result.mae_mw == pytest.approx(10.0)
    assert result.nmae_pct == pytest.approx(10.0)


def test_evaluate_model_on_oot_rejects_missing_ground_truth():
    model = SimpleNamespace(
        feature_names_in_=np.array(["feature"]),
        predict=lambda X: np.array([0.5]),
    )

    X = pd.DataFrame({"feature": [1.0]})
    df_oot = pd.DataFrame({"capacidade_mw": [100.0]})

    with pytest.raises(
        ValueError,
        match="ground truth e capacidade",
    ):
        evaluate_model_on_oot(
            model,
            X,
            df_oot,
        )


def test_evaluate_model_on_oot_rejects_invalid_capacity():
    model = SimpleNamespace(
        feature_names_in_=np.array(["feature"]),
        predict=lambda X: np.array([0.5]),
    )

    X = pd.DataFrame({"feature": [1.0]})
    df_oot = pd.DataFrame(
        {
            "wind_generation_mw": [0.0],
            "capacidade_mw": [0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Capacidade máxima inválida",
    ):
        evaluate_model_on_oot(
            model,
            X,
            df_oot,
        )


def test_align_features_to_model_rejects_missing_feature():
    model = SimpleNamespace(
        feature_names_in_=np.array(
            ["wind_speed_100m", "temperature_2m"]
        )
    )

    X = pd.DataFrame(
        {
            "temperature_2m": [25.0],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Features obrigatórias",
    ):
        align_features_to_model(
            model,
            X,
        )
