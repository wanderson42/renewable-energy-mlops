# tests/pipelines/test_training_flow.py
from unittest.mock import patch

import pandas as pd
import pytest

from energy_mlops.pipelines.training_flow import (
    continuous_training_pipeline,
    resolve_metric,
)


def test_resolve_metric_prefers_canonical_name():
    metrics = {
        "oot_nmae_pct": 7.1,
        "oot_nmae_pct_2026": 99.0,
    }

    result = resolve_metric(
        metrics,
        "oot_nmae_pct",
    )

    assert result == 7.1


def test_resolve_metric_accepts_single_legacy_year():
    metrics = {
        "oot_nmae_pct_2026": 7.04,
    }

    result = resolve_metric(
        metrics,
        "oot_nmae_pct",
    )

    assert result == 7.04


def test_resolve_metric_is_year_agnostic():
    metrics = {
        "oot_nmae_pct_2031": 6.8,
    }

    result = resolve_metric(
        metrics,
        "oot_nmae_pct",
    )

    assert result == 6.8


def test_resolve_metric_rejects_ambiguous_legacy_metrics():
    metrics = {
        "oot_nmae_pct_2025": 7.2,
        "oot_nmae_pct_2026": 7.0,
    }

    with pytest.raises(
        RuntimeError,
        match="múltiplas variantes",
    ):
        resolve_metric(
            metrics,
            "oot_nmae_pct",
        )


def test_resolve_metric_raises_when_required_metric_missing():
    with pytest.raises(
        RuntimeError,
        match="Métrica obrigatória",
    ):
        resolve_metric(
            {},
            "oot_nmae_pct",
        )

def dummy_trainer(
    df_train,
    df_test,
    train_file,
    test_file,
    best_params,
    optimization_run_id,
):
    assert best_params == {
        "lgb_params": {
            "n_estimators": 200,
        },
        "xgb_params": {
            "max_depth": 5,
        },
        "rf_params": {
            "n_estimators": 100,
        },
    }

    assert optimization_run_id == "fake_optimization_run_123"

    return "fake_run_id_999", 3.14


def dummy_optimizer(
    df_train,
    train_file,
    n_trials=10,
    n_splits=3,
):
    return {
        "optimization_run_id": "fake_optimization_run_123",
        "best_params": {
            "lgb_params": {
                "n_estimators": 200,
            },
            "xgb_params": {
                "max_depth": 5,
            },
            "rf_params": {
                "n_estimators": 100,
            },
        },
    }


@patch(
    "energy_mlops.pipelines.training_flow.evaluate_and_promote"
)
@patch(
    "energy_mlops.pipelines.training_flow.fetch_expanding_window_data"
)
@patch.dict(
    "energy_mlops.pipelines.training_flow.TRAINER_REGISTRY",
    {"dummy": dummy_trainer},
)
@patch.dict(
    "energy_mlops.pipelines.training_flow.OPTIMIZER_REGISTRY",
    {"dummy": dummy_optimizer},
)
def test_continuous_training_pipeline_orchestration(
    mock_fetch,
    mock_evaluate,
):
    df_vazio = pd.DataFrame(
        columns=["feature_1", "target_fc"]
    )

    mock_fetch.return_value = (
        df_vazio,
        df_vazio,
        "dummy_train.parquet",
        "dummy_test.parquet",
    )

    mock_evaluate.return_value = None

    continuous_training_pipeline(
        trainer_name="dummy",
        optimizer_name="dummy",
    )

    mock_fetch.assert_called_once()

    mock_evaluate.assert_called_once_with(
        "fake_run_id_999",
        3.14,
    )