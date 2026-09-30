# tests/pipelines/test_training_flow.py
from unittest.mock import patch

import pandas as pd
import pytest

from energy_mlops.pipelines import training_flow
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

def test_execute_training_without_optimizer_passes_none_to_trainer():
    """
    Um trainer independente de otimização deve receber explicitamente
    best_params=None e optimization_run_id=None.
    """

    received = {}

    def independent_trainer(
        df_train,
        df_test,
        train_file,
        test_file,
        best_params=None,
        optimization_run_id=None,
    ):
        received["df_train"] = df_train
        received["df_test"] = df_test
        received["train_file"] = train_file
        received["test_file"] = test_file
        received["best_params"] = best_params
        received["optimization_run_id"] = optimization_run_id

        return "independent-run-123", 321.5

    df_train = pd.DataFrame({"value": [1, 2]})
    df_test = pd.DataFrame({"value": [3]})

    run_id, mae = training_flow.execute_training.fn(
        trainer_func=independent_trainer,
        optimization_result=None,
        df_train=df_train,
        df_test=df_test,
        train_file="train.parquet",
        test_file="test.parquet",
    )

    assert run_id == "independent-run-123"
    assert mae == pytest.approx(321.5)

    assert received["df_train"] is df_train
    assert received["df_test"] is df_test
    assert received["train_file"] == "train.parquet"
    assert received["test_file"] == "test.parquet"

    assert received["best_params"] is None
    assert received["optimization_run_id"] is None


def test_continuous_training_pipeline_skips_optimizer_when_none():
    """
    optimizer_name='none' deve pular completamente a etapa de otimização
    e encaminhar optimization_result=None para o treinamento.
    """

    df_train = pd.DataFrame({"value": [1, 2]})
    df_test = pd.DataFrame({"value": [3]})

    fake_datasets = (
        df_train,
        df_test,
        "train.parquet",
        "test.parquet",
    )

    with (
        patch.object(
            training_flow,
            "fetch_expanding_window_data",
            return_value=fake_datasets,
        ) as mock_fetch,
        patch.object(
            training_flow,
            "optimize_hyperparameters",
        ) as mock_optimize,
        patch.object(
            training_flow,
            "execute_training",
            return_value=("challenger-run-123", 456.7),
        ) as mock_training,
        patch.object(
            training_flow,
            "evaluate_and_promote",
        ) as mock_quality_gate,
    ):
        training_flow.continuous_training_pipeline.fn(
            trainer_name="stacking",
            optimizer_name="none",
        )

    mock_fetch.assert_called_once()

    # O ponto principal do teste:
    # optimizer_name="none" não deve executar Optuna.
    mock_optimize.assert_not_called()

    mock_training.assert_called_once_with(
        trainer_func=training_flow.TRAINER_REGISTRY["stacking"],
        optimization_result=None,
        df_train=df_train,
        df_test=df_test,
        train_file="train.parquet",
        test_file="test.parquet",
    )

    mock_quality_gate.assert_called_once_with(
        "challenger-run-123",
        456.7,
    )


def test_continuous_training_pipeline_rejects_unknown_optimizer():
    """
    Um nome de optimizer inexistente deve ser tratado como erro
    de configuração, e não confundido com optimizer_name='none'.
    """

    with pytest.raises(
        ValueError,
        match="Optimizer 'unknown' não encontrado no Registry",
    ):
        training_flow.continuous_training_pipeline.fn(
            trainer_name="stacking",
            optimizer_name="unknown",
        )