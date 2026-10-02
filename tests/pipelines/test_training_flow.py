# tests/pipelines/test_training_flow.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_mlops.pipelines import training_flow
from energy_mlops.pipelines.training_flow import resolve_metric


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


def test_continuous_training_pipeline_orchestration():
    """
    Valida a orquestração completa com uma rolling window explícita:

    Gold explícito
        -> auditoria
        -> split temporal configurável
        -> auditoria train/OOT
        -> persistência
        -> otimização
        -> treinamento
        -> quality gate
    """

    snapshot_path = (
        "s3://energy-lake/gold/"
        "dataset_renewable_energy_2024_03_2026_09.parquet"
    )

    df_gold = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-31 23:00:00",
                    "2026-09-01 03:00:00",
                ]
            ),
        }
    )

    df_train = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-31 23:00:00",
                ]
            ),
        }
    )

    df_test = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-09-01 03:00:00",
                ]
            ),
        }
    )

    optimization_result = {
        "optimization_run_id": (
            "optimization-run-123"
        ),
        "best_params": {
            "example": "params",
        },
    }

    with (
        patch.object(
            training_flow,
            "load_gold_snapshot_task",
            return_value=df_gold,
        ) as mock_load,
        patch.object(
            training_flow,
            "audit_gold_snapshot_task",
        ) as mock_audit,
        patch.object(
            training_flow,
            "split_training_window_task",
            return_value=(
                df_train,
                df_test,
            ),
        ) as mock_split,
        patch.object(
            training_flow,
            "persist_training_datasets_task",
            return_value=(
                "train.parquet",
                "test.parquet",
            ),
        ) as mock_persist,
        patch.object(
            training_flow,
            "optimize_hyperparameters",
            return_value=optimization_result,
        ) as mock_optimize,
        patch.object(
            training_flow,
            "execute_training",
            return_value=(
                "challenger-run-123",
                456.7,
            ),
        ) as mock_training,
        patch.object(
            training_flow,
            "evaluate_and_promote",
        ) as mock_quality_gate,
    ):
        training_flow.continuous_training_pipeline.fn(
            trainer_name="stacking",
            optimizer_name="stacking",
            snapshot_path=snapshot_path,
            oot_year=2026,
            oot_month=9,
            training_window_months=12,
        )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------
    mock_load.assert_called_once_with(
        snapshot_path=snapshot_path,
    )

    # --------------------------------------------------
    # Auditoria
    # consolidated + train + oot
    # --------------------------------------------------
    assert mock_audit.call_count == 3

    assert (
        mock_audit.call_args_list[0]
        .kwargs["dataset_role"]
        == "consolidated"
    )

    assert (
        mock_audit.call_args_list[0]
        .args[0]
        is df_gold
    )

    assert (
        mock_audit.call_args_list[1]
        .kwargs["dataset_role"]
        == "train"
    )

    assert (
        mock_audit.call_args_list[1]
        .args[0]
        is df_train
    )

    assert (
        mock_audit.call_args_list[2]
        .kwargs["dataset_role"]
        == "oot"
    )

    assert (
        mock_audit.call_args_list[2]
        .args[0]
        is df_test
    )

    # --------------------------------------------------
    # Split
    # --------------------------------------------------
    mock_split.assert_called_once()

    split_args = mock_split.call_args

    assert (
        split_args.args[0]
        is df_gold
    )

    assert (
        split_args.kwargs["oot_year"]
        == 2026
    )

    assert (
        split_args.kwargs["oot_month"]
        == 9
    )

    assert (
        split_args.kwargs[
            "training_window_months"
        ]
        == 12
    )

    # --------------------------------------------------
    # Persistência
    # --------------------------------------------------
    mock_persist.assert_called_once()

    persist_args = mock_persist.call_args

    assert (
        persist_args.args[0]
        is df_train
    )

    assert (
        persist_args.args[1]
        is df_test
    )

    assert (
        persist_args.kwargs["oot_year"]
        == 2026
    )

    assert (
        persist_args.kwargs["oot_month"]
        == 9
    )

    assert (
        persist_args.kwargs[
            "training_window_months"
        ]
        == 12
    )

    # --------------------------------------------------
    # Optuna
    # --------------------------------------------------
    mock_optimize.assert_called_once()

    optimize_kwargs = (
        mock_optimize.call_args.kwargs
    )

    assert (
        optimize_kwargs["df_train"]
        is df_train
    )

    assert (
        optimize_kwargs["train_file"]
        == "train.parquet"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    mock_training.assert_called_once()

    training_kwargs = (
        mock_training.call_args.kwargs
    )

    assert (
        training_kwargs[
            "optimization_result"
        ]
        == optimization_result
    )

    assert (
        training_kwargs["df_train"]
        is df_train
    )

    assert (
        training_kwargs["df_test"]
        is df_test
    )

    assert (
        training_kwargs["train_file"]
        == "train.parquet"
    )

    assert (
        training_kwargs["test_file"]
        == "test.parquet"
    )

    # --------------------------------------------------
    # Quality Gate
    # --------------------------------------------------
    mock_quality_gate.assert_called_once_with(
        "challenger-run-123",
        df_test,
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
    optimizer_name='none' deve pular completamente
    a otimização, mantendo todas as etapas de
    preparação e auditoria dos dados.

    Sem training_window_months explícito, o contrato
    padrão deve continuar sendo expanding window.
    """

    snapshot_path = (
        "s3://energy-lake/gold/"
        "dataset_test.parquet"
    )

    df_gold = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 03:00:00",
                    "2026-09-01 03:00:00",
                ]
            ),
        }
    )

    df_train = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 03:00:00",
                ]
            ),
        }
    )

    df_test = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-09-01 03:00:00",
                ]
            ),
        }
    )

    with (
        patch.object(
            training_flow,
            "load_gold_snapshot_task",
            return_value=df_gold,
        ) as mock_load,
        patch.object(
            training_flow,
            "audit_gold_snapshot_task",
        ) as mock_audit,
        patch.object(
            training_flow,
            "split_training_window_task",
            return_value=(
                df_train,
                df_test,
            ),
        ) as mock_split,
        patch.object(
            training_flow,
            "persist_training_datasets_task",
            return_value=(
                "train.parquet",
                "test.parquet",
            ),
        ) as mock_persist,
        patch.object(
            training_flow,
            "optimize_hyperparameters",
        ) as mock_optimize,
        patch.object(
            training_flow,
            "execute_training",
            return_value=(
                "challenger-run-123",
                456.7,
            ),
        ) as mock_training,
        patch.object(
            training_flow,
            "evaluate_and_promote",
        ) as mock_quality_gate,
    ):
        training_flow.continuous_training_pipeline.fn(
            trainer_name="stacking",
            optimizer_name="none",
            snapshot_path=snapshot_path,
            oot_year=2026,
            oot_month=9,
        )

    mock_load.assert_called_once_with(
        snapshot_path=snapshot_path,
    )

    assert mock_audit.call_count == 3

    mock_split.assert_called_once()

    split_kwargs = (
        mock_split.call_args.kwargs
    )

    assert (
        split_kwargs["training_window_months"]
        is None
    )

    mock_persist.assert_called_once()

    persist_kwargs = (
        mock_persist.call_args.kwargs
    )

    assert (
        persist_kwargs["training_window_months"]
        is None
    )

    # Este é o contrato central deste teste.
    mock_optimize.assert_not_called()

    mock_training.assert_called_once()

    training_kwargs = (
        mock_training.call_args.kwargs
    )

    assert (
        training_kwargs[
            "optimization_result"
        ]
        is None
    )

    assert (
        training_kwargs["df_train"]
        is df_train
    )

    assert (
        training_kwargs["df_test"]
        is df_test
    )

    assert (
        training_kwargs["train_file"]
        == "train.parquet"
    )

    assert (
        training_kwargs["test_file"]
        == "test.parquet"
    )

    mock_quality_gate.assert_called_once_with(
        "challenger-run-123",
        df_test,
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


def test_get_local_month_bounds_utc():
    start, end = (
        training_flow
        .get_local_month_bounds_utc(
            2026,
            9,
        )
    )

    assert start == pd.Timestamp(
        "2026-09-01 03:00:00"
    )

    assert end == pd.Timestamp(
        "2026-10-01 03:00:00"
    )


@pytest.mark.parametrize(
    (
        "training_window_months",
        "expected_train_start",
        "expected_train_file",
    ),
    [
        (
            None,
            "2024-03-21 00:00:00",
            (
                "train_wind_energy_2024_03_"
                "expanding_up_to_2026_09.parquet"
            ),
        ),
        (
            24,
            "2024-09-01 03:00:00",
            (
                "train_wind_energy_2024_09_"
                "rolling_24m_up_to_2026_09.parquet"
            ),
        ),
        (
            12,
            "2025-09-01 03:00:00",
            (
                "train_wind_energy_2025_09_"
                "rolling_12m_up_to_2026_09.parquet"
            ),
        ),
    ],
)
def test_training_window_split_and_labels_track_strategy(
    training_window_months,
    expected_train_start,
    expected_train_file,
):
    """
    Expanding, rolling 24m e rolling 12m devem
    compartilhar exatamente o mesmo OOT.

    Somente o limite inferior do treino e o label
    do artefato de treino podem mudar.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-03-21 00:00:00",
                    "2024-09-01 02:00:00",
                    "2024-09-01 03:00:00",
                    "2025-09-01 02:00:00",
                    "2025-09-01 03:00:00",
                    "2026-09-01 01:00:00",
                    "2026-09-01 02:00:00",
                    "2026-09-01 03:00:00",
                    "2026-09-01 04:00:00",
                    "2026-10-01 02:00:00",
                ]
            ),
        }
    )

    train, oot = (
        training_flow
        .split_training_window_snapshot(
            df,
            oot_year=2026,
            oot_month=9,
            training_window_months=(
                training_window_months
            ),
        )
    )

    assert train["date"].min() == (
        pd.Timestamp(
            expected_train_start
        )
    )

    assert train["date"].max() == (
        pd.Timestamp(
            "2026-09-01 02:00:00"
        )
    )

    assert oot["date"].min() == (
        pd.Timestamp(
            "2026-09-01 03:00:00"
        )
    )

    assert oot["date"].max() == (
        pd.Timestamp(
            "2026-10-01 02:00:00"
        )
    )

    assert (
        train["date"].max()
        < oot["date"].min()
    )

    train_file, test_file = (
        training_flow
        .build_training_dataset_labels(
            train,
            oot_year=2026,
            oot_month=9,
            training_window_months=(
                training_window_months
            ),
        )
    )

    assert (
        train_file
        == expected_train_file
    )

    assert (
        test_file
        == (
            "oot_test_wind_energy_"
            "2026_09.parquet"
        )
    )


def test_split_training_window_snapshot_rejects_incomplete_oot():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 03:00:00",
                    "2026-09-01 03:00:00",
                    "2026-09-30 02:00:00",
                ]
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="mês OOT completo",
    ):
        (
            training_flow
            .split_training_window_snapshot(
                df,
                oot_year=2026,
                oot_month=9,
                training_window_months=None,
            )
        )


def test_split_training_window_snapshot_rejects_insufficient_history():
    """
    Uma rolling window não pode representar mais
    histórico do que o snapshot realmente contém.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-03-21 00:00:00",
                    "2026-09-01 02:00:00",
                    "2026-09-01 03:00:00",
                    "2026-10-01 02:00:00",
                ]
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="não cobre integralmente",
    ):
        (
            training_flow
            .split_training_window_snapshot(
                df,
                oot_year=2026,
                oot_month=9,
                training_window_months=24,
            )
        )


def test_split_training_window_snapshot_rejects_invalid_window():
    """
    A largura da rolling window deve ser um inteiro
    positivo. None é reservado à expanding window.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-03-21 00:00:00",
                    "2026-09-01 02:00:00",
                    "2026-09-01 03:00:00",
                    "2026-10-01 02:00:00",
                ]
            ),
        }
    )

    for invalid_window in (
        0,
        -1,
        1.5,
        True,
    ):
        with pytest.raises(
            ValueError,
            match=(
                "training_window_months deve ser "
                "um inteiro positivo ou None"
            ),
        ):
            (
                training_flow
                .split_training_window_snapshot(
                    df,
                    oot_year=2026,
                    oot_month=9,
                    training_window_months=(
                        invalid_window
                    ),
                )
            )


def test_split_expanding_window_snapshot_remains_compatible():
    """
    O wrapper histórico de expanding window deve
    continuar equivalente à nova função genérica
    com training_window_months=None.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-31 23:00:00",
                    "2026-09-01 02:00:00",
                    "2026-09-01 03:00:00",
                    "2026-10-01 02:00:00",
                ]
            ),
        }
    )

    legacy_train, legacy_oot = (
        training_flow
        .split_expanding_window_snapshot(
            df,
            oot_year=2026,
            oot_month=9,
        )
    )

    generic_train, generic_oot = (
        training_flow
        .split_training_window_snapshot(
            df,
            oot_year=2026,
            oot_month=9,
            training_window_months=None,
        )
    )

    pd.testing.assert_frame_equal(
        legacy_train,
        generic_train,
    )

    pd.testing.assert_frame_equal(
        legacy_oot,
        generic_oot,
    )

# ==============================================================================
# QUALITY GATE — SAME OOT
# ==============================================================================


def _make_quality_gate_run(
    *,
    num_features: int,
    historical_mae: float,
    historical_nmae: float,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            params={
                "num_features": str(num_features),
            },
            metrics={
                "oot_mae_mw": historical_mae,
                "oot_nmae_pct": historical_nmae,
            },
        )
    )


def _make_quality_gate_oot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-09-01 03:00:00",
                    "2026-09-01 04:00:00",
                ]
            ),
            "temperature_2m": [25.0, 25.5],
            "wind_speed_100m": [8.0, 9.0],
            "wind_direction_100m": [180.0, 190.0],
            "wind_generation_mw": [50.0, 55.0],
            "capacidade_mw": [100.0, 100.0],
            "target_fc": [0.50, 0.55],
        }
    )


def _run_quality_gate_with_same_oot_metrics(
    *,
    champion_mae: float,
    champion_nmae: float,
    challenger_mae: float,
    challenger_nmae: float,
    champion_features: int = 9,
    challenger_features: int = 9,
):
    df_oot = _make_quality_gate_oot()

    challenger_run = _make_quality_gate_run(
        num_features=challenger_features,
        # Valores históricos deliberadamente incompatíveis
        # com a avaliação atual. O gate não deve usá-los.
        historical_mae=9999.0,
        historical_nmae=99.0,
    )

    champion_run = _make_quality_gate_run(
        num_features=champion_features,
        historical_mae=1.0,
        historical_nmae=1.0,
    )

    fake_client = MagicMock()
    fake_client.get_run.side_effect = [
        challenger_run,
        champion_run,
    ]
    fake_client.get_model_version_by_alias.return_value = (
        SimpleNamespace(
            run_id="champion-run-10",
            version="10",
        )
    )
    fake_client.search_model_versions.return_value = [
        SimpleNamespace(
            version="20",
        )
    ]

    champion_model = MagicMock(name="champion_model")
    challenger_model = MagicMock(name="challenger_model")

    X_oot = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    champion_eval = SimpleNamespace(
        mae_mw=champion_mae,
        nmae_pct=champion_nmae,
    )
    challenger_eval = SimpleNamespace(
        mae_mw=challenger_mae,
        nmae_pct=challenger_nmae,
    )

    with (
        patch.object(
            training_flow,
            "MlflowClient",
            return_value=fake_client,
        ),
        patch.object(
            training_flow.mlflow,
            "set_tracking_uri",
        ),
        patch.object(
            training_flow.mlflow.sklearn,
            "load_model",
            side_effect=[
                champion_model,
                challenger_model,
            ],
        ) as mock_load_model,
        patch.object(
            training_flow,
            "select_model_features",
            return_value=X_oot,
        ) as mock_select_features,
        patch.object(
            training_flow,
            "evaluate_model_on_oot",
            side_effect=[
                champion_eval,
                challenger_eval,
            ],
        ) as mock_evaluate,
        patch.object(
            training_flow.requests,
            "post",
            return_value=MagicMock(),
        ),
    ):
        training_flow.evaluate_and_promote.fn(
            "challenger-run-20",
            df_oot,
        )

    return {
        "client": fake_client,
        "df_oot": df_oot,
        "X_oot": X_oot,
        "champion_model": champion_model,
        "challenger_model": challenger_model,
        "mock_load_model": mock_load_model,
        "mock_select_features": mock_select_features,
        "mock_evaluate": mock_evaluate,
    }


def test_quality_gate_promotes_challenger_using_same_oot():
    result = _run_quality_gate_with_same_oot_metrics(
        champion_mae=900.0,
        champion_nmae=7.60,
        challenger_mae=800.0,
        challenger_nmae=6.80,
    )

    client = result["client"]

    client.set_registered_model_alias.assert_called_once_with(
        training_flow.MODEL_NAME,
        training_flow.MODEL_ALIAS,
        "20",
    )

    result["mock_evaluate"].assert_any_call(
        result["champion_model"],
        result["X_oot"],
        result["df_oot"],
    )
    result["mock_evaluate"].assert_any_call(
        result["challenger_model"],
        result["X_oot"],
        result["df_oot"],
    )

    # A métrica histórica do Champion era artificialmente melhor
    # (1 MW), mas a decisão correta usa 900 vs 800 no mesmo OOT.
    client.set_tag.assert_any_call(
        "challenger-run-20",
        "quality_gate_decision_reason",
        "MAE_IMPROVED_SAME_OOT",
    )


def test_quality_gate_rejects_worse_challenger_on_same_oot():
    result = _run_quality_gate_with_same_oot_metrics(
        champion_mae=800.0,
        champion_nmae=6.80,
        challenger_mae=900.0,
        challenger_nmae=7.60,
    )

    client = result["client"]

    client.set_registered_model_alias.assert_not_called()
    client.set_tag.assert_any_call(
        "challenger-run-20",
        "quality_gate_status",
        "REJECTED",
    )


def test_quality_gate_promotes_parsimonious_challenger_within_same_oot_tolerance():
    result = _run_quality_gate_with_same_oot_metrics(
        champion_mae=800.0,
        champion_nmae=7.00,
        challenger_mae=801.0,
        challenger_nmae=7.04,
        champion_features=9,
        challenger_features=8,
    )

    client = result["client"]

    client.set_registered_model_alias.assert_called_once()
    client.set_tag.assert_any_call(
        "challenger-run-20",
        "quality_gate_decision_reason",
        "PARSIMONY_WITHIN_NMAE_TOLERANCE_SAME_OOT",
    )


def test_quality_gate_rejects_parsimonious_challenger_outside_same_oot_tolerance():
    result = _run_quality_gate_with_same_oot_metrics(
        champion_mae=800.0,
        champion_nmae=7.00,
        challenger_mae=801.0,
        challenger_nmae=7.06,
        champion_features=9,
        challenger_features=8,
    )

    result["client"].set_registered_model_alias.assert_not_called()


def test_quality_gate_promotes_first_champion_without_same_oot_comparison():
    df_oot = _make_quality_gate_oot()

    challenger_run = _make_quality_gate_run(
        num_features=9,
        historical_mae=800.0,
        historical_nmae=6.8,
    )

    fake_client = MagicMock()
    fake_client.get_run.return_value = challenger_run
    fake_client.search_model_versions.return_value = [
        SimpleNamespace(version="1")
    ]
    fake_client.get_model_version_by_alias.side_effect = (
        training_flow.MlflowException(
            "champion ausente"
        )
    )

    with (
        patch.object(
            training_flow,
            "MlflowClient",
            return_value=fake_client,
        ),
        patch.object(
            training_flow,
            "evaluate_model_on_oot",
        ) as mock_evaluate,
        patch.object(
            training_flow.mlflow.sklearn,
            "load_model",
        ) as mock_load_model,
        patch.object(
            training_flow.requests,
            "post",
            return_value=MagicMock(),
        ),
    ):
        training_flow.evaluate_and_promote.fn(
            "challenger-run-1",
            df_oot,
        )

    mock_evaluate.assert_not_called()
    mock_load_model.assert_not_called()
    fake_client.set_registered_model_alias.assert_called_once_with(
        training_flow.MODEL_NAME,
        training_flow.MODEL_ALIAS,
        "1",
    )
    fake_client.set_tag.assert_any_call(
        "challenger-run-1",
        "quality_gate_decision_reason",
        "FIRST_CHAMPION",
    )

