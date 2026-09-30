from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_mlops.data.feature_utils import (
    get_model_feature_columns,
)
from energy_mlops.models.train_examples import (
    _calculate_metrics,
    _prepare_training_data,
    _validate_optimization_context,
    train_ridge_regression,
)


def _make_dataset(
    target_fc: list[float],
    capacidade_mw: list[float],
) -> pd.DataFrame:
    """
    Cria um dataset mínimo respeitando o contrato atual
    de features e targets do projeto.
    """

    n = len(target_fc)

    data = {
        "temperature_2m": np.linspace(
            20.0,
            25.0,
            n,
        ),
        "wind_speed_100m": np.linspace(
            8.0,
            12.0,
            n,
        ),
        "wind_direction_100m": np.linspace(
            90.0,
            120.0,
            n,
        ),
        "wind_temp_ratio": np.linspace(
            0.4,
            0.5,
            n,
        ),
        "hour_sin": np.linspace(
            0.0,
            0.5,
            n,
        ),
        "hour_cos": np.linspace(
            1.0,
            0.5,
            n,
        ),
        "month_sin": np.full(
            n,
            0.5,
        ),
        "month_cos": np.full(
            n,
            0.8,
        ),
        "wind_speed_roll_mean_3h": np.linspace(
            7.5,
            11.5,
            n,
        ),
        "target_fc": target_fc,
        "capacidade_mw": capacidade_mw,
    }

    df = pd.DataFrame(data)

    df["wind_generation_mw"] = (
        df["target_fc"]
        * df["capacidade_mw"]
    )

    # Coluna propositalmente fora do contrato.
    # O trainer não deve utilizá-la como feature.
    df["unused_feature"] = 999.0

    return df


def test_optimization_context_accepts_valid_states():
    """
    O contrato aceita tanto um trainer sem optimizer
    quanto um trainer vinculado a uma Optimization Run.
    """

    _validate_optimization_context(
        best_params=None,
        optimization_run_id=None,
    )

    _validate_optimization_context(
        best_params={
            "ridge_params": {
                "alpha": 0.25,
            }
        },
        optimization_run_id="optimization-run-123",
    )


def test_optimization_context_rejects_partial_state():
    """
    Parâmetros e lineage da otimização devem existir juntos.
    """

    with pytest.raises(
        ValueError,
        match="Contexto de otimização inconsistente",
    ):
        _validate_optimization_context(
            best_params={
                "ridge_params": {
                    "alpha": 0.25,
                }
            },
            optimization_run_id=None,
        )

    with pytest.raises(
        ValueError,
        match="Contexto de otimização inconsistente",
    ):
        _validate_optimization_context(
            best_params=None,
            optimization_run_id="optimization-run-123",
        )


def test_prepare_training_data_uses_model_feature_contract():
    """
    Os exemplos devem consumir exatamente o contrato
    central de features, sem descoberta manual de colunas.
    """

    df_train = _make_dataset(
        target_fc=[
            0.10,
            0.20,
            0.30,
        ],
        capacidade_mw=[
            100.0,
            100.0,
            100.0,
        ],
    )

    df_test = _make_dataset(
        target_fc=[
            0.40,
            0.50,
        ],
        capacidade_mw=[
            100.0,
            100.0,
        ],
    )

    (
        X_train,
        X_test,
        *_,
    ) = _prepare_training_data(
        df_train,
        df_test,
    )

    expected_features = (
        get_model_feature_columns()
    )

    assert list(
        X_train.columns
    ) == expected_features

    assert list(
        X_test.columns
    ) == expected_features

    assert "unused_feature" not in X_train.columns
    assert "target_fc" not in X_train.columns
    assert "capacidade_mw" not in X_train.columns


def test_calculate_metrics_converts_capacity_factor_to_mw():
    """
    O MAE em MW deve ser calculado depois da conversão:

        predicted_fc * capacidade_mw

    e não diretamente sobre target_fc.
    """

    metrics = _calculate_metrics(
        y_train_fc=pd.Series(
            [0.10, 0.20]
        ),
        y_test_fc=pd.Series(
            [0.20, 0.40]
        ),
        y_train_mw=pd.Series(
            [10.0, 40.0]
        ),
        y_test_mw=pd.Series(
            [20.0, 80.0]
        ),
        cap_train=pd.Series(
            [100.0, 200.0]
        ),
        cap_test=pd.Series(
            [100.0, 200.0]
        ),
        train_preds_fc=np.array(
            [0.10, 0.20]
        ),
        test_preds_fc=np.array(
            [0.10, 0.50]
        ),
    )

    # Predição OOT em MW:
    #
    # 0.10 * 100 = 10 MW
    # 0.50 * 200 = 100 MW
    #
    # Truth:
    # 20 MW e 80 MW
    #
    # MAE = (10 + 20) / 2 = 15 MW

    assert metrics[
        "oot_mae_mw"
    ] == pytest.approx(
        15.0
    )

    assert metrics[
        "oot_nmae_pct"
    ] == pytest.approx(
        7.5
    )

    assert metrics[
        "oot_mae_fc_pct"
    ] == pytest.approx(
        10.0
    )

    assert metrics[
        "oot_r2_score"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "train_mae_mw"
    ] == pytest.approx(
        0.0
    )


def test_ridge_trainer_runs_without_optimizer():
    """
    Demonstra o caminho principal do exemplo:

        optimizer_name="none"
            ↓
        best_params=None
        optimization_run_id=None
            ↓
        Ridge treina normalmente
            ↓
        retorna (run_id, oot_mae_mw)
    """

    df_train = _make_dataset(
        target_fc=[
            0.10,
            0.20,
            0.30,
            0.40,
        ],
        capacidade_mw=[
            100.0,
            100.0,
            100.0,
            100.0,
        ],
    )

    df_test = _make_dataset(
        target_fc=[
            0.20,
            0.40,
            0.60,
        ],
        capacidade_mw=[
            100.0,
            100.0,
            100.0,
        ],
    )

    fake_model = MagicMock()

    fake_model.predict.side_effect = [
        np.array(
            [0.10, 0.20, 0.30, 0.40]
        ),
        np.array(
            [0.10, 0.50, 0.60]
        ),
    ]

    fake_run = MagicMock()
    fake_run.info.run_id = "ridge-run-123"

    with (
        patch(
            "energy_mlops.models.train_examples.Ridge",
            return_value=fake_model,
        ) as mock_ridge,
        patch(
            "energy_mlops.models.train_examples."
            "_prepare_mlflow_experiment",
        ),
        patch(
            "energy_mlops.models.train_examples."
            "_log_dataset_lineage",
        ) as mock_lineage,
        patch(
            "energy_mlops.models.train_examples."
            "_log_common_metadata",
        ) as mock_common_metadata,
        patch(
            "energy_mlops.models.train_examples."
            "infer_signature",
            return_value="fake-signature",
        ),
        patch(
            "energy_mlops.models.train_examples.mlflow",
        ) as mock_mlflow,
    ):
        (
            mock_mlflow
            .start_run
            .return_value
            .__enter__
            .return_value
        ) = fake_run

        run_id, oot_mae_mw = (
            train_ridge_regression(
                df_train=df_train,
                df_test=df_test,
                train_file="train.parquet",
                test_file="test.parquet",
                best_params=None,
                optimization_run_id=None,
            )
        )

    assert run_id == "ridge-run-123"

    # Erros OOT:
    #
    # Truth MW:      20, 40, 60
    # Prediction MW: 10, 50, 60
    #
    # MAE = (10 + 10 + 0) / 3

    assert oot_mae_mw == pytest.approx(
        20.0 / 3.0
    )

    mock_ridge.assert_called_once_with(
        alpha=1.0
    )

    fake_model.fit.assert_called_once()

    mock_lineage.assert_called_once()

    metadata = (
        mock_common_metadata
        .call_args
        .kwargs
    )

    assert metadata[
        "model_type"
    ] == "Ridge"

    assert metadata[
        "num_features"
    ] == 9

    assert metadata[
        "optimization_run_id"
    ] is None

    metrics = metadata["metrics"]

    assert set(metrics) == {
        "train_mae_mw",
        "train_nmae_pct",
        "train_mae_fc_pct",
        "oot_mae_mw",
        "oot_nmae_pct",
        "oot_mae_fc_pct",
        "oot_r2_score",
    }

    assert metrics[
        "oot_mae_mw"
    ] == pytest.approx(
        20.0 / 3.0
    )

    mock_mlflow.sklearn.log_model.assert_called_once()

    log_model_kwargs = (
        mock_mlflow
        .sklearn
        .log_model
        .call_args
        .kwargs
    )

    assert (
        log_model_kwargs[
            "registered_model_name"
        ]
        == "ensemble_lgb_xgb_rf_bahia"
    )