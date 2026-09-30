# tests/pipelines/test_training_flow.py
from unittest.mock import patch

import pandas as pd

from energy_mlops.pipelines.training_flow import (
    continuous_training_pipeline,
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