from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import energy_mlops.models.optimize_stacking_ensemble as optimization_module
from energy_mlops.models.optimize_stacking_ensemble import (
    objective,
    run_optimization,
)


def test_run_optimization_returns_optimization_result():
    '''
    run_optimization()
        │
        ├── optuna.create_study() → mock
        │
        ├── study.optimize()      → mock
        │
        ├── mlflow.data.from_pandas() → mock
        │
        ├── mlflow.log_input()    → mock
        │
        └── retorna OptimizationResult
    '''
    df_train = pd.DataFrame(
        {
            "temperature_2m": [20.0, 21.0, 22.0],
            "wind_speed_100m": [8.0, 9.0, 10.0],
            "wind_direction_100m": [180.0, 190.0, 200.0],
            "wind_temp_ratio": [0.4, 0.43, 0.45],
            "hour_sin": [0.0, 0.26, 0.5],
            "hour_cos": [1.0, 0.97, 0.87],
            "month_sin": [0.5, 0.5, 0.5],
            "month_cos": [0.86, 0.86, 0.86],
            "wind_speed_roll_mean_3h": [8.0, 8.5, 9.0],
            "target_fc": [0.30, 0.35, 0.40],
            "capacidade_mw": [100.0, 200.0, 400.0],
            "wind_generation_mw": [30.0, 70.0, 160.0]
        }
    )

    best_params_raw = {
        "lgb_n_estimators": 200,
        "lgb_learning_rate": 0.05,
        "lgb_num_leaves": 50,
        "lgb_max_depth": 6,
        "xgb_n_estimators": 300,
        "xgb_learning_rate": 0.03,
        "xgb_max_depth": 5,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.9,
        "rf_n_estimators": 150,
        "rf_max_depth": 10,
        "rf_min_samples_split": 5,
    }

    fake_trial = SimpleNamespace(
        params=best_params_raw,
        value=5.833333333333333,
        user_attrs={
            "cv_nmae_std": 0.0,
            "cv_mae_fc_pct_mean": 10.0,
            "cv_r2_mean": -149.0,
        },
    )

    fake_study = MagicMock()
    fake_study.best_trial = fake_trial

    fake_run = MagicMock()
    fake_run.info.run_id = "fake-optimization-run-123"

    fake_objective_trial = MagicMock()

    with (
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "optuna.create_study",
            return_value=fake_study,
        ) as mock_create_study,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "objective",
            return_value=5.833333333333333,
        ) as mock_objective,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.start_run",
        ) as mock_start_run,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.set_tracking_uri",
        ),
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.set_experiment",
        ),
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.data.from_pandas",
            return_value=MagicMock(),
        ) as mock_from_pandas,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.log_input",
        ) as mock_log_input,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.log_params",
        ),
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.log_param",
        ) as mock_log_param,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.log_metric",
        ) as mock_log_metric,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "mlflow.log_dict",
        ),
    ):
        mock_start_run.return_value.__enter__.return_value = (
            fake_run
        )

        def execute_fake_optimization(
            objective_callback,
            n_trials,
        ):
            objective_callback(
                fake_objective_trial
            )

        fake_study.optimize.side_effect = (
            execute_fake_optimization
        )

        result = run_optimization(
            df_train=df_train,
            train_file="dataset_train.parquet",
            n_trials=10,
            n_splits=3,
            stacking_n_splits=5,
        )

    # --------------------------------------------------
    # Estudo Optuna
    # --------------------------------------------------
    mock_create_study.assert_called_once_with(
        direction="minimize",
        study_name="Ensemble_Optimization_LGB_XGB_RF",
    )

    fake_study.optimize.assert_called_once()

    # --------------------------------------------------
    # run_optimization -> objective
    # --------------------------------------------------
    mock_objective.assert_called_once()

    objective_call = mock_objective.call_args

    assert (
        objective_call.args[0]
        is fake_objective_trial
    )

    X_received = objective_call.args[1]
    y_fc_received = objective_call.args[2]
    y_mw_received = objective_call.args[3]
    capacity_received = objective_call.args[4]

    assert X_received.columns.tolist() == [
        "temperature_2m",
        "wind_speed_100m",
        "wind_direction_100m",
        "wind_temp_ratio",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "wind_speed_roll_mean_3h",
    ]

    pd.testing.assert_series_equal(
        y_fc_received,
        df_train["target_fc"],
    )

    pd.testing.assert_series_equal(
        y_mw_received,
        df_train["wind_generation_mw"],
    )

    pd.testing.assert_series_equal(
        capacity_received,
        df_train["capacidade_mw"],
    )

    assert (
        objective_call.kwargs["n_splits"]
        == 3
    )

    assert (
        objective_call.kwargs["stacking_n_splits"]
        == 5
    )

    assert (
        objective_call.kwargs["enabled_estimators"]
        == (
            "lgbm",
            "xgboost",
            "rf",
        )
    )

    # --------------------------------------------------
    # Dataset lineage
    # --------------------------------------------------
    mock_from_pandas.assert_called_once()
    mock_log_input.assert_called_once()

    # --------------------------------------------------
    # Governança da otimização
    # --------------------------------------------------
    mock_log_param.assert_any_call(
        "outer_cv_n_splits",
        3,
    )

    mock_log_param.assert_any_call(
        "stacking_cv_n_splits",
        5,
    )

    mock_log_param.assert_any_call(
        "num_features",
        9,
    )

    mock_log_param.assert_any_call(
        "optimization_metric",
        "nmae_pct",
    )

    # --------------------------------------------------
    # Métricas da melhor trial
    # --------------------------------------------------
    mock_log_metric.assert_any_call(
        "best_cv_nmae_pct_mean",
        pytest.approx(5.833333333333333),
    )

    mock_log_metric.assert_any_call(
        "best_cv_nmae_pct_std",
        pytest.approx(0.0),
    )

    mock_log_metric.assert_any_call(
        "best_cv_mae_fc_pct_mean",
        pytest.approx(10.0),
    )

    mock_log_metric.assert_any_call(
        "best_cv_r2_mean",
        pytest.approx(-149.0),
    )

    # --------------------------------------------------
    # Contrato OptimizationResult
    # --------------------------------------------------
    assert (
        result["optimization_run_id"]
        == "fake-optimization-run-123"
    )

    assert result["best_params"] == {
        "lgb_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 50,
            "max_depth": 6,
            "objective": "regression",
            "metric": "mae",
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        },
        "xgb_params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "objective": "reg:absoluteerror",
            "random_state": 42,
            "n_jobs": -1,
        },
        "rf_params": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
    }


def test_objective_uses_temporal_stacking_without_future_leakage():
    X = pd.DataFrame(
        {
            "feature_1": np.arange(
                12,
                dtype=float,
            ),
            "feature_2": (
                np.arange(
                    12,
                    dtype=float,
                )
                * 2
            ),
        }
    )

    # Target em fator de capacidade.
    y = pd.Series(
        [
            0.40,
            0.41,
            0.42,
            0.43,
            0.44,
            0.45,
            0.46,
            0.47,
            0.48,
            0.49,
            0.50,
            0.51,
        ]
    )

    # Capacidade varia ao longo do tempo.
    #
    # Isso é proposital: permite distinguir
    # MAE_FC de nMAE em MW.
    capacity = pd.Series(
        [
            50.0,
            50.0,
            50.0,
            100.0,
            200.0,
            400.0,
            200.0,
            400.0,
            800.0,
            300.0,
            600.0,
            1200.0,
        ]
    )

    # Ground truth operacional em MW.
    #
    # geração_mw = target_fc * capacidade_mw
    y_mw = y * capacity

    # --------------------------------------------------
    # Trial Optuna falso
    # --------------------------------------------------
    int_params = {
        "lgb_n_estimators": 200,
        "lgb_num_leaves": 50,
        "lgb_max_depth": 6,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 5,
        "rf_n_estimators": 150,
        "rf_max_depth": 10,
        "rf_min_samples_split": 5,
    }

    float_params = {
        "lgb_learning_rate": 0.05,
        "xgb_learning_rate": 0.03,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.9,
    }

    fake_trial = MagicMock()

    fake_trial.suggest_int.side_effect = (
        lambda name, *args, **kwargs: int_params[name]
    )

    fake_trial.suggest_float.side_effect = (
        lambda name, *args, **kwargs: float_params[name]
    )

    # --------------------------------------------------
    # Modelos base falsos
    # --------------------------------------------------
    fake_lgb = MagicMock()
    fake_xgb = MagicMock()
    fake_rf = MagicMock()

    fake_meta_learner = MagicMock()

    # --------------------------------------------------
    # Previsões do Temporal Stacking
    #
    # TimeSeriesSplit(3):
    #
    # Fold 1 -> val índices 3,4,5
    # y = [0.43, 0.44, 0.45]
    #
    # Fold 2 -> val índices 6,7,8
    # y = [0.46, 0.47, 0.48]
    #
    # Fold 3 -> val índices 9,10,11
    # y = [0.49, 0.50, 0.51]
    #
    # Em todos os casos introduzimos erro FC = 0.10.
    # --------------------------------------------------
    fake_ensemble = MagicMock()

    fake_ensemble.predict.side_effect = [
        np.array(
            [
                0.33,
                0.34,
                0.35,
            ]
        ),
        np.array(
            [
                0.36,
                0.37,
                0.38,
            ]
        ),
        np.array(
            [
                0.39,
                0.40,
                0.41,
            ]
        ),
    ]

    with (
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "lgb.LGBMRegressor",
            return_value=fake_lgb,
        ) as mock_lgb,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "XGBRegressor",
            return_value=fake_xgb,
        ) as mock_xgb,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "RandomForestRegressor",
            return_value=fake_rf,
        ) as mock_rf,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "LinearRegression",
            return_value=fake_meta_learner,
        ) as mock_meta,
        patch(
            "energy_mlops.models.optimize_stacking_ensemble."
            "TemporalStackingRegressor",
            return_value=fake_ensemble,
        ) as mock_stacking,
    ):
        result = objective(
            trial=fake_trial,
            X=X,
            y_fc=y,
            y_mw=y_mw,
            capacity=capacity,
            n_splits=3,
            stacking_n_splits=3,
        )

    # --------------------------------------------------
    # Validação matemática do nMAE
    #
    # Em cada fold:
    #
    # erro FC = 0.10
    #
    # Fold 1:
    # cap = [100, 200, 400]
    #
    # erros MW = [10, 20, 40]
    #
    # MAE MW = (10 + 20 + 40) / 3
    #        = 23.3333...
    #
    # nMAE = 23.3333 / 400 * 100
    #      = 5.8333%
    #
    # Os folds seguintes possuem a mesma proporção.
    # --------------------------------------------------
    expected_nmae_pct = 5.833333333333333

    assert result == pytest.approx(
        expected_nmae_pct
    )

    # --------------------------------------------------
    # Métricas auxiliares registradas no Trial
    # --------------------------------------------------
    trial_attrs = {
        call.args[0]: call.args[1]
        for call in fake_trial.set_user_attr.call_args_list
    }

    assert trial_attrs[
        "cv_nmae_std"
    ] == pytest.approx(
        0.0
    )

    # MAE em FC = 0.10 -> 10%
    assert trial_attrs[
        "cv_mae_fc_pct_mean"
    ] == pytest.approx(
        10.0
    )

    assert trial_attrs[
        "cv_r2_mean"
    ] == pytest.approx(
        -149.0
    )

    # --------------------------------------------------
    # Outer TimeSeriesSplit
    # --------------------------------------------------
    assert fake_ensemble.fit.call_count == 3
    assert fake_ensemble.predict.call_count == 3

    for fit_call, predict_call in zip(
        fake_ensemble.fit.call_args_list,
        fake_ensemble.predict.call_args_list,
    ):
        X_train_fold = fit_call.args[0]
        X_val_fold = predict_call.args[0]

        # Nenhuma observação futura pode participar
        # do treino que gera aquela validação.
        assert (
            X_train_fold.index.max()
            < X_val_fold.index.min()
        )

    # --------------------------------------------------
    # Temporal Stacking usado nos 3 folds externos
    # --------------------------------------------------
    assert mock_stacking.call_count == 3

    inner_cv_splits = [
        call.kwargs["cv"].n_splits
        for call in mock_stacking.call_args_list
    ]

    assert inner_cv_splits == [
        2,
        3,
        3,
    ]

    # --------------------------------------------------
    # Arquitetura
    # --------------------------------------------------
    for call in mock_stacking.call_args_list:
        estimators = call.kwargs[
            "estimators"
        ]

        assert estimators == [
            ("lgbm", fake_lgb),
            ("xgboost", fake_xgb),
            ("rf", fake_rf),
        ]

        assert (
            call.kwargs["final_estimator"]
            is fake_meta_learner
        )

    assert mock_lgb.call_count == 3
    assert mock_xgb.call_count == 3
    assert mock_rf.call_count == 3
    assert mock_meta.call_count == 3

    # --------------------------------------------------
    # Os modelos base NÃO são treinados diretamente
    # pelo objective().
    #
    # Quem controla seus fits é o Temporal Stacking.
    # Isso protege contra o retorno acidental à antiga
    # média simples LGB + XGB + RF.
    # --------------------------------------------------
    fake_lgb.fit.assert_not_called()
    fake_lgb.predict.assert_not_called()

    fake_xgb.fit.assert_not_called()
    fake_xgb.predict.assert_not_called()

    fake_rf.fit.assert_not_called()
    fake_rf.predict.assert_not_called()


def test_suggest_estimator_params_excludes_xgboost_for_lgbm_rf():
    trial = MagicMock()

    trial.suggest_int.side_effect = [
        200,  # lgb_n_estimators
        50,   # lgb_num_leaves
        6,    # lgb_max_depth
        150,  # rf_n_estimators
        10,   # rf_max_depth
        5,    # rf_min_samples_split
    ]

    trial.suggest_float.side_effect = [
        0.05,  # lgb_learning_rate
    ]

    params = optimization_module._suggest_estimator_params(
        trial,
        enabled_estimators=(
            "lgbm",
            "rf",
        ),
    )

    assert set(params) == {
        "lgbm",
        "rf",
    }

    suggested_parameter_names = {
        call.args[0]
        for call in (
            trial.suggest_int.call_args_list
            + trial.suggest_float.call_args_list
        )
    }

    assert {
        "lgb_n_estimators",
        "lgb_learning_rate",
        "lgb_num_leaves",
        "lgb_max_depth",
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_split",
    } <= suggested_parameter_names

    assert not any(
        name.startswith("xgb_")
        for name in suggested_parameter_names
    )


def test_reconstruct_best_params_for_lgbm_rf_excludes_xgboost():
    best_params_raw = {
        "lgb_n_estimators": 200,
        "lgb_learning_rate": 0.05,
        "lgb_num_leaves": 50,
        "lgb_max_depth": 6,
        "rf_n_estimators": 150,
        "rf_max_depth": 10,
        "rf_min_samples_split": 5,
    }

    result = optimization_module._reconstruct_best_params(
        best_params_raw,
        enabled_estimators=(
            "lgbm",
            "rf",
        ),
    )

    assert set(result) == {
        "lgb_params",
        "rf_params",
    }

    assert "xgb_params" not in result

    assert result["lgb_params"] == {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 50,
        "max_depth": 6,
        "objective": "regression",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    assert result["rf_params"] == {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }