from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import energy_mlops.models.train_stacking_ensemble as training_module
from energy_mlops.data.feature_utils import get_model_feature_columns


def make_model_dataframe(
    target_fc: list[float],
    wind_generation_mw: list[float],
) -> pd.DataFrame:
    """Dataset mínimo válido para o contrato do trainer."""

    return pd.DataFrame(
        {
            "temperature_2m": [20.0, 21.0, 22.0],
            "wind_speed_100m": [8.0, 9.0, 10.0],
            "wind_direction_100m": [180.0, 190.0, 200.0],
            "wind_temp_ratio": [0.40, 0.43, 0.45],
            "hour_sin": [0.0, 0.26, 0.50],
            "hour_cos": [1.0, 0.97, 0.87],
            "month_sin": [0.50, 0.50, 0.50],
            "month_cos": [0.86, 0.86, 0.86],
            "wind_speed_roll_mean_3h": [8.0, 8.5, 9.0],
            "target_fc": target_fc,
            "wind_generation_mw": wind_generation_mw,
            "capacidade_mw": [100.0, 100.0, 100.0],
        }
    )


def test_prepare_features_uses_model_feature_contract():
    df_train = make_model_dataframe(
        target_fc=[0.40, 0.50, 0.60],
        wind_generation_mw=[40.0, 50.0, 60.0],
    )

    df_test = make_model_dataframe(
        target_fc=[0.55, 0.60, 0.65],
        wind_generation_mw=[55.0, 60.0, 65.0],
    )

    (
        X_train,
        X_test,
        y_train_fc,
        y_test_fc,
        y_train_mw,
        y_test_mw,
        cap_train,
        cap_test,
    ) = training_module.prepare_features(
        df_train,
        df_test,
    )

    expected_features = get_model_feature_columns()

    assert X_train.columns.tolist() == expected_features
    assert X_test.columns.tolist() == expected_features

    assert len(X_train.columns) == 9
    assert len(X_test.columns) == 9

    pd.testing.assert_series_equal(
        y_train_fc,
        df_train["target_fc"],
    )

    pd.testing.assert_series_equal(
        y_test_fc,
        df_test["target_fc"],
    )

    pd.testing.assert_series_equal(
        y_train_mw,
        df_train["wind_generation_mw"],
    )

    pd.testing.assert_series_equal(
        y_test_mw,
        df_test["wind_generation_mw"],
    )

    pd.testing.assert_series_equal(
        cap_train,
        df_train["capacidade_mw"],
    )

    pd.testing.assert_series_equal(
        cap_test,
        df_test["capacidade_mw"],
    )


@pytest.mark.parametrize(
    "missing_group",
    [
        "lgb_params",
        "xgb_params",
        "rf_params",
    ],
)
def test_train_stacking_regressor_rejects_missing_parameter_group(
    missing_group: Literal['lgb_params', 'xgb_params', 'rf_params'],
):
    df_train = make_model_dataframe(
        target_fc=[0.40, 0.50, 0.60],
        wind_generation_mw=[40.0, 50.0, 60.0],
    )

    df_test = make_model_dataframe(
        target_fc=[0.55, 0.60, 0.65],
        wind_generation_mw=[55.0, 60.0, 65.0],
    )

    best_params = {
        "lgb_params": {
            "n_estimators": 200,
        },
        "xgb_params": {
            "n_estimators": 300,
        },
        "rf_params": {
            "n_estimators": 150,
        },
    }

    best_params[missing_group] = {}

    fake_client = MagicMock()
    fake_client.get_experiment_by_name.return_value = None

    fake_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="training-run-123",
        )
    )

    with (
        patch.object(
            training_module,
            "mlflow",
        ) as mock_mlflow,
        patch.object(
            training_module,
            "MlflowClient",
            return_value=fake_client,
        ),
    ):
        mock_mlflow.start_run.return_value.__enter__.return_value = (
            fake_run
        )

        mock_mlflow.data.from_pandas.side_effect = [
            "train-dataset",
            "test-dataset",
        ]

        with pytest.raises(
            ValueError,
            match="Parâmetros de otimização ausentes",
        ):
            training_module.train_stacking_regressor(
                df_train=df_train,
                df_test=df_test,
                train_file="train.parquet",
                test_file="test.parquet",
                best_params=best_params,
                optimization_run_id="optimization-run-123",
            )


def test_train_stacking_regressor_happy_path():
    df_train = make_model_dataframe(
        target_fc=[0.40, 0.50, 0.60],
        wind_generation_mw=[40.0, 50.0, 60.0],
    )

    df_test = make_model_dataframe(
        target_fc=[0.55, 0.60, 0.65],
        wind_generation_mw=[55.0, 60.0, 65.0],
    )

    best_params = {
        "lgb_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
        },
        "xgb_params": {
            "n_estimators": 300,
            "max_depth": 5,
        },
        "rf_params": {
            "n_estimators": 150,
            "max_depth": 10,
        },
    }

    fake_lgb = MagicMock()
    fake_xgb = MagicMock()
    fake_rf = MagicMock()

    fake_ensemble = MagicMock()

    fake_ensemble.final_estimator_ = SimpleNamespace(
        coef_=np.array(
            [
                0.20,
                0.30,
                0.50,
            ]
        )
    )

    fake_ensemble.named_estimators_ = {
        "lgbm": fake_lgb,
        "xgboost": fake_xgb,
        "rf": fake_rf,
    }

    train_predictions_fc = np.array(
        [
            0.40,
            0.50,
            0.60,
        ]
    )

    test_predictions_fc = np.array(
        [
            0.50,
            0.55,
            0.60,
        ]
    )

    fake_ensemble.predict.side_effect = [
        train_predictions_fc,
        test_predictions_fc,
    ]

    fake_client = MagicMock()
    fake_client.get_experiment_by_name.return_value = None

    fake_run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="training-run-123",
        )
    )

    fake_signature = MagicMock()

    fake_explanation = MagicMock()
    fake_explanation.values = np.full(
        (3, 9),
        0.01,
    )
    fake_explanation.__getitem__.return_value = MagicMock()

    fake_explainer = MagicMock()
    fake_explainer.return_value = fake_explanation

    fake_figure = MagicMock()
    fake_axis = MagicMock()

    with (
        patch.object(
            training_module,
            "mlflow",
        ) as mock_mlflow,
        patch.object(
            training_module,
            "MlflowClient",
            return_value=fake_client,
        ),
        patch.object(
            training_module.lgb,
            "LGBMRegressor",
            return_value=fake_lgb,
        ) as mock_lgb,
        patch.object(
            training_module,
            "XGBRegressor",
            return_value=fake_xgb,
        ) as mock_xgb,
        patch.object(
            training_module,
            "RandomForestRegressor",
            return_value=fake_rf,
        ) as mock_rf,
        patch.object(
            training_module,
            "LinearRegression",
            return_value=MagicMock(),
        ),
        patch.object(
            training_module,
            "TemporalStackingRegressor",
            return_value=fake_ensemble,
        ) as mock_stacking,
        patch.object(
            training_module,
            "infer_signature",
            return_value=fake_signature,
        ) as mock_infer_signature,
        patch.object(
            training_module,
            "shap",
        ) as mock_shap,
        patch.object(
            training_module,
            "plt",
        ) as mock_plt,
        patch.object(
            training_module.pd.DataFrame,
            "to_csv",
        ),
    ):
        mock_mlflow.start_run.return_value.__enter__.return_value = (
            fake_run
        )

        mock_mlflow.data.from_pandas.side_effect = [
            "train-dataset",
            "test-dataset",
        ]

        mock_shap.TreeExplainer.return_value = fake_explainer

        mock_plt.figure.return_value = fake_figure
        mock_plt.subplots.return_value = (
            fake_figure,
            fake_axis,
        )

        run_id, mae = (
            training_module.train_stacking_regressor(
                df_train=df_train,
                df_test=df_test,
                train_file="train.parquet",
                test_file="test.parquet",
                best_params=best_params,
                optimization_run_id="optimization-run-123",
            )
        )

    # --------------------------------------------------
    # Contrato de retorno
    # --------------------------------------------------
    assert run_id == "training-run-123"
    assert mae == pytest.approx(5.0)

    # --------------------------------------------------
    # Linhagem Optimization -> Training
    # --------------------------------------------------
    mock_mlflow.set_tag.assert_any_call(
        "optimization_run_id",
        "optimization-run-123",
    )

    mock_mlflow.set_tag.assert_any_call(
        "optimization_experiment",
        "wind_power_optimization_bahia",
    )

    # --------------------------------------------------
    # Features + estratégia temporal
    # --------------------------------------------------
    mock_mlflow.log_param.assert_any_call(
        "num_features",
        9,
    )

    mock_mlflow.log_param.assert_any_call(
        "stacking_cv_strategy",
        "TimeSeriesSplit",
    )

    mock_mlflow.log_param.assert_any_call(
        "stacking_cv_n_splits",
        5,
    )

    # --------------------------------------------------
    # Temporal Stacking
    # --------------------------------------------------
    stacking_kwargs = mock_stacking.call_args.kwargs

    assert (
        type(stacking_kwargs["cv"]).__name__
        == "TimeSeriesSplit"
    )

    assert stacking_kwargs["cv"].n_splits == 5

    # --------------------------------------------------
    # Hiperparâmetros recebidos do optimizer
    # --------------------------------------------------
    mock_lgb.assert_called_once_with(
        **best_params["lgb_params"]
    )

    mock_xgb.assert_called_once_with(
        **best_params["xgb_params"]
    )

    mock_rf.assert_called_once_with(
        **best_params["rf_params"]
    )

    mock_mlflow.log_dict.assert_any_call(
        best_params,
        "optimization_params_used.json",
    )

    # --------------------------------------------------
    # Linhagem dos datasets
    # --------------------------------------------------
    assert mock_mlflow.data.from_pandas.call_count == 2
    assert mock_mlflow.log_input.call_count == 2

    # --------------------------------------------------
    # Treinamento
    # --------------------------------------------------
    fake_ensemble.fit.assert_called_once()

    assert fake_ensemble.predict.call_count == 2

    stacking_kwargs = (
        mock_stacking.call_args.kwargs
    )

    assert (
        type(stacking_kwargs["cv"]).__name__
        == "TimeSeriesSplit"
    )

    assert (
        stacking_kwargs["cv"].n_splits
        == 5
    )

    # --------------------------------------------------
    # Métricas
    #
    # test:
    # real MW = [55, 60, 65]
    # pred MW = [50, 55, 60]
    #
    # MAE = 5 MW
    # nMAE = 5 / 100 * 100 = 5%
    # --------------------------------------------------
    logged_metrics = {
        call.args[0]: call.args[1]
        for call in mock_mlflow.log_metric.call_args_list
    }

    assert logged_metrics["oot_mae_mw"] == pytest.approx(
        5.0
    )

    assert logged_metrics["oot_nmae_pct"] == pytest.approx(
        5.0
    )

    assert logged_metrics[
        "oot_mae_fc_pct"
    ] == pytest.approx(
        5.0
    )

    assert "oot_r2_score" in logged_metrics

    # --------------------------------------------------
    # Signature MLflow
    # --------------------------------------------------
    mock_infer_signature.assert_called_once()

    signature_X, signature_predictions = (
        mock_infer_signature.call_args.args
    )

    expected_features = get_model_feature_columns()

    pd.testing.assert_frame_equal(
        signature_X,
        df_test[expected_features],
    )

    np.testing.assert_array_equal(
        signature_predictions,
        test_predictions_fc,
    )

    # --------------------------------------------------
    # Registry
    # --------------------------------------------------
    log_model_kwargs = (
        mock_mlflow.sklearn.log_model.call_args.kwargs
    )

    assert log_model_kwargs["sk_model"] is fake_ensemble
    assert log_model_kwargs["name"] == "model"

    assert (
        log_model_kwargs["registered_model_name"]
        == "ensemble_lgb_xgb_rf_bahia"
    )

    assert log_model_kwargs["signature"] is fake_signature

    pd.testing.assert_frame_equal(
        log_model_kwargs["input_example"],
        df_test[expected_features].head(1),
    )

    # --------------------------------------------------
    # SHAP isolado do teste unitário
    # --------------------------------------------------
    mock_shap.TreeExplainer.assert_called_once_with(
        fake_lgb
    )

def test_train_stacking_regressor_rejects_missing_best_params():
    """
    O pipeline pode operar sem optimizer, mas o trainer de Temporal
    Stacking exige parâmetros produzidos pela otimização.
    """

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="best_params",
    ):
        training_module.train_stacking_regressor(
            df_train=df_train,
            df_test=df_test,
            train_file="train.parquet",
            test_file="test.parquet",
            best_params=None,
            optimization_run_id="optimization-run-123",
        )


def test_train_stacking_regressor_rejects_missing_optimization_run_id():
    """
    O Temporal Stacking também exige a Optimization Run para manter
    a linhagem entre otimização e treinamento.
    """

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="optimization_run_id",
    ):
        training_module.train_stacking_regressor(
            df_train=df_train,
            df_test=df_test,
            train_file="train.parquet",
            test_file="test.parquet",
            best_params={},
            optimization_run_id=None,
        )