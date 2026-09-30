from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_mlops.data.feature_utils import get_model_feature_columns
from energy_mlops.pipelines import monitoring_flow

''''
    Testes unitários para o pipeline de monitoramento.
    Foco em:
    - Preparação de features
    - Geração de relatório Evidently
    - Avaliação de drift de performance
    - Fluxo completo do pipeline de monitoramento

prepare_monitoring_features
├── mesmas 9 features do modelo
└── mantém ground truth/capacidade para performance

Data Drift
├── formato Evidently atual
├── fallback count/share
├── erro quando métrica não existe
└── geração do relatório

Performance Drift
├── nMAE atual sem degradação
├── nMAE atual com degradação
└── fallback para oot_nmae_pct_2026

Governança
├── Data Drift → retraining
├── Performance Drift → retraining
└── nenhum drift → não treina
'''

def make_raw_monitoring_dataframe() -> pd.DataFrame:
    """Dataset mínimo válido para engenharia de features."""

    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-01 00:00:00",
                    "2026-08-01 01:00:00",
                    "2026-08-01 02:00:00",
                ]
            ),
            "temperature_2m": [25.0, 25.5, 26.0],
            "wind_speed_100m": [8.0, 9.0, 10.0],
            "wind_direction_100m": [180.0, 190.0, 200.0],
            "wind_generation_mw": [5000.0, 5200.0, 5400.0],
        }
    )


def make_model_features_dataframe() -> pd.DataFrame:
    """Dataset contendo exatamente as features esperadas pelo modelo."""

    features = get_model_feature_columns()

    return pd.DataFrame(
        {
            feature: [1.0, 2.0, 3.0]
            for feature in features
        }
    )


def test_prepare_monitoring_features_uses_model_feature_contract():
    df_ref = make_raw_monitoring_dataframe()
    df_cur = make_raw_monitoring_dataframe()

    with patch(
        "energy_mlops.pipelines.monitoring_flow.get_run_logger",
        return_value=MagicMock(),
    ):
        X_ref, X_cur, df_cur_feat = (
            monitoring_flow.prepare_monitoring_features.fn(
                df_ref,
                df_cur,
            )
        )

    expected_features = get_model_feature_columns()

    assert X_ref.columns.tolist() == expected_features
    assert X_cur.columns.tolist() == expected_features

    assert len(X_ref.columns) == 9
    assert len(X_cur.columns) == 9

    assert X_ref.shape[0] == len(df_ref)
    assert X_cur.shape[0] == len(df_cur)

    assert "wind_generation_mw" in df_cur_feat.columns
    assert "capacidade_mw" in df_cur_feat.columns
    assert "target_fc" in df_cur_feat.columns


def test_extract_drift_result_evidently_format():
    snapshot = MagicMock()

    snapshot.dict.return_value = {
        "metrics": [
            {
                "result": {
                    "number_of_drifted_columns": 4,
                    "share_of_drifted_columns": 4 / 9,
                }
            }
        ]
    }

    drifted_count, share_drifted = (
        monitoring_flow.extract_drift_result(snapshot)
    )

    assert drifted_count == 4
    assert share_drifted == pytest.approx(4 / 9)


def test_extract_drift_result_count_share_fallback():
    snapshot = MagicMock()

    snapshot.dict.return_value = {
        "metrics": {
            "drift": {
                "count": 5,
                "share": 5 / 9,
            }
        }
    }

    drifted_count, share_drifted = (
        monitoring_flow.extract_drift_result(snapshot)
    )

    assert drifted_count == 5
    assert share_drifted == pytest.approx(5 / 9)


def test_extract_drift_result_raises_when_metrics_are_missing():
    snapshot = MagicMock()

    snapshot.dict.return_value = {
        "metrics": {
            "some_metric": {
                "value": 123,
            }
        }
    }

    with pytest.raises(
        RuntimeError,
        match="Não foi possível extrair os indicadores de Drift",
    ):
        monitoring_flow.extract_drift_result(snapshot)


def test_generate_evidently_report_detects_drift():
    features = get_model_feature_columns()

    X_ref = make_model_features_dataframe()
    X_cur = make_model_features_dataframe()

    fake_snapshot = MagicMock()

    fake_snapshot.dict.return_value = {
        "metrics": {
            "number_of_drifted_columns": 5,
            "share_of_drifted_columns": 5 / 9,
        }
    }

    def fake_save_html(path):
        with open(path, "w", encoding="utf-8") as file:
            file.write(
                "<html><body>"
                "Fake Evidently Report"
                "</body></html>"
            )

    fake_snapshot.save_html.side_effect = fake_save_html

    fake_report = MagicMock()
    fake_report.run.return_value = fake_snapshot

    fake_preset = MagicMock()

    with (
        patch(
            "energy_mlops.pipelines.monitoring_flow.get_run_logger",
            return_value=MagicMock(),
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.DataDriftPreset",
            return_value=fake_preset,
        ) as mock_preset,
        patch(
            "energy_mlops.pipelines.monitoring_flow.Report",
            return_value=fake_report,
        ) as mock_report,
    ):
        html_content, drift_detected, share_drifted = (
            monitoring_flow.generate_evidently_report.fn(
                X_ref,
                X_cur,
            )
        )

    mock_preset.assert_called_once_with(
        columns=features,
        drift_share=monitoring_flow.DRIFT_SHARE_THRESHOLD,
    )

    mock_report.assert_called_once_with(
        [fake_preset]
    )

    fake_report.run.assert_called_once()

    run_kwargs = fake_report.run.call_args.kwargs

    pd.testing.assert_frame_equal(
        run_kwargs["current_data"],
        X_cur,
    )

    pd.testing.assert_frame_equal(
        run_kwargs["reference_data"],
        X_ref,
    )

    assert "Fake Evidently Report" in html_content
    assert drift_detected is True
    assert share_drifted == pytest.approx(5 / 9)


def test_evaluate_performance_drift_without_degradation():
    X_cur = make_model_features_dataframe()

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [60.0, 60.0, 60.0],
            "capacidade_mw": [100.0, 100.0, 100.0],
        }
    )

    fake_model = MagicMock()
    fake_model.predict.return_value = [
        0.50,
        0.50,
        0.50,
    ]

    fake_client = MagicMock()

    fake_client.get_model_version_by_alias.return_value = (
        SimpleNamespace(
            run_id="champion-run-123",
        )
    )

    fake_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "oot_nmae_pct": 9.0,
            }
        )
    )

    with (
        patch(
            "energy_mlops.pipelines.monitoring_flow.get_run_logger",
            return_value=MagicMock(),
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.set_tracking_uri",
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.sklearn.load_model",
            return_value=fake_model,
        ) as mock_load_model,
    ):
        (
            drift_detected,
            baseline_nmae,
            current_nmae,
            delta_nmae,
        ) = monitoring_flow.evaluate_performance_drift.fn(
            X_cur,
            df_cur_feat,
        )

    fake_client.get_model_version_by_alias.assert_called_once_with(
        monitoring_flow.MODEL_NAME,
        monitoring_flow.MODEL_ALIAS,
    )

    fake_client.get_run.assert_called_once_with(
        "champion-run-123"
    )

    mock_load_model.assert_called_once_with(

            f"models:/{monitoring_flow.MODEL_NAME}"
            f"@{monitoring_flow.MODEL_ALIAS}"

    )

    fake_model.predict.assert_called_once()

    assert baseline_nmae == pytest.approx(9.0)
    assert current_nmae == pytest.approx(10.0)
    assert delta_nmae == pytest.approx(1.0)

    assert drift_detected is False


def test_evaluate_performance_drift_detects_degradation():
    X_cur = make_model_features_dataframe()

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [65.0, 65.0, 65.0],
            "capacidade_mw": [100.0, 100.0, 100.0],
        }
    )

    fake_model = MagicMock()
    fake_model.predict.return_value = [
        0.50,
        0.50,
        0.50,
    ]

    fake_client = MagicMock()

    fake_client.get_model_version_by_alias.return_value = (
        SimpleNamespace(
            run_id="champion-run-123",
        )
    )

    fake_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "oot_nmae_pct": 10.0,
            }
        )
    )

    with (
        patch(
            "energy_mlops.pipelines.monitoring_flow.get_run_logger",
            return_value=MagicMock(),
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.set_tracking_uri",
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.sklearn.load_model",
            return_value=fake_model,
        ),
    ):
        (
            drift_detected,
            baseline_nmae,
            current_nmae,
            delta_nmae,
        ) = monitoring_flow.evaluate_performance_drift.fn(
            X_cur,
            df_cur_feat,
        )

    assert baseline_nmae == pytest.approx(10.0)
    assert current_nmae == pytest.approx(15.0)
    assert delta_nmae == pytest.approx(5.0)

    assert drift_detected is True


def test_evaluate_performance_drift_uses_legacy_metric_fallback():
    X_cur = make_model_features_dataframe()

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [60.0, 60.0, 60.0],
            "capacidade_mw": [100.0, 100.0, 100.0],
        }
    )

    fake_model = MagicMock()
    fake_model.predict.return_value = [
        0.50,
        0.50,
        0.50,
    ]

    fake_client = MagicMock()

    fake_client.get_model_version_by_alias.return_value = (
        SimpleNamespace(
            run_id="legacy-champion-run",
        )
    )

    fake_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            metrics={
                "oot_nmae_pct_2026": 7.0,
            }
        )
    )

    with (
        patch(
            "energy_mlops.pipelines.monitoring_flow.get_run_logger",
            return_value=MagicMock(),
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.set_tracking_uri",
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "energy_mlops.pipelines.monitoring_flow.mlflow.sklearn.load_model",
            return_value=fake_model,
        ),
    ):
        (
            drift_detected,
            baseline_nmae,
            current_nmae,
            delta_nmae,
        ) = monitoring_flow.evaluate_performance_drift.fn(
            X_cur,
            df_cur_feat,
        )

    assert baseline_nmae == pytest.approx(7.0)
    assert current_nmae == pytest.approx(10.0)
    assert delta_nmae == pytest.approx(3.0)

    assert drift_detected is True


@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "continuous_training_pipeline"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "save_report_to_s3"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "evaluate_performance_drift"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "generate_evidently_report"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "prepare_monitoring_features"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "fetch_monitoring_data"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "get_run_logger"
)
def test_batch_monitoring_pipeline_triggers_training_on_data_drift(
    mock_logger,
    mock_fetch,
    mock_prepare,
    mock_generate_report,
    mock_performance,
    mock_save_report,
    mock_training,
):
    mock_logger.return_value = MagicMock()

    df_ref = pd.DataFrame({"raw": [1.0]})
    df_cur = pd.DataFrame({"raw": [2.0]})

    X_ref = pd.DataFrame({"feature": [1.0]})
    X_cur = pd.DataFrame({"feature": [2.0]})

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [50.0],
            "capacidade_mw": [100.0],
        }
    )

    mock_fetch.return_value = (
        df_ref,
        df_cur,
    )

    mock_prepare.return_value = (
        X_ref,
        X_cur,
        df_cur_feat,
    )

    mock_generate_report.return_value = (
        "<html>report</html>",
        True,
        0.60,
    )

    mock_performance.return_value = (
        False,
        8.0,
        8.5,
        0.5,
    )

    monitoring_flow.batch_monitoring_pipeline.fn(
        reference_path="reference.parquet",
        current_path="current.parquet",
        output_report_path="monitoring/report.html",
    )

    mock_fetch.assert_called_once_with(
        "reference.parquet",
        "current.parquet",
    )

    mock_prepare.assert_called_once_with(
        df_ref,
        df_cur,
    )

    mock_generate_report.assert_called_once_with(
        X_ref,
        X_cur,
    )

    mock_performance.assert_called_once_with(
        X_cur,
        df_cur_feat,
    )

    mock_save_report.assert_called_once_with(
        "<html>report</html>",
        "monitoring/report.html",
    )

    mock_training.assert_called_once_with()


@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "continuous_training_pipeline"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "save_report_to_s3"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "evaluate_performance_drift"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "generate_evidently_report"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "prepare_monitoring_features"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "fetch_monitoring_data"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "get_run_logger"
)
def test_batch_monitoring_pipeline_triggers_training_on_performance_drift(
    mock_logger,
    mock_fetch,
    mock_prepare,
    mock_generate_report,
    mock_performance,
    mock_save_report,
    mock_training,
):
    mock_logger.return_value = MagicMock()

    df_ref = pd.DataFrame({"raw": [1.0]})
    df_cur = pd.DataFrame({"raw": [2.0]})

    X_ref = pd.DataFrame({"feature": [1.0]})
    X_cur = pd.DataFrame({"feature": [2.0]})

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [50.0],
            "capacidade_mw": [100.0],
        }
    )

    mock_fetch.return_value = (
        df_ref,
        df_cur,
    )

    mock_prepare.return_value = (
        X_ref,
        X_cur,
        df_cur_feat,
    )

    mock_generate_report.return_value = (
        "<html>report</html>",
        False,
        0.20,
    )

    mock_performance.return_value = (
        True,
        8.0,
        12.0,
        4.0,
    )

    monitoring_flow.batch_monitoring_pipeline.fn(
        reference_path="reference.parquet",
        current_path="current.parquet",
        output_report_path="monitoring/report.html",
    )

    mock_generate_report.assert_called_once_with(
        X_ref,
        X_cur,
    )

    mock_performance.assert_called_once_with(
        X_cur,
        df_cur_feat,
    )

    mock_save_report.assert_called_once_with(
        "<html>report</html>",
        "monitoring/report.html",
    )

    mock_training.assert_called_once_with()


@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "continuous_training_pipeline"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "save_report_to_s3"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "evaluate_performance_drift"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "generate_evidently_report"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "prepare_monitoring_features"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "fetch_monitoring_data"
)
@patch(
    "energy_mlops.pipelines.monitoring_flow."
    "get_run_logger"
)
def test_batch_monitoring_pipeline_does_not_train_without_drift(
    mock_logger,
    mock_fetch,
    mock_prepare,
    mock_generate_report,
    mock_performance,
    mock_save_report,
    mock_training,
):
    mock_logger.return_value = MagicMock()

    df_ref = pd.DataFrame({"raw": [1.0]})
    df_cur = pd.DataFrame({"raw": [2.0]})

    X_ref = pd.DataFrame({"feature": [1.0]})
    X_cur = pd.DataFrame({"feature": [2.0]})

    df_cur_feat = pd.DataFrame(
        {
            "wind_generation_mw": [50.0],
            "capacidade_mw": [100.0],
        }
    )

    mock_fetch.return_value = (
        df_ref,
        df_cur,
    )

    mock_prepare.return_value = (
        X_ref,
        X_cur,
        df_cur_feat,
    )

    mock_generate_report.return_value = (
        "<html>report</html>",
        False,
        0.20,
    )

    mock_performance.return_value = (
        False,
        8.0,
        8.5,
        0.5,
    )

    monitoring_flow.batch_monitoring_pipeline.fn(
        reference_path="reference.parquet",
        current_path="current.parquet",
        output_report_path="monitoring/report.html",
    )

    mock_generate_report.assert_called_once_with(
        X_ref,
        X_cur,
    )

    mock_performance.assert_called_once_with(
        X_cur,
        df_cur_feat,
    )

    mock_save_report.assert_called_once_with(
        "<html>report</html>",
        "monitoring/report.html",
    )

    mock_training.assert_not_called()