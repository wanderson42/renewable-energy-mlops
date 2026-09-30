import os
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import s3fs
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.metrics import mean_absolute_error

from energy_mlops.config import settings
from energy_mlops.data.build_features import generate_wind_and_time_features
from energy_mlops.data.feature_utils import select_model_features
from energy_mlops.pipelines.training_flow import continuous_training_pipeline

'''
                 Monitoring
                     │
          ┌──────────┴──────────┐
          │                     │
      DATA DRIFT         PERFORMANCE DRIFT
          │                     │
   Evidently Share        Champion MLflow
          │                     │
      >= 50%?           Current nMAE
          │                     │
          │            Delta >= 2.0 p.p.?
          │                     │
          └──────────┬──────────┘
                     │
                  OR rule
                     │
             Continuous Training
'''

# Configurações de Governança
DRIFT_SHARE_THRESHOLD = 0.50

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"

# Unidade: pontos percentuais de nMAE.
PERFORMANCE_NMAE_DELTA_PP_THRESHOLD = 2.0

# Credenciais
os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT


@task(name="Extrair Dados de Monitoramento", retries=2)
def fetch_monitoring_data(reference_path: str, current_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai os datasets Reference e Current do Data Lake (RustFS/S3)."""
    logger = get_run_logger()
    fs = s3fs.S3FileSystem()

    logger.info(f"Lendo Reference: {reference_path}")
    with fs.open(reference_path, "rb") as f:
        df_ref = pd.read_parquet(f)

    logger.info(f"Lendo Current: {current_path}")
    with fs.open(current_path, "rb") as f:
        df_cur = pd.read_parquet(f)

    return df_ref, df_cur


@task(name="Preparar Features para Monitoramento")
def prepare_monitoring_features(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica a mesma engenharia e o mesmo contrato de features do modelo."""

    logger = get_run_logger()

    logger.info(
        "Aplicando engenharia de features ao dataset Reference..."
    )
    df_ref_feat = generate_wind_and_time_features(df_ref)

    logger.info(
        "Aplicando engenharia de features ao dataset Current..."
    )
    df_cur_feat = generate_wind_and_time_features(df_cur)

    logger.info(
        "Validando e selecionando features do modelo..."
    )
    X_ref = select_model_features(df_ref_feat)
    X_cur = select_model_features(df_cur_feat)

    logger.info(
        f"Features monitoradas ({len(X_ref.columns)}): "
        f"{X_ref.columns.tolist()}"
    )

    return X_ref, X_cur, df_cur_feat


def extract_drift_result(snapshot: Any) -> tuple[int, float]:
    """Extrai contagem e proporção de colunas com drift do Evidently."""

    report_dict = snapshot.dict()

    def find_metrics(obj: Any) -> tuple[int, float] | None:
        if isinstance(obj, dict):
            if (
                "number_of_drifted_columns" in obj
                and "share_of_drifted_columns" in obj
            ):
                return (
                    int(obj["number_of_drifted_columns"]),
                    float(obj["share_of_drifted_columns"]),
                )

            if (
                "count" in obj
                and "share" in obj
                and isinstance(obj["count"], (int, float))
            ):
                return int(obj["count"]), float(obj["share"])

            for value in obj.values():
                result = find_metrics(value)
                if result is not None:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = find_metrics(item)
                if result is not None:
                    return result

        return None

    result = find_metrics(report_dict)

    if result is None:
        raise RuntimeError(
            "Não foi possível extrair os indicadores de Drift "
            "do relatório do Evidently."
        )

    return result


@task(name="Gerar Relatório Evidently & Avaliar Drift")
def generate_evidently_report(
    X_ref: pd.DataFrame,
    X_cur: pd.DataFrame,
) -> tuple[str, bool, float]:
    """Gera o relatório de Data Drift das features utilizadas pelo modelo."""

    logger = get_run_logger()

    features = X_ref.columns.tolist()

    report = Report(
        [
            DataDriftPreset(
                columns=features,
                drift_share=DRIFT_SHARE_THRESHOLD,
            )
        ]
    )

    snapshot = report.run(
        current_data=X_cur,
        reference_data=X_ref,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".html",
        delete=False,
    ) as tmp:
        tmp_path = tmp.name

    try:
        snapshot.save_html(tmp_path)

        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        os.unlink(tmp_path)

    drifted_count, share_drifted = extract_drift_result(snapshot)

    drift_detected = share_drifted >= DRIFT_SHARE_THRESHOLD

    logger.info(
        f"Drift Analysis: "
        f"{drifted_count}/{len(features)} features afetadas "
        f"({share_drifted:.1%})"
    )

    return html_content, drift_detected, share_drifted


@task(name="Salvar Relatório no RustFS")
def save_report_to_s3(html_content: str, destination_path: str):
    """Persiste o relatório HTML no Data Lake."""
    fs = s3fs.S3FileSystem()
    with fs.open(destination_path, "w", encoding="utf-8") as f:
        f.write(html_content)


@task(name="Avaliar Performance Drift")
def evaluate_performance_drift(
    X_cur: pd.DataFrame,
    df_cur_feat: pd.DataFrame,
) -> tuple[bool, float, float, float]:
    """
    Compara a performance atual do Champion com sua baseline OOT.

    Returns:
        performance_drift_detected
        baseline_nmae_pct
        current_nmae_pct
        nmae_delta_pp
    """

    logger = get_run_logger()

    required_columns = {
        "wind_generation_mw",
        "capacidade_mw",
    }

    missing_columns = required_columns.difference(
        df_cur_feat.columns
    )

    if missing_columns:
        raise ValueError(
            "Performance Drift exige ground truth e capacidade. "
            f"Colunas ausentes: {sorted(missing_columns)}"
        )

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    client = MlflowClient()

    champion_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    champion_run = client.get_run(
        champion_version.run_id
    )

    metrics = champion_run.data.metrics

    baseline_nmae_pct = metrics.get(
        "oot_nmae_pct"
    )

    # Compatibilidade com runs antigos.
    if baseline_nmae_pct is None:
        baseline_nmae_pct = metrics.get(
            "oot_nmae_pct_2026"
        )

    if baseline_nmae_pct is None:
        raise RuntimeError(
            "Champion não possui a métrica de baseline "
            "'oot_nmae_pct' nem 'oot_nmae_pct_2026'."
        )

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    champion_model = mlflow.sklearn.load_model(
        model_uri
    )

    predictions_fc = champion_model.predict(
        X_cur
    )

    capacidade_mw = (
        df_cur_feat["capacidade_mw"]
        .to_numpy()
    )

    actual_mw = (
        df_cur_feat["wind_generation_mw"]
        .to_numpy()
    )

    predictions_mw = (
        np.asarray(predictions_fc)
        * capacidade_mw
    )

    current_mae_mw = mean_absolute_error(
        actual_mw,
        predictions_mw,
    )

    current_nmae_pct = (
        current_mae_mw
        / float(np.max(capacidade_mw))
    ) * 100

    nmae_delta_pp = (
        current_nmae_pct
        - float(baseline_nmae_pct)
    )

    performance_drift_detected = (
        nmae_delta_pp
        >= PERFORMANCE_NMAE_DELTA_PP_THRESHOLD
    )

    logger.info(
        "Performance Monitoring | "
        f"Champion baseline nMAE: "
        f"{baseline_nmae_pct:.2f}% | "
        f"Current nMAE: "
        f"{current_nmae_pct:.2f}% | "
        f"Delta: {nmae_delta_pp:+.2f} p.p."
    )

    if performance_drift_detected:
        logger.warning(
            "🚨 PERFORMANCE DRIFT DETECTADO | "
            f"Degradação de {nmae_delta_pp:.2f} p.p. "
            f"(threshold: "
            f"{PERFORMANCE_NMAE_DELTA_PP_THRESHOLD:.2f} p.p.)"
        )

    return (
        performance_drift_detected,
        float(baseline_nmae_pct),
        float(current_nmae_pct),
        float(nmae_delta_pp),
    )


@flow(name="Pipeline de Monitoramento em Lote (Evidently AI)")
def batch_monitoring_pipeline(
    reference_path: str = (
        "energy-lake/gold/"
        "train_wind_energy_2024_01_expanding_up_to_2026_08.parquet"
    ),
    current_path: str = (
        "energy-lake/gold/"
        "oot_test_wind_energy_2026_08.parquet"
    ),
    output_report_path: str = (
        "energy-lake/monitoring/drift_report.html"
    ),
):
    logger = get_run_logger()

    logger.info("Iniciando Pipeline de Monitoramento MLOps")

    # 1. Carga
    df_ref, df_cur = fetch_monitoring_data(
        reference_path,
        current_path,
    )

    # 2. Engenharia + contrato de features
    X_ref, X_cur, df_cur_feat = (
        prepare_monitoring_features(
            df_ref,
            df_cur,
        )
    )

    # 3. Data Drift
    html_report, data_drift_detected, share_drifted = (
        generate_evidently_report(
            X_ref,
            X_cur,
        )
    )

    # 4. Performance Drift
    (
        performance_drift_detected,
        _baseline_nmae_pct,
        current_nmae_pct,
        nmae_delta_pp,
    ) = evaluate_performance_drift(
        X_cur,
        df_cur_feat,
    )

    # 5. Persistência
    save_report_to_s3(
        html_report,
        output_report_path,
    )

    # 6. Governance / Continuous Training
    if data_drift_detected or performance_drift_detected:

        reasons = []

        if data_drift_detected:
            reasons.append(
                f"Data Drift={share_drifted:.1%}"
            )

        if performance_drift_detected:
            reasons.append(
                f"Performance Drift="
                f"{nmae_delta_pp:+.2f} p.p."
            )

        logger.warning(
            "🚨 RETRAINING TRIGGERED | "
            + " | ".join(reasons)
        )

        continuous_training_pipeline()

    else:
        logger.info(
            "✅ Data Drift e Performance Drift "
            "dentro dos limites de governança. "
            f"Current nMAE: {current_nmae_pct:.2f}%."
        )


if __name__ == "__main__":
    batch_monitoring_pipeline()