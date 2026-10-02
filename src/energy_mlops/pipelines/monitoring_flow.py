import os
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import s3fs
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

from energy_mlops.config import settings
from energy_mlops.data.build_features import (
    generate_wind_and_time_features,
)
from energy_mlops.data.feature_utils import (
    select_model_features,
)
from energy_mlops.models.evaluation import (
    evaluate_model_on_oot,
    get_model_feature_order,
)
from energy_mlops.pipelines.training_flow import (
    continuous_training_pipeline,
    resolve_metric,
)

"""
                 Monitoring
                     │
           ┌─────────┴─────────┐
           │                   │
       DATA DRIFT       PERFORMANCE DRIFT
           │                   │
    Evidently Share      Champion MLflow
           │                   │
        >= 50%?          Current nMAE
           │                   │
           │          Delta >= 2.0 p.p.?
           │                   │
           └─────────┬─────────┘
                     │
                  OR rule
                     │
              Continuous Training
"""


# ==============================================================================
# CONFIGURAÇÃO DE GOVERNANÇA
# ==============================================================================

DRIFT_SHARE_THRESHOLD = 0.50

# Unidade: pontos percentuais de nMAE.
PERFORMANCE_NMAE_DELTA_PP_THRESHOLD = 2.0

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"


# ==============================================================================
# PATHS DEFAULT PARA A VALIDAÇÃO ATUAL
# ==============================================================================

DEFAULT_REFERENCE_PATH = (
    "energy-lake/gold/"
    "train_wind_energy_2024_01_expanding_up_to_2026_08.parquet"
)

DEFAULT_CURRENT_PATH = (
    "energy-lake/gold/"
    "oot_test_wind_energy_2026_08.parquet"
)

DEFAULT_REPORT_PATH = (
    "energy-lake/monitoring/drift_report.html"
)


# ==============================================================================
# CREDENCIAIS RUSTFS / MLFLOW
# ==============================================================================

os.environ["AWS_ACCESS_KEY_ID"] = (
    settings.RUSTFS_ROOT_USER
)

os.environ["AWS_SECRET_ACCESS_KEY"] = (
    settings.RUSTFS_ROOT_PASSWORD
)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
    settings.RUSTFS_ENDPOINT
)

os.environ["AWS_ENDPOINT_URL"] = (
    settings.RUSTFS_ENDPOINT
)


# ==============================================================================
# 1. EXTRAÇÃO DOS DADOS
# ==============================================================================

@task(
    name="Extrair Dados de Monitoramento",
    retries=2,
)
def fetch_monitoring_data(
    reference_path: str,
    current_path: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Extrai os datasets Reference e Current
    do Data Lake RustFS/S3.
    """

    logger = get_run_logger()

    fs = s3fs.S3FileSystem()

    logger.info(
        f"Lendo Reference: {reference_path}"
    )

    with fs.open(
        reference_path,
        "rb",
    ) as file:
        df_ref = pd.read_parquet(
            file
        )

    logger.info(
        f"Lendo Current: {current_path}"
    )

    with fs.open(
        current_path,
        "rb",
    ) as file:
        df_cur = pd.read_parquet(
            file
        )

    return (
        df_ref,
        df_cur,
    )


# ==============================================================================
# 2. ENGENHARIA + CONTRATO CANÔNICO DE FEATURES
# ==============================================================================

@task(
    name="Preparar Features para Monitoramento"
)
def prepare_monitoring_features(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Aplica a mesma engenharia de features
    utilizada pelo pipeline de treinamento.

    O Data Drift utiliza sempre o contrato
    canônico atual de features do projeto.

    A compatibilidade com modelos históricos
    é tratada posteriormente, somente no momento
    da inferência do Champion.
    """

    logger = get_run_logger()

    logger.info(
        "Aplicando engenharia de features "
        "ao dataset Reference..."
    )

    df_ref_feat = (
        generate_wind_and_time_features(
            df_ref
        )
    )

    logger.info(
        "Aplicando engenharia de features "
        "ao dataset Current..."
    )

    df_cur_feat = (
        generate_wind_and_time_features(
            df_cur
        )
    )

    logger.info(
        "Validando e selecionando "
        "features do modelo..."
    )

    X_ref = select_model_features(
        df_ref_feat
    )

    X_cur = select_model_features(
        df_cur_feat
    )

    logger.info(
        "Features monitoradas "
        f"({len(X_ref.columns)}): "
        f"{X_ref.columns.tolist()}"
    )

    return (
        X_ref,
        X_cur,
        df_cur_feat,
    )


# ==============================================================================
# 3. EXTRAÇÃO DOS INDICADORES DO EVIDENTLY
# ==============================================================================

def extract_drift_result(
    snapshot: Any,
) -> tuple[
    int,
    float,
]:
    """
    Extrai a contagem e a proporção de
    features com drift do relatório Evidently.

    O parser aceita diferentes estruturas
    produzidas por versões do Evidently.
    """

    report_dict = snapshot.dict()

    def find_metrics(
        obj: Any,
    ) -> tuple[int, float] | None:

        if isinstance(
            obj,
            dict,
        ):
            if (
                "number_of_drifted_columns"
                in obj
                and "share_of_drifted_columns"
                in obj
            ):
                return (
                    int(
                        obj[
                            "number_of_drifted_columns"
                        ]
                    ),
                    float(
                        obj[
                            "share_of_drifted_columns"
                        ]
                    ),
                )

            if (
                "count" in obj
                and "share" in obj
                and isinstance(
                    obj["count"],
                    (int, float),
                )
            ):
                return (
                    int(
                        obj["count"]
                    ),
                    float(
                        obj["share"]
                    ),
                )

            for value in obj.values():
                result = find_metrics(
                    value
                )

                if result is not None:
                    return result

        elif isinstance(
            obj,
            list,
        ):
            for item in obj:
                result = find_metrics(
                    item
                )

                if result is not None:
                    return result

        return None

    result = find_metrics(
        report_dict
    )

    if result is None:
        raise RuntimeError(
            "Não foi possível extrair os "
            "indicadores de Drift do relatório "
            "do Evidently."
        )

    return result


# ==============================================================================
# 4. DATA DRIFT
# ==============================================================================

@task(
    name=(
        "Gerar Relatório Evidently "
        "& Avaliar Drift"
    )
)
def generate_evidently_report(
    X_ref: pd.DataFrame,
    X_cur: pd.DataFrame,
) -> tuple[
    str,
    bool,
    float,
]:
    """
    Gera o relatório de Data Drift somente
    sobre as features do contrato do modelo.
    """

    logger = get_run_logger()

    features = (
        X_ref.columns.tolist()
    )

    report = Report(
        [
            DataDriftPreset(
                columns=features,
                drift_share=(
                    DRIFT_SHARE_THRESHOLD
                ),
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
        snapshot.save_html(
            tmp_path
        )

        with open(
            tmp_path,
            "r",
            encoding="utf-8",
        ) as file:
            html_content = file.read()

    finally:
        os.unlink(
            tmp_path
        )

    (
        drifted_count,
        share_drifted,
    ) = extract_drift_result(
        snapshot
    )

    drift_detected = (
        share_drifted
        >= DRIFT_SHARE_THRESHOLD
    )

    logger.info(
        "Drift Analysis: "
        f"{drifted_count}/"
        f"{len(features)} "
        "features afetadas "
        f"({share_drifted:.1%})"
    )

    return (
        html_content,
        drift_detected,
        share_drifted,
    )


# ==============================================================================
# 5. PERSISTÊNCIA DO RELATÓRIO
# ==============================================================================

@task(
    name="Salvar Relatório no RustFS"
)
def save_report_to_s3(
    html_content: str,
    destination_path: str,
) -> None:
    """
    Persiste o relatório HTML de Data Drift
    no Data Lake.
    """

    logger = get_run_logger()

    fs = s3fs.S3FileSystem()

    with fs.open(
        destination_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            html_content
        )

    logger.info(
        "Relatório Evidently salvo em: "
        f"{destination_path}"
    )


# ==============================================================================
# 6. PERFORMANCE DRIFT
# ==============================================================================


@task(
    name="Avaliar Performance Drift"
)
def evaluate_performance_drift(
    X_cur: pd.DataFrame,
    df_cur_feat: pd.DataFrame,
) -> tuple[
    bool,
    float,
    float,
    float,
]:
    """
    Compara a performance atual do Champion
    contra sua baseline OOT registrada no MLflow.

    Returns
    -------
    performance_drift_detected
        True quando a degradação ultrapassa
        o threshold de governança.

    baseline_nmae_pct
        nMAE OOT registrada pelo Champion.

    current_nmae_pct
        nMAE calculada no dataset Current.

    nmae_delta_pp
        Diferença Current - Baseline
        em pontos percentuais.
    """

    logger = get_run_logger()

    required_columns = {
        "wind_generation_mw",
        "capacidade_mw",
    }

    missing_columns = (
        required_columns.difference(
            df_cur_feat.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Performance Drift exige ground "
            "truth e capacidade. "
            "Colunas ausentes: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------------------------
    # MLflow Champion
    # --------------------------------------------------------------------------

    mlflow.set_tracking_uri(
        settings.MLFLOW_TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=(
            settings.MLFLOW_TRACKING_URI
        )
    )

    champion_version = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )
    )

    champion_run = client.get_run(
        champion_version.run_id
    )

    metrics = (
        champion_run.data.metrics
    )

    # Contrato canônico com compatibilidade
    # histórica genérica resolvida em training_flow.
    baseline_nmae_pct = (
        resolve_metric(
            metrics,
            "oot_nmae_pct",
        )
    )

    # --------------------------------------------------------------------------
    # Carregamento do Champion
    # --------------------------------------------------------------------------

    model_uri = (
        f"models:/"
        f"{MODEL_NAME}"
        f"@{MODEL_ALIAS}"
    )

    champion_model = (
        mlflow.sklearn.load_model(
            model_uri
        )
    )

    logger.info(
        "Contrato de inferência do Champion "
        f"v{champion_version.version}: "
        f"{get_model_feature_order(champion_model)}"
    )

    current_evaluation = evaluate_model_on_oot(
        champion_model,
        X_cur,
        df_cur_feat,
    )

    current_nmae_pct = (
        current_evaluation.nmae_pct
    )

    nmae_delta_pp = (
        current_nmae_pct
        - baseline_nmae_pct
    )

    performance_drift_detected = (
        nmae_delta_pp
        >= (
            PERFORMANCE_NMAE_DELTA_PP_THRESHOLD
        )
    )

    logger.info(
        "Performance Monitoring | "
        "Champion baseline nMAE: "
        f"{baseline_nmae_pct:.2f}% | "
        "Current nMAE: "
        f"{current_nmae_pct:.2f}% | "
        "Delta: "
        f"{nmae_delta_pp:+.2f} p.p."
    )

    if performance_drift_detected:
        logger.warning(
            "🚨 PERFORMANCE DRIFT DETECTADO | "
            "Degradação de "
            f"{nmae_delta_pp:.2f} p.p. "
            "(threshold: "
            f"{PERFORMANCE_NMAE_DELTA_PP_THRESHOLD:.2f} "
            "p.p.)"
        )

    return (
        performance_drift_detected,
        float(
            baseline_nmae_pct
        ),
        float(
            current_nmae_pct
        ),
        float(
            nmae_delta_pp
        ),
    )


# ==============================================================================
# 7. PIPELINE DE MONITORAMENTO
# ==============================================================================

@flow(
    name=(
        "Pipeline de Monitoramento "
        "em Lote (Evidently AI)"
    )
)
def batch_monitoring_pipeline(
    reference_path: str = (
        DEFAULT_REFERENCE_PATH
    ),
    current_path: str = (
        DEFAULT_CURRENT_PATH
    ),
    output_report_path: str = (
        DEFAULT_REPORT_PATH
    ),
):
    """
    Pipeline principal de Monitoring.

    A decisão de Continuous Training utiliza
    uma regra OR:

        Data Drift
            OR
        Performance Drift

    O relatório Evidently é persistido antes
    da avaliação de Performance Drift para
    preservar a evidência de Data Drift mesmo
    caso uma etapa posterior falhe.
    """

    logger = get_run_logger()

    logger.info(
        "Iniciando Pipeline de "
        "Monitoramento MLOps"
    )

    # --------------------------------------------------------------------------
    # 1. Carga
    # --------------------------------------------------------------------------

    (
        df_ref,
        df_cur,
    ) = fetch_monitoring_data(
        reference_path,
        current_path,
    )

    # --------------------------------------------------------------------------
    # 2. Engenharia + contrato canônico
    # --------------------------------------------------------------------------

    (
        X_ref,
        X_cur,
        df_cur_feat,
    ) = prepare_monitoring_features(
        df_ref,
        df_cur,
    )

    # --------------------------------------------------------------------------
    # 3. Data Drift
    # --------------------------------------------------------------------------

    (
        html_report,
        data_drift_detected,
        share_drifted,
    ) = generate_evidently_report(
        X_ref,
        X_cur,
    )

    # --------------------------------------------------------------------------
    # 4. Persistência imediata do relatório
    # --------------------------------------------------------------------------

    save_report_to_s3(
        html_report,
        output_report_path,
    )

    # --------------------------------------------------------------------------
    # 5. Performance Drift
    # --------------------------------------------------------------------------

    (
        performance_drift_detected,
        baseline_nmae_pct,
        current_nmae_pct,
        nmae_delta_pp,
    ) = evaluate_performance_drift(
        X_cur,
        df_cur_feat,
    )

    # --------------------------------------------------------------------------
    # 6. Governança / Continuous Training
    # --------------------------------------------------------------------------

    if (
        data_drift_detected
        or performance_drift_detected
    ):
        reasons = []

        if data_drift_detected:
            reasons.append(
                "Data Drift="
                f"{share_drifted:.1%}"
            )

        if performance_drift_detected:
            reasons.append(
                "Performance Drift="
                f"{nmae_delta_pp:+.2f} p.p."
            )

        logger.warning(
            "🚨 RETRAINING TRIGGERED | "
            + " | ".join(
                reasons
            )
        )

        continuous_training_pipeline()

    else:
        logger.info(
            "✅ Data Drift e Performance Drift "
            "dentro dos limites de governança. "
            "Champion baseline nMAE: "
            f"{baseline_nmae_pct:.2f}% | "
            "Current nMAE: "
            f"{current_nmae_pct:.2f}%."
        )


if __name__ == "__main__":
    batch_monitoring_pipeline()