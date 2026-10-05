import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
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
    get_model_feature_columns,
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
    "oot_test_wind_energy_2026_09.parquet"
)

DEFAULT_CURRENT_PATH = (
    "energy-lake/gold/"
    "oot_test_wind_energy_2026_09.parquet"
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

    fs = s3fs.S3FileSystem(**settings.storage_options)

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

def prepare_feature_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Preserva features Gold; gera features apenas para entradas brutas."""
    features = get_model_feature_columns()
    if set(features).issubset(df.columns) and "capacidade_mw" in df:
        return df.copy()
    return generate_wind_and_time_features(df)


def describe_window(df: pd.DataFrame) -> dict:
    """Cobertura horária no intervalo observado, sem imputar lacunas."""
    if df.empty or "date" not in df:
        raise ValueError("Monitoring exige dataset não vazio com date.")
    dates = pd.to_datetime(df["date"], utc=True, errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Monitoring exige timestamps válidos e únicos.")
    if not dates.eq(dates.dt.floor("h")).all():
        raise ValueError("Monitoring exige timestamps na grade horária.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Monitoring exige ordenação temporal.")
    expected = pd.date_range(dates.min(), dates.max(), freq="h")
    local = dates.dt.tz_convert("America/Sao_Paulo")
    first = local.min()
    next_month = (first.tz_localize(None).to_period("M") + 1).start_time
    end_month = next_month.tz_localize("America/Sao_Paulo") - pd.Timedelta(hours=1)
    complete_month = (
        local.min() == first.normalize().replace(day=1)
        and local.max() == end_month
        and len(dates) == len(expected)
    )
    return {
        "rows": len(df), "start_utc": dates.min().isoformat(),
        "end_utc": dates.max().isoformat(),
        "missing_hours_in_observed_interval": len(expected.difference(dates)),
        "window_status": "COMPLETE_MONTH" if complete_month else "PARTIAL_WINDOW",
    }


def resolve_monitoring_model() -> dict:
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(version.run_id)
    return {
        "model_name": MODEL_NAME, "version": str(version.version),
        "run_id": version.run_id,
        "baseline_nmae_pct": float(resolve_metric(run.data.metrics, "oot_nmae_pct")),
        "training_end": getattr(run.data, "tags", {}).get("training_end"),
    }


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
        prepare_feature_snapshot(df_ref)
    )

    logger.info(
        "Aplicando engenharia de features "
        "ao dataset Current..."
    )

    df_cur_feat = (
        prepare_feature_snapshot(df_cur)
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

    fs = s3fs.S3FileSystem(**settings.storage_options)

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
    model_context: dict | None = None,
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

    context = model_context or resolve_monitoring_model()
    baseline_nmae_pct = context["baseline_nmae_pct"]
    model_uri = f"models:/{MODEL_NAME}/{context['version']}"

    champion_model = (
        mlflow.sklearn.load_model(
            model_uri
        )
    )

    logger.info(
        "Contrato de inferência do Champion "
        f"v{context['version']}: "
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


def build_hourly_comparison(
    model: Any, X: pd.DataFrame, frame: pd.DataFrame, context: dict, execution_id: str,
) -> pd.DataFrame:
    """Estimativas retrospectivas; truth indisponível permanece ausente."""
    if len(X) != len(frame) or not X.index.equals(frame.index):
        raise ValueError("Features e dados horários devem estar alinhados.")
    predictions = np.asarray(
        model.predict(X.loc[:, get_model_feature_order(model)]), dtype=float,
    ).reshape(-1)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise ValueError("Previsões horárias inválidas.")
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    capacity = pd.to_numeric(frame["capacidade_mw"], errors="coerce")
    truth = pd.to_numeric(
        frame.get("wind_generation_mw", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    capacity_valid = np.isfinite(capacity) & capacity.gt(0)
    truth_valid = np.isfinite(truth) & truth.ge(0) & capacity_valid
    predicted_mw = predictions * capacity.where(capacity_valid)
    observed_mw = truth.where(truth_valid)
    return pd.DataFrame({
        "execution_id": execution_id,
        "model_version": context["version"], "model_run_id": context["run_id"],
        "date": dates, "dia_local": dates.dt.tz_convert("America/Sao_Paulo").dt.strftime("%Y-%m-%d"),
        "predicted_mw": predicted_mw, "observed_mw": observed_mw,
        "capacidade_mw": capacity, "truth_valid": truth_valid,
        "error_mw": predicted_mw - observed_mw,
        "absolute_error_mw": (predicted_mw - observed_mw).abs(),
    }).reset_index(drop=True)


def summarize_daily_comparison(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in hourly.groupby("dia_local", sort=True):
        valid = group.loc[group["truth_valid"]]
        mae = float(valid["absolute_error_mw"].mean()) if len(valid) else np.nan
        capacity = float(valid["capacidade_mw"].max()) if len(valid) else np.nan
        rows.append({
            "dia_local": day, "horas_meteorologicas": len(group),
            "horas_truth_validas": len(valid), "horas_esperadas": 24,
            "status": "SEM_TRUTH" if valid.empty else
                      "DIA_COMPLETO" if len(valid) == 24 else "TRUTH_PARCIAL",
            "mae_mw": mae, "nmae_pct": mae / capacity * 100 if len(valid) else np.nan,
            "bias_mw": float(valid["error_mw"].mean()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


@task(name="Salvar comparação horária de geração")
def persist_generation_comparison(
    X: pd.DataFrame, frame: pd.DataFrame, context: dict, execution_id: str,
    history_base: str, performance: dict,
) -> dict:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(
        f"models:/{context['model_name']}/{context['version']}",
    )
    hourly = build_hourly_comparison(model, X, frame, context, execution_id)
    selected = hourly.loc[hourly["truth_valid"]]
    if len(selected) != performance.get("evaluated_rows", 0):
        raise ValueError("Cobertura horária diverge da avaliação de performance.")
    if not selected.empty:
        nmae = selected["absolute_error_mw"].mean() / selected["capacidade_mw"].max() * 100
        if not np.isclose(nmae, performance["current_nmae_pct"], atol=1e-6, rtol=0):
            raise ValueError("Artefato horário diverge do nMAE registrado.")
    daily = summarize_daily_comparison(hourly)
    fs = s3fs.S3FileSystem(**settings.storage_options)
    paths = {"hourly_csv_path": history_base + ".hourly.csv",
             "daily_csv_path": history_base + ".daily.csv"}
    for key, table in [("hourly_csv_path", hourly), ("daily_csv_path", daily)]:
        with fs.open(paths[key], "w", encoding="utf-8") as stream:
            table.to_csv(stream, index=False)
    return {**paths, "evaluation_kind": "observed_weather_retrospective"}


# ==============================================================================
# 7. PIPELINE DE MONITORAMENTO
# ==============================================================================

@flow(name="Pipeline de Monitoramento em Lote (Evidently AI)")
def batch_monitoring_pipeline(
    reference_path: str = DEFAULT_REFERENCE_PATH,
    current_path: str = DEFAULT_CURRENT_PATH,
    output_report_path: str = DEFAULT_REPORT_PATH,
    *,
    trigger_training: bool = False,
):
    """Observação por padrão; CT explícito somente em mês completo com truth."""
    logger = get_run_logger()
    context = resolve_monitoring_model()
    df_ref, df_cur = fetch_monitoring_data(reference_path, current_path)
    reference_window = describe_window(df_ref)
    current_window = describe_window(df_cur)
    if context["training_end"]:
        training_end = pd.Timestamp(context["training_end"])
        if training_end.tzinfo is None:
            training_end = training_end.tz_localize("UTC")
        if pd.Timestamp(current_window["start_utc"]) <= training_end:
            raise ValueError("Current sobrepõe o TRAIN do modelo avaliado.")
    X_ref, X_cur, df_cur_feat = prepare_monitoring_features(df_ref, df_cur)
    for X in (X_ref, X_cur):
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Features de monitoring devem ser finitas.")
    html, data_drift, share = generate_evidently_report(X_ref, X_cur)
    execution_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    history_base = output_report_path.removesuffix(".html") + "/" + execution_id
    # Evidência histórica primeiro; caminho atual preserva compatibilidade do dashboard.
    save_report_to_s3(html, history_base + ".html")
    save_report_to_s3(html, output_report_path)

    performance_drift = False
    performance = {"status": "UNAVAILABLE_GROUND_TRUTH"}
    if "wind_generation_mw" in df_cur_feat:
        truth = pd.to_numeric(df_cur_feat["wind_generation_mw"], errors="coerce")
        capacity = pd.to_numeric(df_cur_feat["capacidade_mw"], errors="coerce")
        valid = np.isfinite(truth) & truth.ge(0) & np.isfinite(capacity) & capacity.gt(0)
        if valid.any():
            performance_drift, baseline, current, delta = evaluate_performance_drift(
                X_cur.loc[valid], df_cur_feat.loc[valid], context,
            )
            performance = {
                "status": "AVAILABLE" if valid.all() else "PARTIAL_GROUND_TRUTH",
                "evaluated_rows": int(valid.sum()),
                "baseline_nmae_pct": baseline, "current_nmae_pct": current,
                "delta_nmae_pp": delta, "drift_detected": performance_drift,
            }
    generation_comparison = persist_generation_comparison(
        X_cur, df_cur_feat, context, execution_id, history_base, performance,
    )
    drift = data_drift or performance_drift
    eligible = (
        current_window["window_status"] == "COMPLETE_MONTH"
        and performance["status"] == "AVAILABLE"
    )
    summary = {
        "execution_id": execution_id, "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "model": context, "reference_path": reference_path, "current_path": current_path,
        "reference_window": reference_window, "current_window": current_window,
        "data_drift": {"share": share, "detected": data_drift, "threshold": DRIFT_SHARE_THRESHOLD},
        "performance": performance,
        "generation_comparison": generation_comparison,
        "performance_threshold_pp": PERFORMANCE_NMAE_DELTA_PP_THRESHOLD,
        "training_requested": trigger_training,
        "training_eligible": eligible,
        "training_status": "NOT_TRIGGERED",
        "report_path": history_base + ".html",
    }
    fs = s3fs.S3FileSystem(**settings.storage_options)
    summary_path = history_base + ".json"
    def persist_summary():
        with fs.open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    persist_summary()
    if trigger_training and eligible and drift:
        summary["training_status"] = "STARTED"
        persist_summary()
        try:
            continuous_training_pipeline()
        except Exception:
            summary["training_status"] = "FAILED"
            persist_summary()
            raise
        summary["training_status"] = "COMPLETED"
        persist_summary()
    elif trigger_training and not eligible:
        logger.warning("CT não executado: janela parcial ou ground truth incompleto.")
    logger.info(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring observacional por padrão.")
    parser.add_argument("--reference-path", default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--current-path", required=True)
    parser.add_argument("--output-report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--trigger-training", action="store_true")
    args = parser.parse_args()
    batch_monitoring_pipeline(
        reference_path=args.reference_path, current_path=args.current_path,
        output_report_path=args.output_report_path, trigger_training=args.trigger_training,
    )
