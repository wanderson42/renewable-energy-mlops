"""Diagnóstico observacional do protocolo inicial de monitoramento v0.3.

Usa o JSON histórico de uma execução como contrato: modelo, referência e
snapshot atuais fixos. Não altera dados, regras de alerta ou aliases.
Métricas reproduzem evaluate_model_on_oot: MAE / capacidade máxima da janela.
As observações horárias são dependentes; quantis e erros diários são descritivos.
Dados meteorológicos são observados/reanálise, não um backtest day-ahead.
"""
import argparse
import json
import os
import zipfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import s3fs
from mlflow.tracking import MlflowClient

from energy_mlops.config import settings
from energy_mlops.models.evaluation import (
    evaluate_model_on_oot,
    get_model_feature_order,
)

METEOROLOGICAL = [
    "temperature_2m", "wind_speed_100m", "wind_direction_100m",
    "wind_temp_ratio", "wind_speed_roll_mean_3h",
]


def normalize(frame):
    result = frame.copy()
    dates = pd.to_datetime(result["date"], utc=True, errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Datas ausentes ou duplicadas.")
    if not dates.eq(dates.dt.floor("h")).all():
        raise ValueError("Datas fora da grade horária.")
    result["date"] = dates
    result["dia_local"] = dates.dt.tz_convert("America/Sao_Paulo").dt.strftime("%Y-%m-%d")
    return result.sort_values("date").reset_index(drop=True)


def valid_truth(frame):
    truth = pd.to_numeric(frame["wind_generation_mw"], errors="coerce")
    capacity = pd.to_numeric(frame["capacidade_mw"], errors="coerce")
    return np.isfinite(truth) & truth.ge(0) & np.isfinite(capacity) & capacity.gt(0)


def distribution_rows(frame, window):
    rows = []
    for feature in METEOROLOGICAL:
        values = pd.to_numeric(frame[feature], errors="raise")
        row = {"janela": window, "feature": feature, "horas": len(values)}
        if feature == "wind_direction_100m":
            # Direção é circular: média aritmética e quantis podem enganar em 0/360.
            angles = np.deg2rad(values.to_numpy())
            sine, cosine = np.sin(angles).mean(), np.cos(angles).mean()
            resultant = float(np.hypot(sine, cosine))
            row.update(
                media_circular_graus=(float(np.degrees(np.arctan2(sine, cosine)) % 360)
                                     if resultant > 1e-12 else None),
                concentracao_circular=resultant,
            )
        else:
            row.update(media=float(values.mean()), mediana=float(values.median()))
            row.update({f"p{q}": float(values.quantile(q / 100)) for q in [10, 25, 75, 90]})
        rows.append(row)
    return rows


def main(summary_path, output_dir):
    fs = s3fs.S3FileSystem(**settings.storage_options)
    with fs.open(summary_path, "r", encoding="utf-8") as stream:
        summary = json.load(stream)
    context = summary["model"]
    # MLflow usa boto3 para artefatos; storage_options configura apenas s3fs.
    os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
    os.environ["AWS_ENDPOINT_URL_S3"] = settings.RUSTFS_ENDPOINT
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version(context["model_name"], context["version"])
    if version.run_id != context["run_id"]:
        raise ValueError("Proveniência do modelo diverge do resumo da execução.")
    model = mlflow.sklearn.load_model(f"models:/{context['model_name']}/{context['version']}")
    features = get_model_feature_order(model)
    reference = normalize(pd.read_parquet(summary["reference_path"], storage_options=settings.storage_options))
    current = normalize(pd.read_parquet(summary["current_path"], storage_options=settings.storage_options))
    for name, frame, metadata in [
        ("reference", reference, summary["reference_window"]),
        ("current", current, summary["current_window"]),
    ]:
        if len(frame) != metadata["rows"] or frame.empty:
            raise ValueError(f"Snapshot {name} não corresponde ao número de linhas registrado.")
        if frame["date"].min() != pd.Timestamp(metadata["start_utc"]) or frame["date"].max() != pd.Timestamp(metadata["end_utc"]):
            raise ValueError(f"Snapshot {name} diverge das datas registradas.")
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f"Features inválidas em {name}.")
    if not valid_truth(reference).all():
        raise ValueError("Referência possui ground truth inválido.")
    baseline = evaluate_model_on_oot(model, reference[features], reference)
    if not np.isclose(baseline.nmae_pct, context["baseline_nmae_pct"], atol=1e-6, rtol=0):
        raise ValueError("Baseline recalculada diverge da baseline registrada; investigar antes de comparar.")
    valid = valid_truth(current)
    if int(valid.sum()) != summary["performance"]["evaluated_rows"]:
        raise ValueError("Cobertura de ground truth diverge da execução registrada.")
    daily = []
    for day, frame in current.groupby("dia_local", sort=True):
        selected = frame.loc[valid_truth(frame)]
        row = {"dia_local": day, "horas_meteorologicas": len(frame),
               "horas_truth_validas": len(selected), "horas_esperadas": 24,
               "status": "SEM_TRUTH" if selected.empty else
                         "DIA_COMPLETO" if len(selected) == 24 else "TRUTH_PARCIAL"}
        if not selected.empty:
            result = evaluate_model_on_oot(model, selected[features], selected)
            row.update(mae_mw=result.mae_mw, nmae_pct=result.nmae_pct,
                       capacidade_max_mw=float(selected["capacidade_mw"].max()),
                       delta_vs_baseline_pp=result.nmae_pct - baseline.nmae_pct)
        daily.append(row)
    distributions = distribution_rows(reference, "referencia_OOT")
    distributions += distribution_rows(current, "current_total")
    for day, frame in current.groupby("dia_local", sort=True):
        distributions += distribution_rows(frame, day)
    # Inclui dias de setembro para contextualizar a variabilidade do erro diário.
    historical = []
    for day, frame in reference.groupby("dia_local", sort=True):
        result = evaluate_model_on_oot(model, frame[features], frame)
        historical.append({"dia_local": day, "horas": len(frame),
                           "mae_mw": result.mae_mw, "nmae_pct": result.nmae_pct})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "performance_diaria.csv": pd.DataFrame(daily),
        "distribuicoes.csv": pd.DataFrame(distributions),
        "performance_diaria_referencia.csv": pd.DataFrame(historical),
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    if valid.any():
        total = evaluate_model_on_oot(model, current.loc[valid, features], current.loc[valid])
        if not np.isclose(total.nmae_pct, summary["performance"]["current_nmae_pct"], atol=1e-6, rtol=0):
            raise ValueError("Performance recalculada diverge da execução registrada.")
    with fs.open(summary["report_path"], "rb") as stream:
        html = stream.read()
    with zipfile.ZipFile(output / "diagnostico_monitoramento.zip", "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("resumo_execucao.json", json.dumps(summary, ensure_ascii=False, indent=2))
        package.writestr("relatorio.html", html)
        for filename, table in tables.items():
            package.writestr(filename, table.to_csv(index=False))
    print(pd.DataFrame(daily).to_string(index=False))
    print("\nDistribuições:")
    print(pd.DataFrame(distributions).to_string(index=False))
    print(f"\nDiagnóstico salvo em: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output-dir", default="validation_20261005/monitoring")
    args = parser.parse_args()
    main(**vars(args))
