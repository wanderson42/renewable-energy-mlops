"""Snapshot parcial de monitoring; não modifica o Gold mensal nem aciona CT."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from prefect import flow

from energy_mlops.config import settings
from energy_mlops.data.build_features import generate_wind_and_time_features
from energy_mlops.data.extract_energy import fetch_ons_wind_generation
from energy_mlops.data.extract_weather import fetch_open_meteo_wind_data
from energy_mlops.data.snapshot_validation import audit_gold_snapshot


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise").dt.tz_localize(None)
    if result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("Datas ausentes ou duplicadas na fonte de monitoring.")
    return result.sort_values("date").reset_index(drop=True)


def build_partial_snapshot(
    weather: pd.DataFrame, energy: pd.DataFrame | None,
    context: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """Intervalo UTC [start,end); contexto anterior apenas para rolling."""
    weather = normalize_dates(weather)
    context = normalize_dates(context)
    current = weather.loc[weather["date"].ge(start) & weather["date"].lt(end)].copy()
    # Contexto meteorológico vem do snapshot canônico, nunca do futuro.
    columns = ["date", "temperature_2m", "wind_speed_100m", "wind_direction_100m"]
    previous = context.loc[
        context["date"].ge(start - pd.Timedelta(hours=2)) & context["date"].lt(start), columns,
    ]
    if len(previous) != 2 or not previous["date"].tolist() == [
        start - pd.Timedelta(hours=2), start - pd.Timedelta(hours=1),
    ]:
        raise ValueError("Contexto exige as duas horas imediatamente anteriores ao início.")
    if current.empty:
        raise ValueError("Nenhuma hora meteorológica observada no intervalo solicitado.")
    combined = pd.concat([previous, current[columns]], ignore_index=True)
    featured = generate_wind_and_time_features(combined)
    result = featured.loc[featured["date"].ge(start)].reset_index(drop=True)
    if energy is not None:
        energy = normalize_dates(energy)
        # Preserva todas as horas meteorológicas; truth ausente continua ausente.
        result = result.merge(
            energy[["date", "wind_generation_mw"]], on="date", how="left", validate="one_to_one",
        )
        result["target_fc"] = (result["wind_generation_mw"] / result["capacidade_mw"]).clip(0, 1)
        observed = result.loc[result["wind_generation_mw"].notna()].copy()
        if not observed.empty:
            audit_gold_snapshot(observed)
    return result


@flow(name="Ingestão parcial para Monitoring", log_prints=True)
def monitoring_ingestion_flow(
    start_date: str, end_date: str,
    context_path: str,
    weather_only: bool = False,
) -> str:
    """Datas locais inclusivas; apenas dias passados, num mesmo mês."""
    start_local = pd.Timestamp(start_date)
    end_local = pd.Timestamp(end_date)
    if start_local != start_local.normalize() or end_local != end_local.normalize():
        raise ValueError("Informe datas sem horário: YYYY-MM-DD.")
    if start_local > end_local or start_local.to_period("M") != end_local.to_period("M"):
        raise ValueError("Intervalo deve estar ordenado e pertencer ao mesmo mês.")
    today = pd.Timestamp(datetime.now(ZoneInfo("America/Sao_Paulo")).date())
    if end_local >= today:
        raise ValueError("Use somente dias passados; o dia atual ainda está em andamento.")
    start = start_local.tz_localize("America/Sao_Paulo").tz_convert("UTC").tz_localize(None)
    end = (end_local + pd.Timedelta(days=1)).tz_localize("America/Sao_Paulo").tz_convert("UTC").tz_localize(None)
    weather = fetch_open_meteo_wind_data(start_date=start_date, end_date=end_date)
    # Falhas de publicação não são convertidas silenciosamente em truth ausente.
    energy = None if weather_only else fetch_ons_wind_generation(start_local.year, start_local.month)
    context = pd.read_parquet(context_path, storage_options=settings.storage_options)
    result = build_partial_snapshot(weather, energy, context, start, end)
    key = f"monitoring/current_{start_date}_{end_date}{'_weather_only' if weather_only else ''}.parquet"
    path = f"s3://{settings.RUSTFS_BUCKET}/{key}"
    result.to_parquet(path, index=False, storage_options=settings.storage_options)
    print(f"Snapshot: {path} | rows={len(result)} | {result['date'].min()} → {result['date'].max()}")
    if "wind_generation_mw" in result:
        print(f"Ground truth disponível: {result['wind_generation_mw'].notna().sum()}/{len(result)}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--context-path", default=f"s3://{settings.RUSTFS_BUCKET}/gold/dataset_renewable_energy_2024_03_2026_09.parquet")
    parser.add_argument("--weather-only", action="store_true")
    args = parser.parse_args()
    monitoring_ingestion_flow(**vars(args))
