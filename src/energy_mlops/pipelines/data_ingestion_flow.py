import argparse
import calendar
import os

import pandas as pd
from prefect import flow, task

from energy_mlops.data.build_features import generate_wind_and_time_features
from energy_mlops.data.extract_energy import fetch_ons_wind_generation
from energy_mlops.data.extract_weather import fetch_open_meteo_wind_data
from energy_mlops.data.snapshot_validation import (
    SnapshotAuditResult,
    audit_gold_snapshot,
)

# Importamos as configurações do nosso Data Lake
from energy_mlops.pipelines.utils import save_dataset_to_lake_or_local


def validate_energy_month_boundary(
    df_energy: pd.DataFrame,
    year: int,
    month: int,
) -> None:
    """
    Valida se a série ONS alcança a última hora
    esperada do mês solicitado.

    Lacunas internas podem existir devido ao controle
    de qualidade do extractor, mas o final truncado do
    mês indica que a publicação ainda está incompleta.
    """

    if df_energy.empty:
        raise ValueError(
            "Dataset ONS vazio para "
            f"{year}-{month:02d}."
        )

    last_day = calendar.monthrange(
        year,
        month,
    )[1]

    expected_last_local = pd.Timestamp(
        year=year,
        month=month,
        day=last_day,
        hour=23,
        tz="America/Sao_Paulo",
    )

    expected_last_utc = (
        expected_last_local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    actual_last = pd.to_datetime(
        df_energy["date"], utc=True
    ).dt.tz_localize(None).max()

    if actual_last < expected_last_utc:
        raise ValueError(
            "Mês ONS ainda incompleto para "
            f"{year}-{month:02d}. "
            f"Última hora disponível: {actual_last}. "
            "Última hora esperada: "
            f"{expected_last_utc}."
        )


@task(
    name="Extrair_Dados_Climaticos",
    retries=2,
    retry_delay_seconds=10,
)
def extract_weather_task(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Extrai os dados meteorológicos do Open-Meteo."""
    return fetch_open_meteo_wind_data(
        start_date=start_date,
        end_date=end_date,
    )


@task(
    name="Extrair_Dados_Energia",
    retries=2,
    retry_delay_seconds=10,
)
def extract_energy_task(
    year: int,
    month: int,
) -> pd.DataFrame:
    """Extrai a geração eólica horária do ONS."""
    return fetch_ons_wind_generation(
        year=year,
        month=month,
    )


@task(
    name="Mesclar_e_Limpar_Dados",
)
def merge_datasets_task(
    df_weather: pd.DataFrame,
    df_energy: pd.DataFrame
) -> pd.DataFrame:
    """Une (JOIN) os DataFrames pela data (já alinhada em UTC) e trata os nulos."""
    
    # Inner join garante que só manteremos as horas que existem em AMBAS as fontes
    df_merged = pd.merge(df_weather, df_energy, on="date", how="inner")
    
    # Ordena cronologicamente e reseta o índice
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    
    return df_merged


@task(name="Build_Derived_Features")
def transform_features_task(df: pd.DataFrame) -> pd.DataFrame:
    """Task do Prefect que envelopa a função pura de feature engineering."""
    return generate_wind_and_time_features(df)



@task(name="Auditar_Snapshot_Gold")
def audit_gold_snapshot_task(
    df: pd.DataFrame,
) -> SnapshotAuditResult:
    report = audit_gold_snapshot(
        df
    )

    print(
        "Gold auditado com sucesso | "
        f"rows={report.rows:,} | "
        f"range={report.start_date} "
        f"→ {report.end_date} | "
        "missing_timestamps="
        f"{report.missing_timestamps} | "
        f"gaps={report.gap_count} | "
        f"features={report.feature_count}"
    )

    return report


@flow(name="Pipeline_de_Ingestao", log_prints=True)
def data_ingestion_flow(year: int = 2025, month: int = 1):
    """Flow principal que orquestra todo o processo."""
    
    # Descobrir o último dia do mês para passar para o Open-Meteo
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day}"

    print(f"Iniciando Pipeline de Ingestão para {year}-{month:02d}...")

    # Executa as tasks de extração
    df_weather = extract_weather_task(
        start_date=start_date,
        end_date=end_date,
    )

    df_energy = extract_energy_task(
        year=year,
        month=month,
    )

    # Garante que a publicação ONS alcançou
    # o fechamento do mês solicitado.
    validate_energy_month_boundary(
        df_energy=df_energy,
        year=year,
        month=month,
    )

    # Une os dados somente após validar
    # o fechamento da série de energia.
    df_merged = merge_datasets_task(
        df_weather=df_weather,
        df_energy=df_energy,
    )

    print("Aplicando Engenharia de Features (Física e Sazonalidade)...")
    df_full = transform_features_task(df_merged)

    print("Auditando o Snapshot Gold Dataset...")
    audit_gold_snapshot_task(df_full)

    print(f"Pipeline concluído! Dataset final gerado com {len(df_full)} registros.")
    print(df_full.head())
    
    # Caminhos para salvamento dos Dados
    key = f"gold/dataset_renewable_energy_{year}_{month:02d}.parquet"
    local_path = f"data/dataset_renewable_energy_{year}_{month:02d}.parquet"
        
    # Prepara o diretório local antecipadamente para o fallback
    os.makedirs("data", exist_ok=True)

    final_path = save_dataset_to_lake_or_local(df_full, key, local_path)

    return final_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa o backfill histórico de dados eólicos e climáticos."
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Ano inicial da extração (ex: 2025)"
    )

    parser.add_argument(
        "--month",
        type=int,
        default=1,
        help="Mês da extração (1 a 12)"
    )

    args = parser.parse_args()

    print(f"🚀 Iniciando Backfill: {args.year}/{args.month:02d}")

    # Executa o fluxo do Prefect passando os argumentos do terminal
    data_ingestion_flow(
        year=args.year,
        month=args.month
    )

# ==============================================================================
# EXEMPLOS DE EXECUÇÃO VIA TERMINAL BASH
# ==============================================================================
# Pré-requisito: Garantir que o servidor local do Prefect está rodando
# ~/renewable-energy-mlops$ poetry run prefect server start > prefect.log 2>&1 &
# ------------------------------------------------------------------------------
# MODO PADRÃO (Executa a função que estiver descomentada no bloco __main__)
# ------------------------------------------------------------------------------
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.data_ingestion_flow
#
# ------------------------------------------------------------------------------
# MODO PARAMETRIZADO (via argparse)
# ------------------------------------------------------------------------------
# Exemplo: Executa a ingestão de dados eólicos e climáticos para Janeiro de 2025
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.data_ingestion_flow --year 2025 --month 1