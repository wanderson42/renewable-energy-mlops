import argparse
import calendar
import os
from datetime import datetime, timezone

import pandas as pd
from prefect import flow

# Importamos as tasks já definidas no pipeline de ingestão para reutilizá-las
from energy_mlops.pipelines.data_ingestion_flow import (
    extract_energy_task,
    extract_weather_task,
    merge_datasets_task,
    transform_features_task,
)

# Importamos as configurações do nosso Data Lake
from energy_mlops.pipelines.utils import save_dataset_to_lake_or_local


@flow(name="Backfill_Historico", log_prints=True)
def historical_backfill_flow(
    start_year: int = 2024,
    end_year: int = 2025,
    start_month: int = 1,
    end_month: int = 12
):
    """Orquestra o download e merge de múltiplos anos/meses de dados históricos."""
    print(f"Iniciando Backfill Histórico de {start_year} até {end_year} (Meses: {start_month} a {end_month})...")
    
    monthly_dfs: list[pd.DataFrame] = []

    now_utc = datetime.now(tz=timezone.utc)  # noqa: UP017
    current_year = now_utc.year
    current_month = now_utc.month

    for year in range(start_year, end_year + 1):
        # Permite selecionar meses específicos no primeiro/último ano do intervalo
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12

        for month in range(m_start, m_end + 1):
            # Não tentar processar meses futuros em relação ao ano atual
            if year == current_year and month > current_month:
                            print(f"Mês {month:02d}/{year} no futuro. Avançando para o próximo ano...")
                            break # Interrompe os meses e avança no loop de anos (se houver)
                        
            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{last_day}"

            print(f"Processando: {year}-{month:02d}...")

            try:
                df_weather = extract_weather_task(start_date=start_date, end_date=end_date)
                df_energy = extract_energy_task(year=year, month=month)

                if df_weather is not None and df_energy is not None:
                    df_merged = merge_datasets_task(df_weather=df_weather, df_energy=df_energy)
                    monthly_dfs.append(df_merged)
                else:
                    print(f"Dados ausentes para {year}-{month:02d}. Pulando...")

            except Exception as e:  # noqa: BLE001
                print(f"Erro ao processar {year}-{month:02d}: {e}")
                continue

    if not monthly_dfs:
        raise RuntimeError("Nenhum dado foi extraído com sucesso durante o backfill.")

    print("Consolidando todos os meses em um único DataFrame...")
    df_full = pd.concat(monthly_dfs, ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Garante que a coluna 'date' seja timezone-aware em UTC
    df_full["date"] = pd.to_datetime(df_full["date"], utc=True)

    # Parâmetro de corte temporal: não permite registros fora do intervalo definido
    # evita que registros das ultimas 3 horas do ano sejam negligenciados (UTC-3 Brasília) 
    cutoff_limit = pd.Timestamp(f"{end_year + 1}-01-01 03:00:00", tz="utc")
    df_full = df_full[df_full["date"] < cutoff_limit]

    # Remove duplicatas mantendo ordenação
    df_full = df_full.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print("Aplicando Engenharia de Features (Física e Sazonalidade)...")
    df_full = transform_features_task(df_full)

    print(f"Backfill concluído! Total de registros: {len(df_full):,}")

    # Caminhos para salvamento dos Dados
    key = f"gold/dataset_renewable_energy_{start_year}_{start_month:02d}_{end_year}_{end_month:02d}.parquet"
    local_path = f"data/dataset_renewable_energy_{start_year}_{start_month:02d}_{end_year}_{end_month:02d}.parquet"
        
    # Prepara o diretório local antecipadamente para o fallback
    os.makedirs("data", exist_ok=True)

    final_path = save_dataset_to_lake_or_local(df_full, key, local_path)

    return final_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa o backfill histórico de dados eólicos e climáticos."
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Ano inicial da extração (ex: 2024)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Ano final da extração (ex: 2025)"
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=1,
        help="Mês inicial (1 a 12)"
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=12,
        help="Mês final (1 a 12)"
    )

    args = parser.parse_args()

    print(f"Iniciando Backfill: {args.start_year}/{args.start_month:02d} até {args.end_year}/{args.end_month:02d}")

    # Executa o fluxo do Prefect passando os argumentos do terminal
    historical_backfill_flow(
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month
    )

# ==============================================================================
# EXEMPLOS DE EXECUÇÃO VIA TERMINAL BASH
# ==============================================================================
# Pré-requisito: Garantir que o servidor local do Prefect está rodando
# ~/renewable-energy-mlops$ poetry run prefect server start > prefect.log 2>&1 &
#
# ------------------------------------------------------------------------------
# MODO PADRÃO (Executa a função que estiver descomentada no bloco __main__)
# ------------------------------------------------------------------------------
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.backfill_flow
#
# ------------------------------------------------------------------------------
# MODO PARAMETRIZADO (via argparse)
# ------------------------------------------------------------------------------
# Caso I (2025):
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.backfill_flow --start-year 2025 --end-year 2025
#
# Caso II (2024-2025 - Treino):
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.backfill_flow --start-year 2024 --end-year 2025
#
# Caso III (2026 Jan-Ago - OOT):
# ~/renewable-energy-mlops$ poetry run python -m energy_mlops.pipelines.backfill_flow --start-year 2026 --end-year 2026 --start-month 1 --end-month 8
# ==============================================================================