import logging

import pandas as pd
import pandera.errors as pa_errors

from energy_mlops.data.schema import EnergySchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_ons_wind_generation(year: int = 2025, month: int = 1) -> pd.DataFrame | None:
    """
    Extrai dados de geração Eólica do subsistema Nordeste do ONS.
    Valida os tipos e limites com Pandera.
    Suporta o novo padrão de arquivos mensais do ONS.
    """
    # URL atualizada com a formatação correta de ano e mês (ex: 2025_01)
    url = f"https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA-2_{year}_{month:02d}.parquet"
    
    logger.info(f"Baixando dados do ONS via Parquet para {year}-{month:02d}...")
    
    try:
        colunas_interesse = [
            "din_instante",       # Data e hora (YYYY-MM-DD HH:MM:SS)
            "nom_estado",         # Estado (ex: BAHIA, PIAUI,)
            "nom_tipousina",      # Tipo de Usina (ex: EOLIELÉTRICA, HIDROELÉTRICA)
            "val_geracao"         # Geração em MW
        ]
        
        df = pd.read_parquet(url, columns=colunas_interesse)
        logger.info(f"Dados brutos lidos da nuvem: {len(df)} registros.")

        # Filtro: Apenas Nordeste e Eoliétrica
        mask = (df["nom_estado"] == "BAHIA") & (df["nom_tipousina"] == "EOLIELÉTRICA")
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            logger.warning(f"Filtro resultou em zero linhas para {year}-{month:02d}.")
            return None

        # Força a conversão para float numérico
        df_filtered["val_geracao"] = (
            pd.to_numeric(
                df_filtered["val_geracao"].astype(str).str.replace(",", "."),
                errors="coerce"
            )
            .fillna(0.0)
        )

        logger.info("Agregando geração total horária...")
        df_grouped = df_filtered.groupby("din_instante")["val_geracao"].sum().reset_index()
        
        df_grouped.rename(columns={
            "din_instante": "date",
            "val_geracao": "wind_generation_mw"
        }, inplace=True)
        
        # Conversão de fuso horário (Brasília -> UTC)
        df_grouped["date"] = pd.to_datetime(df_grouped["date"]) \
                               .dt.tz_localize("America/Sao_Paulo") \
                               .dt.tz_convert("UTC")
        
        logger.info(f"Extração concluída. Validando contrato de dados ({len(df_grouped)} horas)...")
        
        try:
            df_validated = EnergySchema.validate(df_grouped)
            logger.info("Validação do Pandera concluída com sucesso. Dados íntegros.")
            return df_validated
        except pa_errors.SchemaError as exc:
            logger.error(f"Falha no contrato de dados (Pandera): {exc}")
            return None

    except Exception as e:  # noqa: BLE001
        logger.error(f"Erro ao extrair dados do ONS: {e}")
        return None


if __name__ == "__main__":
    # Testando Janeiro de 2025
    df_energia = fetch_ons_wind_generation(year=2025, month=1)
    if df_energia is not None:
        print("\nAmostra dos dados de Geração Eólica (Nordeste):\n")
        print(df_energia.head())
        print(df_energia.info())