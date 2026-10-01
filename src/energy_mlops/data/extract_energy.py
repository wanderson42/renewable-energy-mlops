import logging

import pandas as pd

from energy_mlops.data.schema import (
    EnergySchema,
)

logger = logging.getLogger(__name__)


DEFAULT_STATE = "BAHIA"
DEFAULT_GENERATION_TYPE = (
    "EOLIELÉTRICA"
)

ONS_PARQUET_URL = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/"
    "dataset/geracao_usina_2_ho/"
    "GERACAO_USINA-2_{year}_{month:02d}.parquet"
)


def fetch_ons_wind_generation(
    year: int = 2025,
    month: int = 1,
    state: str = DEFAULT_STATE,
    generation_type: str = DEFAULT_GENERATION_TYPE,
) -> pd.DataFrame:
    """
    Extrai e agrega a geração eólica horária do ONS
    para uma unidade federativa.

    O default operacional do projeto é Bahia.

    O resultado é normalizado para:

        date
        wind_generation_mw

    e validado por EnergySchema.
    """

    if not 1 <= month <= 12:
        raise ValueError(
            "month deve estar entre 1 e 12."
        )

    state = (
        state.strip().upper()
    )

    generation_type = (
        generation_type
        .strip()
        .upper()
    )

    url = ONS_PARQUET_URL.format(
        year=year,
        month=month,
    )

    logger.info(
        "Baixando geração ONS para "
        "%04d-%02d (%s / %s).",
        year,
        month,
        state,
        generation_type,
    )

    columns = [
        "din_instante",
        "nom_estado",
        "nom_tipousina",
        "val_geracao",
    ]

    try:
        df = pd.read_parquet(
            url,
            columns=columns,
        )

        logger.info(
            "ONS: %s registros brutos lidos.",
            len(df),
        )

        state_values = (
            df["nom_estado"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        generation_values = (
            df["nom_tipousina"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        mask = (
            (state_values == state)
            & (
                generation_values
                == generation_type
            )
        )

        df_filtered = (
            df.loc[
                mask,
                [
                    "din_instante",
                    "val_geracao",
                ],
            ]
            .copy()
        )

        if df_filtered.empty:
            raise ValueError(
                "O filtro ONS não retornou "
                "registros para "
                f"{year}-{month:02d}, "
                f"state={state!r}, "
                "generation_type="
                f"{generation_type!r}."
            )

        generation_mw = pd.to_numeric(
            df_filtered[
                "val_geracao"
            ]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .str.replace(
                ",",
                ".",
                regex=False,
            ),
            errors="coerce",
        )

        df_filtered[
            "val_geracao"
        ] = generation_mw


        # --------------------------------------------------
        # Controle de completude horária
        # --------------------------------------------------
        hourly_quality = (
            df_filtered
            .groupby(
                "din_instante"
            )["val_geracao"]
            .agg(
                total_records="size",
                valid_records="count",
            )
        )

        hourly_quality[
            "missing_records"
        ] = (
            hourly_quality[
                "total_records"
            ]
            - hourly_quality[
                "valid_records"
            ]
        )

        incomplete_hours = (
            hourly_quality.index[
                hourly_quality[
                    "missing_records"
                ]
                > 0
            ]
        )

        if len(incomplete_hours) > 0:
            missing_count = int(
                hourly_quality.loc[
                    incomplete_hours,
                    "missing_records",
                ].sum()
            )

            logger.warning(
                "ONS: %s registros de geração ausentes "
                "em %s horas. "
                "As horas incompletas serão descartadas "
                "antes da agregação.",
                missing_count,
                len(incomplete_hours),
            )

            df_filtered = (
                df_filtered.loc[
                    ~df_filtered[
                        "din_instante"
                    ].isin(
                        incomplete_hours
                    )
                ]
                .copy()
            )


        if df_filtered.empty:
            raise ValueError(
                "Nenhuma hora completa permaneceu "
                "após o controle de qualidade ONS."
            )


        logger.info(
            "Agregando geração eólica "
            "apenas em horas completas."
        )

        df_grouped = (
            df_filtered
            .groupby(
                "din_instante",
                as_index=False,
            )["val_geracao"]
            .sum()
        )

        if df_grouped["val_geracao"].isna().any():
            raise ValueError(
                "A agregação ONS produziu "
                "valores horários ausentes."
            )

        df_grouped.rename(
            columns={
                "din_instante": "date",
                "val_geracao": (
                    "wind_generation_mw"
                ),
            },
            inplace=True,
        )

        df_grouped["date"] = (
            pd.to_datetime(
                df_grouped["date"],
                errors="raise",
            )
            .dt.tz_localize(
                "America/Sao_Paulo"
            )
            .dt.tz_convert("UTC")
        )

        validated = (
            EnergySchema.validate(
                df_grouped
            )
        )

        logger.info(
            "Extração ONS concluída: "
            "%s horas validadas.",
            len(validated),
        )

        return validated

    except Exception:
        logger.exception(
            "Falha ao extrair geração "
            "eólica do ONS."
        )
        raise


if __name__ == "__main__":
    df_energy = (
        fetch_ons_wind_generation(
            year=2025,
            month=1,
        )
    )

    print(
        "\nAmostra dos dados ONS:\n"
    )
    print(df_energy.head())

    print()
    print(df_energy.info())