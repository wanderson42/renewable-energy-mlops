import numpy as np
import pandas as pd

from energy_mlops.data.reference_data import (
    load_bahia_wind_capacity_checkpoints,
)
from energy_mlops.data.schema import CapacityCheckpointSchema


def _normalize_datetime_series(
    series: pd.Series,
) -> pd.Series:
    """
    Normaliza uma Series temporal para datetime64[ns] sem timezone.

    O pipeline trabalha internamente com timestamps timezone-naive
    para manter compatibilidade entre ONS, Open-Meteo e referências
    externas.
    """

    result = pd.to_datetime(
        series,
        errors="raise",
    )

    if result.dt.tz is not None:
        result = result.dt.tz_convert("UTC").dt.tz_localize(None)

    return result.astype(
        "datetime64[ns]"
    )


def _attach_wind_capacity(
    df: pd.DataFrame,
    capacity_checkpoints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Associa a capacidade eólica conhecida em cada instante.

    A capacidade é tratada como uma step function:

        available_from ─────────────► próximo checkpoint

    Não existe interpolação entre boletins.

    O merge utiliza exclusivamente `available_from`, garantindo que
    um checkpoint nunca seja utilizado antes da data em que passou
    a estar disponível para o pipeline.

    Datas anteriores ao primeiro checkpoint disponível são
    rejeitadas explicitamente. Não é realizado bfill, pois isso
    introduziria informação futura.
    """

    if capacity_checkpoints is None:
        checkpoints = (
            load_bahia_wind_capacity_checkpoints()
        )
    else:
        checkpoints = (
            CapacityCheckpointSchema.validate(
                capacity_checkpoints.copy()
            )
        )

    checkpoints = checkpoints.copy()

    checkpoints["available_from"] = (
        _normalize_datetime_series(
            checkpoints["available_from"]
        )
    )

    checkpoints = (
        checkpoints[
            [
                "available_from",
                "capacity_mw",
            ]
        ]
        .sort_values("available_from")
        .reset_index(drop=True)
    )

    if checkpoints.empty:
        raise ValueError(
            "Nenhum checkpoint de capacidade "
            "eólica está disponível."
        )

    # Se o DataFrame já passou anteriormente pelo pipeline,
    # recalculamos a capacidade a partir da fonte canônica.
    df = df.drop(
        columns=[
            "capacidade_mw",
            "target_fc",
        ],
        errors="ignore",
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        df,
        checkpoints,
        left_on="date",
        right_on="available_from",
        direction="backward",
        allow_exact_matches=True,
    )

    missing_capacity = (
        merged["capacity_mw"].isna()
    )

    if missing_capacity.any():
        first_available = (
            checkpoints[
                "available_from"
            ].min()
        )

        first_missing = (
            merged.loc[
                missing_capacity,
                "date",
            ].min()
        )

        last_missing = (
            merged.loc[
                missing_capacity,
                "date",
            ].max()
        )

        raise ValueError(
            "Não existe checkpoint causal de capacidade "
            "eólica disponível para parte do dataset. "
            f"Intervalo sem cobertura: "
            f"{first_missing} até {last_missing}. "
            f"Primeiro checkpoint disponível em: "
            f"{first_available}."
        )

    merged.rename(
        columns={
            "capacity_mw": "capacidade_mw",
        },
        inplace=True,
    )

    merged.drop(
        columns=["available_from"],
        inplace=True,
    )

    return merged


def generate_wind_and_time_features(
    df: pd.DataFrame,
    use_exogenous_lags: bool = True,
    *,
    capacity_checkpoints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Gera as features utilizadas pelo modelo de geração eólica.

    A capacidade eólica da Bahia é obtida dos checkpoints
    INFOVENTO validados pelo contrato CapacityCheckpointSchema.

    `capacity_checkpoints` pode ser injetado explicitamente em
    testes; quando omitido, a referência canônica versionada no
    projeto é carregada por reference_data.py.
    """

    df_feat = df.copy()

    if "date" not in df_feat.columns:
        raise ValueError(
            "Coluna obrigatória 'date' ausente."
        )

    # ------------------------------------------------------------------
    # Normalização temporal
    # ------------------------------------------------------------------
    df_feat["date"] = (
        _normalize_datetime_series(
            df_feat["date"]
        )
    )

    df_feat.sort_values(
        "date",
        inplace=True,
    )

    df_feat.reset_index(
        drop=True,
        inplace=True,
    )

    # ------------------------------------------------------------------
    # Capacidade eólica instalada — Bahia
    #
    # Fonte canônica:
    # ABEEólica / INFOVENTO
    #
    # Sem interpolação.
    # Sem backward fill.
    # Somente o último checkpoint causalmente disponível.
    # ------------------------------------------------------------------
    df_feat = _attach_wind_capacity(
        df_feat,
        capacity_checkpoints=capacity_checkpoints,
    )

    # ------------------------------------------------------------------
    # Target normalizado
    # ------------------------------------------------------------------
    if "wind_generation_mw" in df_feat.columns:
        df_feat["target_fc"] = (
            df_feat["wind_generation_mw"]
            / df_feat["capacidade_mw"]
        )

        # Mantido nesta refatoração para preservar
        # o contrato atual do target.
        df_feat["target_fc"] = (
            df_feat["target_fc"]
            .clip(
                lower=0.0,
                upper=1.0,
            )
        )

    # ------------------------------------------------------------------
    # Relação física vento / temperatura
    # ------------------------------------------------------------------
    df_feat["wind_temp_ratio"] = (
        df_feat["wind_speed_100m"]
        / (
            df_feat["temperature_2m"]
            + 0.1
        )
    )

    # ------------------------------------------------------------------
    # Features temporais cíclicas
    # ------------------------------------------------------------------
    hour = df_feat["date"].dt.hour
    month = df_feat["date"].dt.month

    df_feat["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df_feat["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    df_feat["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df_feat["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # ------------------------------------------------------------------
    # Rolling exógeno causal
    # ------------------------------------------------------------------
    if (
        use_exogenous_lags
        and "wind_speed_100m"
        in df_feat.columns
    ):
        wind_speed_time_indexed = (
            df_feat
            .set_index("date")[
                "wind_speed_100m"
            ]
        )

        df_feat[
            "wind_speed_roll_mean_3h"
        ] = (
            wind_speed_time_indexed
            .rolling(
                window="3h",
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )

    return df_feat