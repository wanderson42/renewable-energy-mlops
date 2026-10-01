from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_mlops.data.feature_utils import (
    get_model_feature_columns,
)
from energy_mlops.data.reference_data import (
    load_bahia_wind_capacity_checkpoints,
)


@dataclass(frozen=True)
class SnapshotAuditResult:
    rows: int
    columns: int

    start_date: pd.Timestamp
    end_date: pd.Timestamp

    missing_timestamps: int
    gap_count: int

    feature_count: int

    first_capacity_date: pd.Timestamp
    last_capacity_mw: float

    target_fc_min: float
    target_fc_max: float
    max_target_reconstruction_diff: float


def audit_gold_snapshot(
    df: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> SnapshotAuditResult:
    """
    Valida invariantes estruturais, temporais,
    físicas e causais de um dataset Gold.

    Raises
    ------
    ValueError
        Quando qualquer contrato obrigatório
        do snapshot é violado.
    """

    if df.empty:
        raise ValueError(
            "Snapshot Gold vazio."
        )

    required_columns = {
        "date",
        "wind_generation_mw",
        "capacidade_mw",
        "target_fc",
    }

    missing_required = (
        required_columns
        - set(df.columns)
    )

    if missing_required:
        raise ValueError(
            "Snapshot Gold sem colunas obrigatórias: "
            f"{sorted(missing_required)}"
        )

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )

    if result["date"].dt.tz is not None:
        result["date"] = (
            result["date"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

    # --------------------------------------------------
    # Temporal integrity
    # --------------------------------------------------

    if not result[
        "date"
    ].is_monotonic_increasing:
        raise ValueError(
            "Snapshot Gold não está "
            "ordenado cronologicamente."
        )

    duplicate_dates = int(
        result[
            "date"
        ].duplicated().sum()
    )

    if duplicate_dates:
        raise ValueError(
            "Snapshot Gold possui "
            f"{duplicate_dates} timestamps duplicados."
        )

    expected_index = pd.date_range(
        start=result["date"].min(),
        end=result["date"].max(),
        freq="1h",
    )

    missing_dates = (
        expected_index.difference(
            pd.DatetimeIndex(
                result["date"]
            )
        )
    )

    if len(missing_dates):
        missing_frame = pd.DataFrame(
            {
                "date": missing_dates,
            }
        )

        missing_frame["group"] = (
            missing_frame["date"]
            .diff()
            .ne(
                pd.Timedelta(hours=1)
            )
            .cumsum()
        )

        gap_count = int(
            missing_frame[
                "group"
            ].nunique()
        )

    else:
        gap_count = 0

    # --------------------------------------------------
    # Feature contract
    # --------------------------------------------------

    features = (
        get_model_feature_columns()
    )

    missing_features = (
        set(features)
        - set(result.columns)
    )

    if missing_features:
        raise ValueError(
            "Snapshot Gold não satisfaz "
            "o contrato de features: "
            f"{sorted(missing_features)}"
        )

    if result[
        features
    ].isna().any().any():
        raise ValueError(
            "Snapshot Gold possui NaN "
            "nas features do modelo."
        )

    # --------------------------------------------------
    # Targets / physical values
    # --------------------------------------------------

    target_columns = [
        "wind_generation_mw",
        "capacidade_mw",
        "target_fc",
    ]

    if result[
        target_columns
    ].isna().any().any():
        raise ValueError(
            "Snapshot Gold possui NaN "
            "nas variáveis de target/capacidade."
        )

    if (
        result[
            "wind_generation_mw"
        ]
        < 0.0
    ).any():
        raise ValueError(
            "Snapshot Gold possui geração negativa."
        )

    if (
        result[
            "capacidade_mw"
        ]
        <= 0.0
    ).any():
        raise ValueError(
            "Snapshot Gold possui capacidade "
            "não positiva."
        )

    if not result[
        "target_fc"
    ].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            "Snapshot Gold possui target_fc "
            "fora de [0, 1]."
        )

    expected_target_fc = (
        result[
            "wind_generation_mw"
        ]
        / result[
            "capacidade_mw"
        ]
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    max_target_diff = float(
        np.abs(
            result["target_fc"]
            - expected_target_fc
        ).max()
    )

    if max_target_diff > tolerance:
        raise ValueError(
            "target_fc não corresponde a "
            "wind_generation_mw / capacidade_mw. "
            f"Máxima diferença: {max_target_diff}."
        )

    # --------------------------------------------------
    # Capacity provenance / causality
    # --------------------------------------------------

    checkpoints = (
        load_bahia_wind_capacity_checkpoints()
        .sort_values(
            "available_from"
        )
        .reset_index(drop=True)
    )

    checkpoints = checkpoints.copy()

    checkpoints[
        "available_from"
    ] = pd.to_datetime(
        checkpoints[
            "available_from"
        ]
    )

    first_available = (
        checkpoints[
            "available_from"
        ].min()
    )

    if (
        result["date"]
        < first_available
    ).any():
        raise ValueError(
            "Snapshot Gold contém observações "
            "anteriores ao primeiro checkpoint "
            "causal de capacidade."
        )

    expected_capacity = (
        pd.merge_asof(
            result[
                ["date"]
            ].sort_values(
                "date"
            ),
            checkpoints[
                [
                    "available_from",
                    "capacity_mw",
                ]
            ].sort_values(
                "available_from"
            ),
            left_on="date",
            right_on="available_from",
            direction="backward",
            allow_exact_matches=True,
        )[
            "capacity_mw"
        ]
        .to_numpy()
    )

    observed_capacity = (
        result[
            "capacidade_mw"
        ]
        .to_numpy()
    )

    if not np.allclose(
        observed_capacity,
        expected_capacity,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "capacidade_mw não corresponde "
            "à step function causal INFOVENTO."
        )

    return SnapshotAuditResult(
        rows=len(result),
        columns=len(result.columns),
        start_date=result["date"].min(),
        end_date=result["date"].max(),
        missing_timestamps=len(
            missing_dates
        ),
        gap_count=gap_count,
        feature_count=len(features),
        first_capacity_date=(
            first_available
        ),
        last_capacity_mw=float(
            result[
                "capacidade_mw"
            ].iloc[-1]
        ),
        target_fc_min=float(
            result[
                "target_fc"
            ].min()
        ),
        target_fc_max=float(
            result[
                "target_fc"
            ].max()
        ),
        max_target_reconstruction_diff=(
            max_target_diff
        ),
    )