from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


GOVERNANCE_POLICY_VERSION = "same_oot_v1"


def _stringify(value: Any) -> str:
    """Normaliza valores para persistência como tag MLflow."""

    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def summarize_oot(
    df_oot: pd.DataFrame,
    *,
    dataset_name: str | None,
) -> dict[str, str]:
    """Resume o OOT usado pelo Quality Gate em metadados auditáveis."""

    if "date" not in df_oot.columns:
        raise ValueError(
            "O OOT do Quality Gate deve conter a coluna 'date'."
        )

    if df_oot.empty:
        raise ValueError(
            "O OOT do Quality Gate não pode estar vazio."
        )

    dates = pd.to_datetime(
        df_oot["date"],
        errors="raise",
        utc=True,
    )

    return {
        "governance_oot_dataset": _stringify(dataset_name),
        "governance_oot_rows": str(len(df_oot)),
        "governance_oot_start_utc": dates.min().isoformat(),
        "governance_oot_end_utc": dates.max().isoformat(),
    }


def build_automatic_governance_tags(
    *,
    df_oot: pd.DataFrame,
    oot_dataset: str | None,
    model_name: str,
    model_alias: str,
    automatic_decision: str,
    decision_reason: str,
    previous_champion_version: str | None,
    previous_champion_run_id: str | None,
    challenger_version: str,
    challenger_run_id: str,
    champion_num_features: int | None,
    challenger_num_features: int,
    champion_mae_mw: float | None,
    champion_nmae_pct: float | None,
    challenger_mae_mw: float | None,
    challenger_nmae_pct: float | None,
    nmae_simplification_tolerance_pp: float,
    decided_at_utc: datetime | None = None,
) -> dict[str, str]:
    """
    Constrói o contrato de metadados da decisão automática de governança.

    Nesta etapa do projeto, ``final_decision`` é igual à decisão automática.
    Um futuro manual override deve preservar ``automatic_decision`` e alterar
    somente os campos finais/origem da decisão.
    """

    if automatic_decision not in {"PROMOTE", "REJECT"}:
        raise ValueError(
            "automatic_decision deve ser 'PROMOTE' ou 'REJECT'."
        )

    if decided_at_utc is None:
        decided_at_utc = datetime.now(timezone.utc)
    elif decided_at_utc.tzinfo is None:
        raise ValueError(
            "decided_at_utc deve possuir timezone explícito."
        )

    mae_delta_mw = None
    nmae_delta_pp = None

    if (
        champion_mae_mw is not None
        and challenger_mae_mw is not None
    ):
        mae_delta_mw = (
            challenger_mae_mw
            - champion_mae_mw
        )

    if (
        champion_nmae_pct is not None
        and challenger_nmae_pct is not None
    ):
        nmae_delta_pp = (
            challenger_nmae_pct
            - champion_nmae_pct
        )

    tags = {
        "governance_event": "true",
        "governance_policy_version": GOVERNANCE_POLICY_VERSION,
        "governance_decided_at_utc": (
            decided_at_utc
            .astimezone(timezone.utc)
            .isoformat()
        ),
        "governance_decision_source": "AUTOMATIC",
        "governance_automatic_decision": automatic_decision,
        "governance_final_decision": automatic_decision,
        "governance_decision_reason": decision_reason,
        "governance_model_name": model_name,
        "governance_model_alias": model_alias,
        "governance_previous_champion_version": _stringify(
            previous_champion_version
        ),
        "governance_previous_champion_run_id": _stringify(
            previous_champion_run_id
        ),
        "governance_challenger_version": str(challenger_version),
        "governance_challenger_run_id": challenger_run_id,
        "governance_champion_num_features": _stringify(
            champion_num_features
        ),
        "governance_challenger_num_features": str(
            challenger_num_features
        ),
        "governance_champion_mae_mw_same_oot": _stringify(
            champion_mae_mw
        ),
        "governance_champion_nmae_pct_same_oot": _stringify(
            champion_nmae_pct
        ),
        "governance_challenger_mae_mw_same_oot": _stringify(
            challenger_mae_mw
        ),
        "governance_challenger_nmae_pct_same_oot": _stringify(
            challenger_nmae_pct
        ),
        "governance_mae_delta_mw_same_oot": _stringify(
            mae_delta_mw
        ),
        "governance_nmae_delta_pp_same_oot": _stringify(
            nmae_delta_pp
        ),
        "governance_simplification_tolerance_pp": str(
            nmae_simplification_tolerance_pp
        ),
    }

    tags.update(
        summarize_oot(
            df_oot,
            dataset_name=oot_dataset,
        )
    )

    return tags


def persist_governance_tags(
    client: Any,
    *,
    run_id: str,
    tags: dict[str, str],
) -> None:
    """Persiste o evento de governança na Run do Challenger."""

    for key, value in tags.items():
        client.set_tag(
            run_id,
            key,
            value,
        )
