from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from energy_mlops.models.governance import (
    GOVERNANCE_POLICY_VERSION,
    build_automatic_governance_tags,
    persist_governance_tags,
    summarize_oot,
)


def _make_oot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-09-01 03:00:00",
                    "2026-10-01 02:00:00",
                ]
            )
        }
    )


def test_summarize_oot_records_dataset_range_and_rows():
    summary = summarize_oot(
        _make_oot(),
        dataset_name="oot_test.parquet",
    )

    assert summary == {
        "governance_oot_dataset": "oot_test.parquet",
        "governance_oot_rows": "2",
        "governance_oot_start_utc": (
            "2026-09-01T03:00:00+00:00"
        ),
        "governance_oot_end_utc": (
            "2026-10-01T02:00:00+00:00"
        ),
    }


def test_build_automatic_governance_tags_preserves_same_oot_context():
    decided_at = datetime(
        2026,
        10,
        2,
        18,
        0,
        tzinfo=UTC,
    )

    tags = build_automatic_governance_tags(
        df_oot=_make_oot(),
        oot_dataset="oot_test.parquet",
        model_name="ensemble_lgb_xgb_rf_bahia",
        model_alias="champion",
        automatic_decision="PROMOTE",
        decision_reason="MAE_IMPROVED_SAME_OOT",
        previous_champion_version="10",
        previous_champion_run_id="champion-run-10",
        challenger_version="20",
        challenger_run_id="challenger-run-20",
        champion_num_features=9,
        challenger_num_features=9,
        champion_mae_mw=900.0,
        champion_nmae_pct=7.6,
        challenger_mae_mw=800.0,
        challenger_nmae_pct=6.8,
        nmae_simplification_tolerance_pp=0.05,
        decided_at_utc=decided_at,
    )

    assert tags["governance_event"] == "true"
    assert (
        tags["governance_policy_version"]
        == GOVERNANCE_POLICY_VERSION
    )
    assert tags["governance_decision_source"] == "AUTOMATIC"
    assert tags["governance_automatic_decision"] == "PROMOTE"
    assert tags["governance_final_decision"] == "PROMOTE"
    assert (
        tags["governance_previous_champion_version"]
        == "10"
    )
    assert (
        tags["governance_previous_champion_run_id"]
        == "champion-run-10"
    )
    assert tags["governance_challenger_version"] == "20"
    assert (
        tags["governance_challenger_run_id"]
        == "challenger-run-20"
    )
    assert (
        float(tags["governance_mae_delta_mw_same_oot"])
        == pytest.approx(-100.0)
    )
    assert (
        float(tags["governance_nmae_delta_pp_same_oot"])
        == pytest.approx(-0.8)
    )
    assert (
        tags["governance_decided_at_utc"]
        == "2026-10-02T18:00:00+00:00"
    )


def test_build_automatic_governance_tags_supports_first_champion():
    tags = build_automatic_governance_tags(
        df_oot=_make_oot(),
        oot_dataset=None,
        model_name="ensemble_lgb_xgb_rf_bahia",
        model_alias="champion",
        automatic_decision="PROMOTE",
        decision_reason="FIRST_CHAMPION",
        previous_champion_version=None,
        previous_champion_run_id=None,
        challenger_version="1",
        challenger_run_id="challenger-run-1",
        champion_num_features=None,
        challenger_num_features=9,
        champion_mae_mw=None,
        champion_nmae_pct=None,
        challenger_mae_mw=None,
        challenger_nmae_pct=None,
        nmae_simplification_tolerance_pp=0.05,
    )

    assert (
        tags["governance_previous_champion_version"]
        == "N/A"
    )
    assert (
        tags["governance_champion_mae_mw_same_oot"]
        == "N/A"
    )
    assert (
        tags["governance_mae_delta_mw_same_oot"]
        == "N/A"
    )
    assert tags["governance_oot_dataset"] == "N/A"


def test_build_automatic_governance_tags_rejects_invalid_decision():
    with pytest.raises(
        ValueError,
        match="automatic_decision",
    ):
        build_automatic_governance_tags(
            df_oot=_make_oot(),
            oot_dataset="oot.parquet",
            model_name="model",
            model_alias="champion",
            automatic_decision="UNKNOWN",
            decision_reason="invalid",
            previous_champion_version="10",
            previous_champion_run_id="run-10",
            challenger_version="20",
            challenger_run_id="run-20",
            champion_num_features=9,
            challenger_num_features=9,
            champion_mae_mw=900.0,
            champion_nmae_pct=7.6,
            challenger_mae_mw=800.0,
            challenger_nmae_pct=6.8,
            nmae_simplification_tolerance_pp=0.05,
        )


def test_persist_governance_tags_writes_all_tags():
    client = MagicMock()
    tags = {
        "governance_event": "true",
        "governance_final_decision": "REJECT",
    }

    persist_governance_tags(
        client,
        run_id="challenger-run-20",
        tags=tags,
    )

    assert client.set_tag.call_count == 2
    client.set_tag.assert_any_call(
        "challenger-run-20",
        "governance_event",
        "true",
    )
    client.set_tag.assert_any_call(
        "challenger-run-20",
        "governance_final_decision",
        "REJECT",
    )
