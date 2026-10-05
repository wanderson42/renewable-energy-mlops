from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_mlops.pipelines import monitoring_flow as monitoring
from energy_mlops.pipelines import monitoring_ingestion as ingestion


def frame(start, periods):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=periods, freq="h"),
        "temperature_2m": 25., "wind_speed_100m": 10.,
        "wind_direction_100m": 180.,
    })


def test_window_distinguishes_complete_and_partial_month():
    complete = frame("2026-09-01 03:00", 720)
    assert monitoring.describe_window(complete)["window_status"] == "COMPLETE_MONTH"
    partial = frame("2026-10-01 03:00", 72)
    assert monitoring.describe_window(partial)["window_status"] == "PARTIAL_WINDOW"
    gap = complete.drop(index=20)
    assert monitoring.describe_window(gap)["missing_hours_in_observed_interval"] == 1
    assert monitoring.describe_window(gap)["window_status"] == "PARTIAL_WINDOW"


def test_window_rejects_duplicates():
    data = frame("2026-10-01 03:00", 2)
    with pytest.raises(ValueError, match="únicos"):
        monitoring.describe_window(pd.concat([data, data]))


def test_gold_features_are_preserved_without_rebuilding():
    data = frame("2026-09-01 03:00", 2)
    for feature in monitoring.get_model_feature_columns():
        if feature not in data:
            data[feature] = 42.
    data["capacidade_mw"] = 100.
    with patch.object(monitoring, "generate_wind_and_time_features") as build:
        result = monitoring.prepare_feature_snapshot(data)
    build.assert_not_called()
    pd.testing.assert_frame_equal(result, data)


def test_partial_ingestion_uses_previous_hours_without_scoring_them():
    context = frame("2026-10-01 01:00", 2)
    context["wind_speed_100m"] = [3., 6.]
    weather = frame("2026-10-01 03:00", 3)
    weather["wind_speed_100m"] = [12., 15., 18.]
    energy = pd.DataFrame({"date": weather["date"].iloc[:2], "wind_generation_mw": [100., 200.]})
    start = pd.Timestamp("2026-10-01 03:00")
    result = ingestion.build_partial_snapshot(weather, energy, context, start, start + pd.Timedelta(hours=3))
    assert len(result) == 3
    assert result["wind_speed_roll_mean_3h"].iloc[0] == pytest.approx(7.)
    assert result["wind_generation_mw"].isna().sum() == 1
    assert result["date"].min() == start


def test_partial_ingestion_rejects_missing_context():
    start = pd.Timestamp("2026-10-01 03:00")
    with pytest.raises(ValueError, match="duas horas"):
        ingestion.build_partial_snapshot(frame(start, 3), None, frame(start - pd.Timedelta(hours=1), 1), start, start + pd.Timedelta(hours=3))


@pytest.mark.parametrize("partial,truth,drift,requested,expected_calls", [
    (True, True, True, False, 0),
    (True, True, True, True, 0),
    (False, False, True, True, 0),
    (False, True, True, True, 1),
    (False, True, False, True, 0),
])
def test_ct_requires_explicit_request_complete_month_and_truth(partial, truth, drift, requested, expected_calls):
    ref = frame("2026-09-01 03:00", 720)
    cur = frame("2026-10-01 03:00", 72 if partial else 744)
    context = {"version": "17", "run_id": "run17", "baseline_nmae_pct": 6.6788, "training_end": "2026-09-01 02:00"}
    X_ref = pd.DataFrame({"feature": [1.] * len(ref)})
    X_cur = pd.DataFrame({"feature": [1.] * len(cur)})
    cur["capacidade_mw"] = 100.
    if truth:
        cur["wind_generation_mw"] = 50.
    with (
        patch.object(monitoring, "get_run_logger", return_value=MagicMock()),
        patch.object(monitoring, "resolve_monitoring_model", return_value=context),
        patch.object(monitoring, "fetch_monitoring_data", return_value=(ref, cur)),
        patch.object(monitoring, "prepare_monitoring_features", return_value=(X_ref, X_cur, cur)),
        patch.object(monitoring, "generate_evidently_report", return_value=("html", drift, .7 if drift else 0.)),
        patch.object(monitoring, "evaluate_performance_drift", return_value=(False, 6.6788, 7., .3212)) as evaluate,
        patch.object(monitoring, "persist_generation_comparison", return_value={}),
        patch.object(monitoring, "save_report_to_s3") as save,
        patch.object(monitoring.s3fs, "S3FileSystem", return_value=MagicMock()),
        patch.object(monitoring, "continuous_training_pipeline") as train,
    ):
        result = monitoring.batch_monitoring_pipeline.fn(trigger_training=requested)
    assert train.call_count == expected_calls
    assert result["model"]["version"] == "17"
    assert save.call_count == 2
    assert evaluate.call_count == int(truth)
    if truth:
        assert evaluate.call_args.args[2] is context
