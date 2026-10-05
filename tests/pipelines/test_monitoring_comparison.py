from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_mlops.pipelines.monitoring_flow import (
    build_hourly_comparison,
    summarize_daily_comparison,
)


def inputs():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-10-01T03:00Z", periods=96, freq="h"),
        "capacidade_mw": 100.,
        "wind_generation_mw": [40.] * 72 + [np.nan] * 24,
    })
    model = MagicMock()
    model.feature_names_in_ = np.array(["feature"])
    model.predict.return_value = np.full(96, .5)
    X = pd.DataFrame({"feature": np.ones(96)})
    return model, X, frame, {"version": "17", "run_id": "run17"}


def test_hourly_retains_missing_truth_and_daily_coverage():
    model, X, frame, context = inputs()
    result = build_hourly_comparison(model, X, frame, context, "execution")
    assert len(result) == 96
    assert result.truth_valid.sum() == 72
    assert result.observed_mw.iloc[72:].isna().all()
    assert result.error_mw.iloc[72:].isna().all()
    assert result.error_mw.iloc[:72].eq(10).all()
    daily = summarize_daily_comparison(result)
    assert daily.horas_truth_validas.tolist() == [24, 24, 24, 0]
    assert daily.nmae_pct.iloc[:3].eq(10).all()
    assert pd.isna(daily.nmae_pct.iloc[3])
    assert daily.status.iloc[3] == "SEM_TRUTH"


def test_zero_truth_is_valid_but_negative_truth_is_not():
    model, X, frame, context = inputs()
    frame.loc[0, "wind_generation_mw"] = 0
    frame.loc[1, "wind_generation_mw"] = -1
    result = build_hourly_comparison(model, X, frame, context, "execution")
    assert result.truth_valid.iloc[0]
    assert result.error_mw.iloc[0] == 50
    assert not result.truth_valid.iloc[1]
    assert pd.isna(result.observed_mw.iloc[1])


def test_feature_alignment_and_nonfinite_predictions_are_rejected():
    model, X, frame, context = inputs()
    with pytest.raises(ValueError, match="alinhados"):
        build_hourly_comparison(model, X.set_axis(range(1, 97)), frame, context, "execution")
    model.predict.return_value[0] = np.nan
    with pytest.raises(ValueError, match="inválidas"):
        build_hourly_comparison(model, X, frame, context, "execution")
