"""Tests for src.baselines.run_baselines (offline; foundation models are faked)."""

import numpy as np
import pandas as pd
import pytest

from src.baselines.run_baselines import ar1, history_as_of, no_change, run
from src.gatekeeper import Gatekeeper


def _history(n=400, end="2025-08-01", seed=1):
    idx = pd.date_range(end=end, periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    return pd.Series(4.5 + np.cumsum(rng.normal(0, 0.1, n)) * 0.2, index=idx)


def test_no_change_point_is_last_value_and_interval_brackets_it():
    hist = _history()
    point, lo, hi = no_change(hist, 1)
    assert point == hist.iloc[-1]
    assert lo <= point <= hi


def test_ar1_is_deterministic_and_interval_widens_with_horizon():
    hist = _history()
    a = ar1(hist, 1)
    assert a == ar1(hist, 1)
    _, lo1, hi1 = a
    _, lo2, hi2 = ar1(hist, 2)
    assert lo1 < a[0] < hi1
    assert (hi2 - lo2) > (hi1 - lo1)


def test_pandemic_jump_does_not_blow_up_ar1():
    hist = _history(end="2023-12-01")
    hist.loc["2020-04-01"] += 10  # a 2020-style spike
    point, lo, hi = ar1(hist, 1)
    assert abs(point - hist.iloc[-1]) < 0.5 and (hi - lo) < 1.0


@pytest.fixture
def gk():
    rows = [("UNRATE", f"{d:%Y-%m-%d}", f"{d + pd.DateOffset(months=1, days=4):%Y-%m-%d}", 4.0 + 0.01 * i)
            for i, d in enumerate(pd.date_range("1990-01-01", "2025-08-01", freq="MS"))]
    rows.append(("UNRATE", "2025-09-01", "2025-11-20", 4.4))  # late release
    return Gatekeeper(pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"]))


def test_run_writes_run_log_records_with_correct_horizon(gk):
    episodes = pd.DataFrame({"episode": ["2025-09-20", "2025-11-20"], "target_month": ["2025-09", "2025-11"]})
    fake = lambda hist, h: (float(hist.iloc[-1]) + h, 0.0, 9.0)
    recs = run(gk, episodes, {"no_change": no_change, "fake_fm": fake})
    assert len(recs) == 4
    by = {(r["target_month"], r["system"]): r for r in recs}
    assert by[("2025-09", "fake_fm")]["horizon_months"] == 1   # last visible: Aug
    assert by[("2025-11", "fake_fm")]["horizon_months"] == 2   # last visible: Sep (out Nov 20 a.m.)
    for r in recs:
        assert {"episode", "target_month", "system", "forecast", "lo80", "hi80"} <= set(r)
        assert r["tokens"] == 0 and r["cited"] == []


def test_history_never_includes_the_target(gk):
    hist = history_as_of(gk, "2025-09-20")
    assert hist.index.max() == pd.Timestamp("2025-08-01")
