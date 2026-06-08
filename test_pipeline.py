"""Unit tests for the HMM regime-detection pipeline.

Fast and self-contained: no network, no real dataset. Feature/split tests use
tiny hand-built frames with known answers; model tests use small synthetic data
drawn from well-separated Gaussians so fits are quick and deterministic-ish.

Run:  pytest -q
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    TRADING_DAYS,
    compute_features,
    load_ohlcv,
    to_observations,
    train_test_split,
)
from hmm_model import (
    _count_params,
    fit_hmm,
    regime_summary,
    select_n_states,
)

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------- #
# data_loader: feature engineering
# --------------------------------------------------------------------------- #

def _price_frame(prices, start="2000-01-03"):
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"price": prices}, index=idx)


def test_log_return_formula():
    df = _price_frame([100.0, 110.0, 99.0])
    out = compute_features(df, vol_window=2)
    # log_return_t = ln(P_t / P_{t-1}); first row is dropped (NaN).
    expected = np.log(np.array([110.0, 99.0]) / np.array([100.0, 110.0]))
    # After dropna(realized_vol needs 2 obs) only the last row survives here.
    assert out["log_return"].iloc[-1] == pytest.approx(expected[-1])


def test_realized_vol_is_annualized_rolling_std():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
    df = _price_frame(prices)
    out = compute_features(df, vol_window=20)
    # Recompute the last value independently from the raw log returns.
    lr = np.log(df["price"] / df["price"].shift(1))
    expected = lr.rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS)
    assert out["realized_vol"].iloc[-1] == pytest.approx(expected)
    assert (out["realized_vol"] > 0).all()


def test_compute_features_drops_warmup_nans():
    df = _price_frame(np.linspace(100, 120, 50))
    out = compute_features(df, vol_window=20)
    assert not out[["log_return", "realized_vol"]].isna().any().any()
    # First return + (vol_window-1) rolling warmup rows are dropped.
    assert len(out) == 50 - 20


# --------------------------------------------------------------------------- #
# data_loader: CSV loading
# --------------------------------------------------------------------------- #

def test_load_ohlcv_prefers_adj_close_and_sorts(tmp_path):
    csv = tmp_path / "X.csv"
    csv.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2001-01-02,2,3,1,2.5,5.0,100\n"
        "2000-01-03,1,2,0.5,1.5,3.0,200\n"  # deliberately out of order
    )
    df = load_ohlcv(str(csv))
    assert list(df.index) == sorted(df.index)          # sorted by Date
    assert df["price"].iloc[0] == 3.0                   # Adj Close used as price
    assert df.index[0] == pd.Timestamp("2000-01-03")


def test_load_ohlcv_missing_date_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("Open,Close\n1,2\n")
    with pytest.raises(ValueError):
        load_ohlcv(str(csv))


# --------------------------------------------------------------------------- #
# data_loader: train/test split (no look-ahead leakage)
# --------------------------------------------------------------------------- #

def _dated_frame(years):
    idx = pd.to_datetime([f"{y}-06-01" for y in years])
    return pd.DataFrame({"log_return": range(len(years))}, index=idx)

def test_split_boundaries_and_no_overlap():
    df = _dated_frame([1998, 2001, 2010, 2021, 2022, 2025])
    train, test = train_test_split(df, split_date="2022-01-01", train_start="2000-01-01")
    assert (train.index < pd.Timestamp("2022-01-01")).all()
    assert (train.index >= pd.Timestamp("2000-01-01")).all()  # 1998 excluded
    assert (test.index >= pd.Timestamp("2022-01-01")).all()
    # No date appears in both halves.
    assert set(train.index).isdisjoint(set(test.index))
    assert len(train) == 3 and len(test) == 2  # 1998 dropped by train_start


def test_split_train_start_none_keeps_all_history():
    df = _dated_frame([1990, 2005, 2023])
    train, test = train_test_split(df, split_date="2022-01-01", train_start=None)
    assert len(train) == 2  # 1990 + 2005
    assert len(test) == 1


def test_to_observations_missing_column_raises():
    df = _dated_frame([2001, 2002])
    with pytest.raises(KeyError):
        to_observations(df, ["log_return", "realized_vol"])  # realized_vol absent


# --------------------------------------------------------------------------- #
# hmm_model: parameter counting
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "cov, n_features, expected_cov_params",
    [
        ("full", 2, 3 * (2 * 3 // 2)),   # per-state d(d+1)/2 = 3, times 3 states
        ("diag", 2, 3 * 2),              # per-state d
        ("tied", 2, 2 * 3 // 2),         # one shared d(d+1)/2
        ("spherical", 2, 3),             # one scalar per state
    ],
)
def test_count_params(cov, n_features, expected_cov_params):
    k = 3
    start = k - 1
    trans = k * (k - 1)
    means = k * n_features
    expected = start + trans + means + expected_cov_params
    assert _count_params(k, n_features, cov) == expected


def test_count_params_unknown_type():
    with pytest.raises(ValueError):
        _count_params(3, 2, "banana")


# --------------------------------------------------------------------------- #
# hmm_model: fitting on synthetic data
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_two_regime():
    """1000 obs from two well-separated 2-D Gaussians (low-vol vs high-vol)."""
    rng = np.random.default_rng(42)
    a = rng.normal([0.001, 0.08], [0.005, 0.01], size=(600, 2))
    b = rng.normal([-0.002, 0.40], [0.02, 0.05], size=(400, 2))
    return np.vstack([a, b])


def test_fit_hmm_returns_valid_probabilities(synthetic_two_regime):
    fit = fit_hmm(synthetic_two_regime, n_states=2, n_restarts=3, n_iter=100)
    m = fit.model
    # Transition matrix rows are valid distributions.
    assert np.allclose(m.transmat_.sum(axis=1), 1.0)
    assert (m.transmat_ >= 0).all()
    # Initial distribution is a valid distribution.
    assert m.startprob_.sum() == pytest.approx(1.0)
    assert np.isfinite(fit.log_likelihood)
    assert np.isfinite(fit.bic) and np.isfinite(fit.aic)


def test_bic_aic_formulas(synthetic_two_regime):
    fit = fit_hmm(synthetic_two_regime, n_states=2, n_restarts=2, n_iter=100)
    n = len(synthetic_two_regime)
    k = _count_params(2, 2, "full")
    assert fit.bic == pytest.approx(-2 * fit.log_likelihood + k * np.log(n))
    assert fit.aic == pytest.approx(-2 * fit.log_likelihood + 2 * k)


def test_select_n_states_picks_min_bic(synthetic_two_regime):
    best, results = select_n_states(
        synthetic_two_regime, candidates=[2, 3], n_restarts=2
    )
    assert best.bic == min(r.bic for r in results)
    assert best.n_states in (2, 3)


def test_select_n_states_handles_longer_candidate_range(synthetic_two_regime):
    # The candidate range was widened to 2..8; verify the sweep still returns the
    # min-BIC model over a multi-value range. Use a small subset to stay fast.
    candidates = [2, 3, 4]
    best, results = select_n_states(
        synthetic_two_regime, candidates=candidates, n_restarts=2
    )
    assert [r.n_states for r in results] == candidates
    assert best.bic == min(r.bic for r in results)
    assert best.n_states in candidates


def test_regime_summary_structure(synthetic_two_regime):
    fit = fit_hmm(synthetic_two_regime, n_states=2, n_restarts=2, n_iter=100)
    summary = regime_summary(fit.model, synthetic_two_regime, ["log_return", "realized_vol"])
    assert set(summary.keys()) == {0, 1}
    # Fractions over all states sum to 1.
    assert sum(s["fraction"] for s in summary.values()) == pytest.approx(1.0)
    for s in summary.values():
        assert set(s["mean"].keys()) == {"log_return", "realized_vol"}
        assert 0.0 <= s["self_transition"] <= 1.0


ALLOWED_LABELS = {"crisis", "bull", "neutral/choppy"}


def test_regime_summary_includes_label(synthetic_two_regime):
    fit = fit_hmm(synthetic_two_regime, n_states=2, n_restarts=2, n_iter=100)
    summary = regime_summary(fit.model, synthetic_two_regime, ["log_return", "realized_vol"])
    for s in summary.values():
        assert "label" in s
        assert s["label"] in ALLOWED_LABELS
        # Existing keys are preserved alongside the new label.
        assert {"fraction", "mean", "self_transition"} <= set(s.keys())
