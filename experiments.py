"""Experiments quantifying HMM regime-model design choices on held-out 2022+ data."""

from __future__ import annotations

import logging
import os
import time

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from data_loader import compute_features, load_ohlcv, to_observations, train_test_split
from hmm_model import _count_params, fit_hmm, select_n_states

DATA = "dataset/SPY.csv"
TICKERS = {"SPY": "dataset/SPY.csv", "MSFT": "dataset/MSFT.csv", "AAPL": "dataset/AAPL.csv"}
DEFAULT_COLS = ["log_return", "realized_vol"]
OUT_DIR = "outputs"
FIG_DIR = "figures"

# K fixed for the design-choice experiments (a/b/d) so comparisons are apples-to-apples.
FIXED_K = 4


def _per_obs_ll(model, X: np.ndarray) -> float:
    return model.score(X) / len(X)


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _save_table(df: pd.DataFrame, name: str, title: str, note: str | None = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    if path.endswith(".md"):
        with open(path, "w") as f:
            f.write(f"# {title}\n\n{_to_markdown(df)}\n")
            if note:
                f.write(f"\n> **Note:** {note}\n")
    else:
        df.to_csv(path, index=False)
    print(f"\n{title}")
    print(df.to_string(index=False))
    print(f"saved {os.path.abspath(path)}")


def _load(csv_path: str, cols: list[str] = DEFAULT_COLS, train_start="2000-01-01"):
    feats = compute_features(load_ohlcv(csv_path))
    train, test = train_test_split(feats, train_start=train_start)
    return to_observations(train, cols), to_observations(test, cols)


def exp_covariance_type() -> None:
    """(a) full/diag/tied/spherical at fixed K -> held-out per-obs LL and BIC."""
    X_train, X_test = _load(DATA)
    rows = []
    for cov in ["full", "diag", "tied", "spherical"]:
        t0 = time.perf_counter()
        fit = fit_hmm(X_train, FIXED_K, covariance_type=cov)
        fit_seconds = time.perf_counter() - t0
        rows.append({
            "covariance_type": cov,
            "train_per_obs_LL": round(fit.log_likelihood / len(X_train), 4),
            "heldout_per_obs_LL": round(_per_obs_ll(fit.model, X_test), 4),
            "BIC": round(fit.bic, 1),
            "fit_seconds": round(fit_seconds, 3),
            "converged": fit.converged,
        })
    _save_table(pd.DataFrame(rows), "exp_a_covariance_type.md",
                f"(a) Covariance type comparison (K={FIXED_K}, SPY)")


def exp_feature_ablation() -> None:
    """(b) increasing feature sets -> held-out per-obs LL.

    Note: LL lives in different-dimensional spaces across feature sets, so values are
    not directly comparable in absolute terms; we report each for completeness.
    """
    feature_sets = [
        ["log_return"],
        ["log_return", "realized_vol"],
        ["log_return", "realized_vol", "volume_z"],
    ]
    rows = []
    for cols in feature_sets:
        X_train, X_test = _load(DATA, cols=cols)
        fit = fit_hmm(X_train, FIXED_K)
        rows.append({
            "features": "+".join(cols),
            "n_features": len(cols),
            "train_per_obs_LL": round(fit.log_likelihood / len(X_train), 4),
            "heldout_per_obs_LL": round(_per_obs_ll(fit.model, X_test), 4),
        })
    _save_table(pd.DataFrame(rows), "exp_b_feature_ablation.md",
                f"(b) Feature ablation (K={FIXED_K}, SPY)",
                note="Log-likelihoods live in different-dimensional observation spaces across "
                     "feature sets, so absolute values are NOT directly comparable. Read this as "
                     "'does adding a feature produce a coherent, well-separated regime model', not "
                     "as a strict LL ranking.")


def exp_training_window() -> None:
    """(c) sweep train_start, fixed 2022+ test set -> held-out per-obs LL; plot."""
    rows = []
    for year in [2000, 2005, 2010, 2015]:
        X_train, X_test = _load(DATA, train_start=f"{year}-01-01")
        fit = fit_hmm(X_train, FIXED_K)
        rows.append({
            "train_start": year,
            "n_train": len(X_train),
            "train_per_obs_LL": round(fit.log_likelihood / len(X_train), 4),
            "heldout_per_obs_LL": round(_per_obs_ll(fit.model, X_test), 4),
        })
    df = pd.DataFrame(rows)
    _save_table(df, "exp_c_training_window.md", f"(c) Training-window sweep (K={FIXED_K}, SPY)")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["train_start"], df["heldout_per_obs_LL"], marker="o")
    ax.set_xticks(df["train_start"])
    ax.set_xlabel("training window start year")
    ax.set_ylabel("held-out per-obs LL (2022+)")
    ax.set_title(f"Training window vs held-out LL (K={FIXED_K}, SPY)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "06_training_window.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {os.path.abspath(out)}")


def exp_baseline() -> None:
    """(d) HMM vs no-temporal-structure baselines (single Gaussian, GMM) -> held-out per-obs LL."""
    X_train, X_test = _load(DATA)
    n, d = X_train.shape

    hmm = fit_hmm(X_train, FIXED_K)

    # Single full-covariance Gaussian (1 component).
    single = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(X_train)
    # GMM with the same number of components as the HMM (same emissions, no transitions).
    gmm = GaussianMixture(n_components=FIXED_K, covariance_type="full",
                          n_init=5, random_state=0).fit(X_train)

    rows = [
        {"model": f"HMM (K={FIXED_K}, full)",
         "train_per_obs_LL": round(hmm.log_likelihood / n, 4),
         "heldout_per_obs_LL": round(_per_obs_ll(hmm.model, X_test), 4)},
        {"model": "Single Gaussian",
         "train_per_obs_LL": round(single.score(X_train), 4),
         "heldout_per_obs_LL": round(single.score(X_test), 4)},
        {"model": f"GMM (K={FIXED_K}, full)",
         "train_per_obs_LL": round(gmm.score(X_train), 4),
         "heldout_per_obs_LL": round(gmm.score(X_test), 4)},
    ]
    df = pd.DataFrame(rows)
    _save_table(df, "exp_d_baseline.md",
                f"(d) HMM vs no-temporal-structure baselines (K={FIXED_K}, SPY)")
    delta = rows[0]["heldout_per_obs_LL"] - rows[2]["heldout_per_obs_LL"]
    print(f"HMM held-out per-obs LL advantage over same-K GMM: {delta:+.4f}")


def exp_cross_ticker() -> None:
    """(e) full select_n_states pipeline per ticker -> chosen K, train LL, held-out per-obs LL."""
    rows = []
    for ticker, path in TICKERS.items():
        X_train, X_test = _load(path)
        t0 = time.perf_counter()
        best, _ = select_n_states(X_train, candidates=[2, 3, 4, 5, 6, 7, 8])
        fit_seconds = time.perf_counter() - t0
        rows.append({
            "ticker": ticker,
            "chosen_K": best.n_states,
            "train_per_obs_LL": round(best.log_likelihood / len(X_train), 4),
            "heldout_per_obs_LL": round(_per_obs_ll(best.model, X_test), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "fit_seconds": round(fit_seconds, 3),
        })
    _save_table(pd.DataFrame(rows), "exp_e_cross_ticker.md",
                "(e) Cross-ticker model selection")


def main() -> None:
    exp_covariance_type()
    exp_feature_ablation()
    exp_training_window()
    exp_baseline()
    exp_cross_ticker()


if __name__ == "__main__":
    main()
