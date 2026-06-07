"""Visualize HMM regimes: price/vol/posteriors/emissions/model-selection figures."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from data_loader import compute_features, load_ohlcv, to_observations, train_test_split
from hmm_model import select_n_states, state_posteriors

FEATURE_COLS = ["log_return", "realized_vol"]
FIG_DIR = "figures"


def _state_order(model) -> np.ndarray:
    """States sorted by mean log_return (ascending: bearish -> bullish)."""
    return np.argsort(model.means_[:, 0])


def _relabel(states: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Map raw state ids to ordered ids so colors are interpretable across figures."""
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[states]


def plot_price_regimes(feats, states, n_states, cmap, path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    dates = feats.index
    price = feats["price"].to_numpy()
    # Shade background spans by contiguous regime so the price line stays readable.
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.axvspan(dates[start], dates[i - 1], color=cmap(states[start]), alpha=0.25)
            start = i
    ax.plot(dates, price, color="black", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"Price colored by Viterbi regime (K={n_states}, ordered by mean log-return)")
    ax.set_ylabel("price (log scale)")
    handles = [plt.Line2D([0], [0], color=cmap(s), lw=6, alpha=0.5) for s in range(n_states)]
    ax.legend(handles, [f"regime {s}" for s in range(n_states)], loc="upper left", ncol=n_states)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_realized_vol(feats, path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(feats.index, feats["realized_vol"], color="firebrick", lw=0.8)
    ax.set_title("Annualized realized volatility (20-day rolling)")
    ax.set_ylabel("realized_vol")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_posteriors(feats, proba, n_states, cmap, path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.stackplot(
        feats.index,
        *[proba[:, s] for s in range(n_states)],
        colors=[cmap(s) for s in range(n_states)],
        labels=[f"regime {s}" for s in range(n_states)],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Smoothed state posterior probabilities")
    ax.set_ylabel("P(state | data)")
    ax.legend(loc="upper left", ncol=n_states)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_emissions(model, X, states, order, n_states, cmap, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for s in range(n_states):
        mask = states == s
        ax.scatter(X[mask, 0], X[mask, 1], s=4, alpha=0.3, color=cmap(s), label=f"regime {s}")
    # Draw each regime's emission as a 2-sigma Gaussian ellipse.
    for s in range(n_states):
        raw = order[s]
        mean = model.means_[raw]
        cov = model.covars_[raw]
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * 2 * np.sqrt(vals)  # 2-sigma diameter on each axis
        ell = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor=cmap(s), facecolor="none", lw=2.5,
        )
        ax.add_patch(ell)
        ax.plot(mean[0], mean[1], marker="x", color=cmap(s), ms=10, mew=2.5)
    ax.set_xlabel("log_return")
    ax.set_ylabel("realized_vol")
    ax.set_title("Gaussian emissions: observations + 2-sigma regime ellipses")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_selection(results, path: str) -> None:
    ks = [r.n_states for r in results]
    bics = [r.bic for r in results]
    aics = [r.aic for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, bics, marker="o", label="BIC")
    ax.plot(ks, aics, marker="s", label="AIC")
    best_k = ks[int(np.argmin(bics))]
    ax.axvline(best_k, color="grey", ls="--", lw=1, label=f"best K (BIC) = {best_k}")
    ax.set_xticks(ks)
    ax.set_xlabel("number of states K")
    ax.set_ylabel("information criterion (lower is better)")
    ax.set_title("Model selection: BIC / AIC vs K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(csv_path: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    feats = compute_features(load_ohlcv(csv_path))
    train, _ = train_test_split(feats)

    X_train = to_observations(train, FEATURE_COLS)
    X_full = to_observations(feats, FEATURE_COLS)

    best, all_results = select_n_states(X_train, candidates=[2, 3, 4, 5])
    model = best.model
    n_states = best.n_states
    print(f"Selected K={n_states} by BIC on train ({len(X_train)} obs)")

    order = _state_order(model)
    states_full = _relabel(model.predict(X_full), order)
    proba_full = state_posteriors(model, X_full)[:, order]
    cmap = plt.get_cmap("coolwarm", n_states)

    paths = {
        "01_price_regimes.png": lambda p: plot_price_regimes(feats, states_full, n_states, cmap, p),
        "02_realized_vol.png": lambda p: plot_realized_vol(feats, p),
        "03_state_posteriors.png": lambda p: plot_posteriors(feats, proba_full, n_states, cmap, p),
        "04_emission_ellipses.png": lambda p: plot_emissions(
            model, X_full, states_full, order, n_states, cmap, p
        ),
        "05_model_selection.png": lambda p: plot_model_selection(all_results, p),
    }
    for name, fn in paths.items():
        out = os.path.join(FIG_DIR, name)
        fn(out)
        print(f"saved {os.path.abspath(out)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/SPY.csv"
    main(path)
