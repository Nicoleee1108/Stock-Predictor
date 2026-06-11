"""Gaussian HMM regime model: fit, score, model-select."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class FitResult:
    model: GaussianHMM
    n_states: int
    log_likelihood: float
    bic: float
    aic: float
    converged: bool


def _count_params(n_states: int, n_features: int, covariance_type: str) -> int:
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "full":
        cov = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov = n_states * n_features
    elif covariance_type == "tied":
        cov = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        cov = n_states
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}")
    return start + trans + means + cov


def fit_hmm(
    X: np.ndarray,
    n_states: int,
    covariance_type: str = "full",
    n_iter: int = 200,
    n_restarts: int = 5,
    seed: int = 0,
) -> FitResult:
    """Fit a Gaussian HMM with multiple random restarts; return the best by training log-likelihood."""
    best: FitResult | None = None
    rng = np.random.default_rng(seed)

    for r in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-4,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            model.fit(X)
        except Exception:
            continue

        ll = model.score(X)
        if not np.isfinite(ll):
            continue

        k = _count_params(n_states, X.shape[1], covariance_type)
        n = X.shape[0]
        bic = -2 * ll + k * np.log(n)
        aic = -2 * ll + 2 * k
        result = FitResult(model, n_states, ll, bic, aic, bool(model.monitor_.converged))

        if best is None or result.log_likelihood > best.log_likelihood:
            best = result

    if best is None:
        raise RuntimeError(f"All {n_restarts} restarts failed for n_states={n_states}")
    return best


def select_n_states(
    X: np.ndarray,
    candidates: list[int] = (2, 3, 4, 5, 6, 7, 8),
    covariance_type: str = "full",
    n_restarts: int = 5,
    seed: int = 0,
) -> tuple[FitResult, list[FitResult]]:
    """Sweep candidate state counts; return (best by BIC, all results)."""
    results = [
        fit_hmm(X, k, covariance_type=covariance_type, n_restarts=n_restarts, seed=seed + k)
        for k in candidates
    ]
    best = min(results, key=lambda r: r.bic)
    return best, results


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def state_posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)


def _regime_label(means: dict, vol_median: float) -> str:
    """Heuristic regime name from a state's mean log_return and realized_vol.

    Volatility is judged relative to the median across states; return by its sign.
    """
    ret = means.get("log_return", 0.0)
    vol = means.get("realized_vol", vol_median)
    high_vol = vol >= vol_median
    if ret < 0 and high_vol:
        return "crisis"
    if ret > 0 and not high_vol:
        return "bull"
    return "neutral/choppy"


def regime_summary(model: GaussianHMM, X: np.ndarray, feature_names: list[str]) -> dict:
    """Per-state mean and covariance — used to label regimes (bull/bear/high-vol/etc.)."""
    states = predict_states(model, X)
    means = [dict(zip(feature_names, model.means_[s])) for s in range(model.n_components)]
    vols = [m.get("realized_vol", 0.0) for m in means]
    vol_median = float(np.median(vols)) if vols else 0.0
    summary = {}
    for s in range(model.n_components):
        mask = states == s
        summary[s] = {
            "fraction": float(mask.mean()),
            "mean": means[s],
            "label": _regime_label(means[s], vol_median),
            "self_transition": float(model.transmat_[s, s]),
        }
    return summary


if __name__ == "__main__":
    import sys

    from data_loader import compute_features, load_ohlcv, to_observations, train_test_split

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/SPY.csv"
    feats = compute_features(load_ohlcv(path))
    train, test = train_test_split(feats)

    cols = ["log_return", "realized_vol"]
    X_train = to_observations(train, cols)
    X_test = to_observations(test, cols)

    best, all_results = select_n_states(X_train, candidates=[2, 3, 4, 5, 6, 7, 8])
    print(f"{'K':>3} {'logL':>12} {'BIC':>12} {'AIC':>12} converged")
    for r in all_results:
        print(f"{r.n_states:>3} {r.log_likelihood:>12.1f} {r.bic:>12.1f} {r.aic:>12.1f} {r.converged}")
    print(f"\nBest by BIC: K={best.n_states}")

    test_ll = best.model.score(X_test)
    print(f"Held-out log-likelihood: {test_ll:.1f}  ({test_ll / len(X_test):.4f} per obs)")

    print("\nRegime summary (train):")
    for s, info in regime_summary(best.model, X_train, cols).items():
        print(f"  state {s}: {info}")
