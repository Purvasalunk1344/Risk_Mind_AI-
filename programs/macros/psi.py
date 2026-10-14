import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """Population stability index for score distributions."""
    exp = pd.Series(expected, copy=True).astype(float).dropna()
    act = pd.Series(actual, copy=True).astype(float).dropna()

    if exp.empty or act.empty:
        raise ValueError("expected and actual series must not be empty")

    if len(np.unique(exp)) < 2 or len(np.unique(act)) < 2:
        return {"psi": 0.0, "breakdown": pd.DataFrame({
            "bin": ["all"], "expected_pct": [1.0], "actual_pct": [1.0], "component": [0.0]
        })}

    edges = np.quantile(exp, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    exp_bins = pd.cut(exp, bins=edges, include_lowest=True, duplicates="drop")
    act_bins = pd.cut(act, bins=edges, include_lowest=True, duplicates="drop")

    exp_pct = exp_bins.value_counts(normalize=True, sort=False)
    act_pct = act_bins.value_counts(normalize=True, sort=False)
    idx = exp_pct.index.union(act_pct.index)
    exp_pct = exp_pct.reindex(idx, fill_value=0)
    act_pct = act_pct.reindex(idx, fill_value=0)

    diff = (act_pct - exp_pct)
    comp = (diff * np.log((act_pct + 1e-6) / (exp_pct + 1e-6))).fillna(0.0)
    breakdown = pd.DataFrame({
        "bin": idx.astype(str),
        "expected_pct": exp_pct.round(6),
        "actual_pct": act_pct.round(6),
        "component": comp.round(6),
    })
    psi = float(comp.sum())
    return {"psi": psi, "breakdown": breakdown}


def psi_summary(train_scores, current_scores, bins=10):
    result = calculate_psi(train_scores, current_scores, bins=bins)
    return result["psi"], result["breakdown"]
