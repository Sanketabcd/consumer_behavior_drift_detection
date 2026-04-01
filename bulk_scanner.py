"""
bulk_scanner.py  —  Bulk drift scanner
Upload multiple CSV files, auto-detect columns in each,
run all pairwise comparisons and return a ranked summary.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def _psi(base, curr, bins=10):
    """Quick PSI calculation."""
    bp = np.percentile(base, np.linspace(0, 100, bins + 1))
    bp[0] = -np.inf; bp[-1] = np.inf
    b_pct = np.histogram(base, bp)[0] / len(base)
    c_pct = np.histogram(curr, bp)[0] / len(curr)
    b_pct = np.where(b_pct == 0, 1e-6, b_pct)
    c_pct = np.where(c_pct == 0, 1e-6, c_pct)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def _chi_stat(base_series, curr_series):
    """Quick Chi-Square stat for categorical columns."""
    cats = sorted(set(base_series.unique()) | set(curr_series.unique()))
    b_cnt = base_series.value_counts().reindex(cats, fill_value=0)
    c_cnt = curr_series.value_counts().reindex(cats, fill_value=0)
    b_exp = (b_cnt / b_cnt.sum()) * c_cnt.sum()
    mask = b_exp > 0
    if mask.sum() < 2:
        return 0.0, 1.0
    chi2, p = stats.chisquare(f_obs=c_cnt[mask], f_exp=b_exp[mask])
    return float(chi2), float(p)


def _severity(ks_stat, psi_val, drift_detected):
    """Return severity label and sort key."""
    if not drift_detected:
        return "🟢 Stable", 0
    if psi_val > 0.25 or ks_stat > 0.35:
        return "🔴 Critical", 3
    if psi_val > 0.10 or ks_stat > 0.20:
        return "🟡 Moderate", 2
    return "🟠 Mild", 1


def scan_pair(name_a, df_a, name_b, df_b, alpha=0.05):
    """
    Run drift detection between two pre-mapped DataFrames.
    Returns a result dict for the summary table.
    """
    result = {
        "Baseline File":  name_a,
        "Current File":   name_b,
        "Base Rows":      len(df_a),
        "Curr Rows":      len(df_b),
        "KS Stat":        None,
        "KS p-value":     None,
        "PSI":            None,
        "Cat Chi2 p":     None,
        "Pay Chi2 p":     None,
        "Base Mean $":    None,
        "Curr Mean $":    None,
        "Mean Change %":  None,
        "Drift":          False,
        "Severity":       "🟢 Stable",
        "_sort_key":      0,
    }

    # Purchase Amount — KS test + PSI
    try:
        base_amt = df_a["Purchase_Amount"].dropna()
        curr_amt = df_b["Purchase_Amount"].dropna()
        if len(base_amt) > 5 and len(curr_amt) > 5:
            ks_stat, ks_p = stats.ks_2samp(base_amt, curr_amt)
            psi_val = _psi(base_amt.values, curr_amt.values)
            b_mean  = base_amt.mean()
            c_mean  = curr_amt.mean()
            pct_chg = ((c_mean / b_mean) - 1) * 100 if b_mean > 0 else 0
            result["KS Stat"]       = round(ks_stat, 4)
            result["KS p-value"]    = round(ks_p, 6)
            result["PSI"]           = round(psi_val, 4)
            result["Base Mean $"]   = round(b_mean, 2)
            result["Curr Mean $"]   = round(c_mean, 2)
            result["Mean Change %"] = round(pct_chg, 1)
            result["Drift"]         = bool(ks_p < alpha or psi_val > 0.10)
    except Exception:
        pass

    # Product Category — Chi-Square
    try:
        if "Product_Category" in df_a and "Product_Category" in df_b:
            _, cat_p = _chi_stat(df_a["Product_Category"].dropna(),
                                  df_b["Product_Category"].dropna())
            result["Cat Chi2 p"] = round(cat_p, 6)
            if cat_p < alpha:
                result["Drift"] = True
    except Exception:
        pass

    # Payment Method — Chi-Square
    try:
        if "Payment_Method" in df_a and "Payment_Method" in df_b:
            _, pay_p = _chi_stat(df_a["Payment_Method"].dropna(),
                                  df_b["Payment_Method"].dropna())
            result["Pay Chi2 p"] = round(pay_p, 6)
            if pay_p < alpha:
                result["Drift"] = True
    except Exception:
        pass

    sev, sort_key = _severity(
        result["KS Stat"] or 0,
        result["PSI"]     or 0,
        result["Drift"],
    )
    result["Severity"]  = sev
    result["_sort_key"] = sort_key

    return result


def run_bulk_scan(file_map: dict, alpha=0.05,
                  mode="consecutive") -> pd.DataFrame:
    """
    Run drift detection across multiple files.

    Parameters
    ----------
    file_map : dict of {filename: mapped_DataFrame}
    alpha    : p-value threshold
    mode     : "consecutive" — each file vs the next one
               "first_vs_all" — first file is baseline, rest are current
               "all_pairs"   — every possible pair (can be large)

    Returns a DataFrame summary sorted by severity descending.
    """
    names = list(file_map.keys())
    dfs   = list(file_map.values())
    rows  = []

    if mode == "consecutive":
        pairs = [(i, i+1) for i in range(len(names)-1)]
    elif mode == "first_vs_all":
        pairs = [(0, i) for i in range(1, len(names))]
    else:  # all_pairs
        pairs = [(i, j) for i in range(len(names))
                          for j in range(i+1, len(names))]

    for i, j in pairs:
        row = scan_pair(names[i], dfs[i], names[j], dfs[j], alpha=alpha)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
    df = df.reset_index(drop=True)
    return df