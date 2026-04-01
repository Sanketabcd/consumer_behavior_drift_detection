"""
drift_detection.py  —  Enhanced with PSI, Wasserstein, Rolling Window
"""

import pandas as pd
import numpy as np
from scipy import stats


ALPHA = 0.05


# ── KS Test ───────────────────────────────────────────────────────────────────
def ks_test(baseline_series, current_series):
    base = baseline_series.dropna()
    curr = current_series.dropna()
    ks_stat, p_value = stats.ks_2samp(base, curr)
    drift_detected = bool(p_value < ALPHA)
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "verdict": "⚠️ Drift Detected" if drift_detected else "✅ No Significant Drift",
    }


# ── Chi-Square Test ───────────────────────────────────────────────────────────
def chi_square_test(baseline_series, current_series):
    all_categories = sorted(set(baseline_series.unique()) | set(current_series.unique()))
    base_counts = baseline_series.value_counts().reindex(all_categories, fill_value=0)
    curr_counts  = current_series.value_counts().reindex(all_categories, fill_value=0)
    base_proportions = base_counts / base_counts.sum()
    expected_counts  = base_proportions * curr_counts.sum()
    mask = expected_counts > 0
    chi2_stat, p_value = stats.chisquare(f_obs=curr_counts[mask], f_exp=expected_counts[mask])
    drift_detected = bool(p_value < ALPHA)
    baseline_pct = (base_counts / base_counts.sum() * 100).round(1).to_dict()
    current_pct  = (curr_counts  / curr_counts.sum()  * 100).round(1).to_dict()
    return {
        "test": "Chi-Square",
        "statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "verdict": "⚠️ Drift Detected" if drift_detected else "✅ No Significant Drift",
        "baseline_distribution": baseline_pct,
        "current_distribution":  current_pct,
    }


# ── PSI Score ─────────────────────────────────────────────────────────────────
def psi_score(baseline_series, current_series, bins=10):
    """
    Population Stability Index — industry standard used in banking/insurance.
    PSI < 0.10  → No significant change (stable)
    PSI 0.10–0.25 → Moderate change (monitor)
    PSI > 0.25  → Significant change (action required)
    """
    base = baseline_series.dropna().values
    curr = current_series.dropna().values

    breakpoints = np.percentile(base, np.linspace(0, 100, bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    base_pct = np.histogram(base, breakpoints)[0] / len(base)
    curr_pct = np.histogram(curr, breakpoints)[0] / len(curr)

    # Avoid log(0)
    base_pct = np.where(base_pct == 0, 1e-6, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))

    if psi < 0.10:
        level, color = "Stable", "#06D6A0"
    elif psi < 0.25:
        level, color = "Moderate Change", "#FFB703"
    else:
        level, color = "Significant Change", "#EF476F"

    # Per-bin breakdown for chart
    bin_labels = [f"B{i+1}" for i in range(bins)]
    bin_data = pd.DataFrame({
        "Bin":      bin_labels,
        "Baseline": (base_pct * 100).round(2),
        "Current":  (curr_pct * 100).round(2),
        "PSI_contribution": ((curr_pct - base_pct) * np.log(curr_pct / base_pct)).round(4),
    })

    return {
        "psi":       round(float(psi), 4),
        "level":     level,
        "color":     color,
        "bin_data":  bin_data,
    }


# ── Wasserstein Distance ──────────────────────────────────────────────────────
def wasserstein_distance(baseline_series, current_series):
    """
    Earth Mover's Distance — how much "work" is needed to transform
    one distribution into another. Interpretable in the original units ($).
    """
    base = baseline_series.dropna().values
    curr = current_series.dropna().values
    dist = stats.wasserstein_distance(base, curr)

    base_mean = np.mean(base)
    pct_shift  = (dist / base_mean * 100) if base_mean > 0 else 0

    return {
        "distance":  round(float(dist), 4),
        "pct_shift": round(float(pct_shift), 2),
        "interpretation": f"Distributions shifted by ~${dist:.2f} on average",
    }


# ── Rolling Drift Window ──────────────────────────────────────────────────────
def rolling_drift_window(combined_df, window_days=30, step_days=7):
    """
    Slide a window across the combined timeline.
    For each window, compare the first half (baseline) vs second half (current).
    Returns a DataFrame of KS statistics and p-values over time.
    """
    df = combined_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    total_days = (max_date - min_date).days

    records = []
    start = min_date

    while start + pd.Timedelta(days=window_days) <= max_date:
        end      = start + pd.Timedelta(days=window_days)
        mid      = start + pd.Timedelta(days=window_days // 2)
        window   = df[(df["Date"] >= start) & (df["Date"] < end)]
        base_win = window[window["Date"] < mid]["Purchase_Amount"].dropna()
        curr_win = window[window["Date"] >= mid]["Purchase_Amount"].dropna()

        if len(base_win) > 5 and len(curr_win) > 5:
            ks_stat, p_val = stats.ks_2samp(base_win, curr_win)
            wass = stats.wasserstein_distance(base_win.values, curr_win.values)
            records.append({
                "window_start": start.date(),
                "window_mid":   mid.date(),
                "ks_stat":      round(float(ks_stat), 4),
                "p_value":      round(float(p_val), 6),
                "wasserstein":  round(float(wass), 2),
                "drift":        bool(p_val < ALPHA),
                "base_mean":    round(float(base_win.mean()), 2),
                "curr_mean":    round(float(curr_win.mean()), 2),
                "n_base":       len(base_win),
                "n_curr":       len(curr_win),
            })

        start += pd.Timedelta(days=step_days)

    return pd.DataFrame(records)


# ── Summary Stats ─────────────────────────────────────────────────────────────
def summary_stats(baseline_df, current_df):
    stats_baseline = baseline_df["Purchase_Amount"].describe().rename("Baseline")
    stats_current  = current_df["Purchase_Amount"].describe().rename("Current")
    return pd.concat([stats_baseline, stats_current], axis=1).round(2)


# ── Run All Checks ────────────────────────────────────────────────────────────
def run_all_drift_checks(baseline_df, current_df, alpha=0.05):
    global ALPHA
    ALPHA = alpha

    results = {}
    results["purchase_amount"]  = ks_test(baseline_df["Purchase_Amount"], current_df["Purchase_Amount"])
    results["product_category"] = chi_square_test(baseline_df["Product_Category"], current_df["Product_Category"])
    results["payment_method"]   = chi_square_test(baseline_df["Payment_Method"],   current_df["Payment_Method"])
    results["summary_stats"]    = summary_stats(baseline_df, current_df)
    results["psi"]              = psi_score(baseline_df["Purchase_Amount"], current_df["Purchase_Amount"])
    results["wasserstein"]      = wasserstein_distance(baseline_df["Purchase_Amount"], current_df["Purchase_Amount"])
    return results