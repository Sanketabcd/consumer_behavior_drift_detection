"""
ai_engine.py  —  AI-powered features using Claude API
  1. Auto Drift Report      — plain-English narrative of what drifted and why it matters
  2. Root Cause Analysis    — probable causes behind the detected drift patterns
  3. Anomaly Annotations    — detect spike/drop points in rolling window and label them
"""

import json
import requests
import pandas as pd
import numpy as np


CLAUDE_MODEL = "claude-sonnet-4-20250514"
API_URL      = "https://api.anthropic.com/v1/messages"


def _call_claude(system_prompt: str, user_prompt: str,
                 max_tokens: int = 1000, api_key: str = "") -> str:
    """Call the Anthropic API and return the text response."""
    if not api_key or not api_key.strip().startswith("sk-ant-"):
        return "[API error: Invalid or missing API key. Please enter your Anthropic API key in the sidebar.]"
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key.strip(),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_prompt}],
            },
            timeout=60,
        )
        data = resp.json()
        if "content" in data and data["content"]:
            for block in data["content"]:
                if block.get("type") == "text":
                    return block["text"].strip()
        return f"[API error: {data.get('error', {}).get('message', 'unknown')}]"
    except Exception as e:
        return f"[Connection error: {str(e)}]"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Auto Drift Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_drift_report(results: dict, baseline_df: pd.DataFrame,
                           current_df: pd.DataFrame, api_key: str = "") -> str:
    """
    Generates a plain-English executive summary of all drift findings.
    Returns a markdown-formatted string.
    """
    pa  = results["purchase_amount"]
    pc  = results["product_category"]
    pm  = results["payment_method"]
    psi = results.get("psi", {})
    w   = results.get("wasserstein", {})
    ss  = results.get("summary_stats", pd.DataFrame())

    base_mean = float(ss.loc["mean", "Baseline"]) if "mean" in ss.index else 0
    curr_mean = float(ss.loc["mean", "Current"])  if "mean" in ss.index else 0

    cat_shifts = {}
    if "baseline_distribution" in pc and "current_distribution" in pc:
        for cat in pc["baseline_distribution"]:
            delta = pc["current_distribution"].get(cat, 0) - pc["baseline_distribution"].get(cat, 0)
            cat_shifts[cat] = round(delta, 1)

    pay_shifts = {}
    if "baseline_distribution" in pm and "current_distribution" in pm:
        for pay in pm["baseline_distribution"]:
            delta = pm["current_distribution"].get(pay, 0) - pm["baseline_distribution"].get(pay, 0)
            pay_shifts[pay] = round(delta, 1)

    context = f"""
DRIFT DETECTION RESULTS:

Purchase Amount (KS Test):
  - Drift detected: {pa['drift_detected']}
  - p-value: {pa['p_value']}
  - KS statistic: {pa['statistic']}
  - Baseline mean: ${base_mean:.2f}
  - Current mean:  ${curr_mean:.2f}
  - Mean change:   ${curr_mean - base_mean:.2f} ({((curr_mean/base_mean - 1)*100):.1f}% change)
  - PSI score:     {psi.get('psi', 'N/A')} ({psi.get('level', '')})
  - Wasserstein:   ${w.get('distance', 'N/A')} (~{w.get('pct_shift', '')}% of baseline mean)

Product Category (Chi-Square):
  - Drift detected: {pc['drift_detected']}
  - p-value: {pc['p_value']}
  - Category shifts (percentage points): {cat_shifts}

Payment Method (Chi-Square):
  - Drift detected: {pm['drift_detected']}
  - p-value: {pm['p_value']}
  - Method shifts (percentage points): {pay_shifts}
"""

    system = (
        "You are a senior data scientist writing an executive drift analysis report "
        "for a business analytics dashboard. Write in clear, confident business language. "
        "Use markdown formatting with ## headers and bullet points. "
        "Be specific with numbers. Focus on business impact. "
        "Keep the report under 350 words. Do not use jargon without explanation."
    )

    user = (
        f"Write a drift detection executive report based on these results:\n{context}\n\n"
        "Structure it as:\n"
        "## Executive Summary (2-3 sentences)\n"
        "## Key Findings (bullet points with specific numbers)\n"
        "## Business Impact (what this means for the company)\n"
        "## Recommended Actions (3 concrete next steps)"
    )

    return _call_claude(system, user, max_tokens=600, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Root Cause Analysis
# ─────────────────────────────────────────────────────────────────────────────

def generate_root_cause(results: dict, baseline_df: pd.DataFrame,
                         current_df: pd.DataFrame, api_key: str = "") -> str:
    """
    Analyzes drift patterns and suggests probable root causes.
    Returns a markdown-formatted string.
    """
    pa  = results["purchase_amount"]
    pc  = results["product_category"]
    pm  = results["payment_method"]
    ss  = results.get("summary_stats", pd.DataFrame())

    base_mean = float(ss.loc["mean", "Baseline"]) if "mean" in ss.index else 0
    curr_mean = float(ss.loc["mean", "Current"])  if "mean" in ss.index else 0
    base_std  = float(ss.loc["std",  "Baseline"]) if "std"  in ss.index else 0
    curr_std  = float(ss.loc["std",  "Current"])  if "std"  in ss.index else 0

    top_cat_gain = ""
    top_cat_loss = ""
    if "current_distribution" in pc and "baseline_distribution" in pc:
        deltas = {k: pc["current_distribution"].get(k,0) - pc["baseline_distribution"].get(k,0)
                  for k in pc["baseline_distribution"]}
        if deltas:
            top_cat_gain = max(deltas, key=deltas.get)
            top_cat_loss = min(deltas, key=deltas.get)

    top_pay_gain = ""
    if "current_distribution" in pm and "baseline_distribution" in pm:
        deltas = {k: pm["current_distribution"].get(k,0) - pm["baseline_distribution"].get(k,0)
                  for k in pm["baseline_distribution"]}
        if deltas:
            top_pay_gain = max(deltas, key=deltas.get)

    context = f"""
DRIFT PATTERN:
- Average purchase increased from ${base_mean:.2f} to ${curr_mean:.2f} (+{((curr_mean/base_mean-1)*100) if base_mean>0 else 0:.1f}%)
- Spending variability changed: std ${base_std:.2f} → ${curr_std:.2f}
- Product category with biggest gain: {top_cat_gain}
- Product category with biggest loss: {top_cat_loss}
- Payment method gaining share: {top_pay_gain}
- All three features show drift: Purchase Amount={pa['drift_detected']}, Category={pc['drift_detected']}, Payment={pm['drift_detected']}
"""

    system = (
        "You are a consumer behavior analyst specializing in root cause investigation "
        "for retail and e-commerce platforms. Be analytical and specific. "
        "Use markdown. Keep response under 300 words."
    )

    user = (
        f"Based on this consumer behavior drift pattern, identify the most likely root causes:\n{context}\n\n"
        "Structure your analysis as:\n"
        "## Most Probable Causes (ranked 1-3 with confidence level)\n"
        "## Supporting Evidence (which patterns point to each cause)\n"
        "## What to Investigate Next (data sources to confirm)"
    )

    return _call_claude(system, user, max_tokens=500, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Anomaly Annotations for Rolling Window
# ─────────────────────────────────────────────────────────────────────────────

def generate_anomaly_annotations(rolling_df: pd.DataFrame, api_key: str = "") -> list:
    """
    Detects significant spikes/drops in the rolling KS statistic
    and returns a list of annotation dicts for the chart.
    Each dict: {"date": str, "text": str, "y": float}
    """
    if rolling_df.empty or len(rolling_df) < 3:
        return []

    ks = rolling_df["ks_stat"].values
    dates = rolling_df["window_mid"].astype(str).values
    p_vals = rolling_df["p_value"].values
    means  = rolling_df["curr_mean"].values

    # Find anomaly candidates: points where KS spikes > 1.5 std above rolling mean
    rolling_mean = pd.Series(ks).rolling(3, min_periods=1).mean().values
    rolling_std  = pd.Series(ks).rolling(3, min_periods=1).std().fillna(0.01).values
    z_scores = (ks - rolling_mean) / rolling_std

    candidates = []
    for i in range(len(ks)):
        if abs(z_scores[i]) > 1.5 and p_vals[i] < 0.05:
            candidates.append({
                "idx":   i,
                "date":  dates[i],
                "ks":    float(ks[i]),
                "pval":  float(p_vals[i]),
                "mean":  float(means[i]),
                "zscore": float(z_scores[i]),
            })

    if not candidates:
        return []

    # Ask Claude to label the top 3 anomalies
    top = sorted(candidates, key=lambda x: abs(x["zscore"]), reverse=True)[:3]

    context = json.dumps([{
        "date": c["date"], "ks_statistic": c["ks"],
        "p_value": c["pval"], "current_mean_purchase": c["mean"],
    } for c in top])

    system = (
        "You are a data analyst annotating anomalies on a drift detection chart. "
        "For each anomaly, write a SHORT label (max 6 words) explaining the likely event. "
        "Return ONLY a valid JSON array of objects with keys 'date' and 'label'. "
        "No markdown, no extra text."
    )

    user = (
        f"Label these anomaly points on a consumer purchase drift chart:\n{context}\n"
        "Examples of good labels: 'Sudden spend spike', 'Category shift detected', "
        "'Holiday effect', 'Pricing change signal'. Return JSON only."
    )

    raw = _call_claude(system, user, max_tokens=200, api_key=api_key)

    try:
        raw_clean = raw.strip().lstrip("```json").rstrip("```").strip()
        labels = json.loads(raw_clean)
        annotations = []
        for c, lbl in zip(top, labels):
            annotations.append({
                "date": c["date"],
                "text": lbl.get("label", "Anomaly detected"),
                "y":    c["ks"],
            })
        return annotations
    except Exception:
        return [{"date": c["date"], "text": "Drift spike", "y": c["ks"]} for c in top[:2]]