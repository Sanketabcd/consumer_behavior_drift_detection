"""
custom_rules.py  —  Custom drift rule engine
Users define rules like:
  "Alert if avg spend rises more than 20%"
  "Alert if Electronics share drops below 25%"
  "Alert if PayPal share exceeds 40%"
Rules are evaluated against live drift results and trigger alerts.
"""

import pandas as pd


RULE_TYPES = [
    "Avg spend change >  X%",
    "Avg spend change <  X%",
    "Category share  >  X%  (current)",
    "Category share  <  X%  (current)",
    "Payment share   >  X%  (current)",
    "Payment share   <  X%  (current)",
    "PSI score       >  X",
    "Wasserstein     >  $X",
    "p-value (amount)<  X",
]


def evaluate_rules(rules: list, results: dict,
                   baseline_df: pd.DataFrame,
                   current_df: pd.DataFrame) -> list:
    """
    Evaluate a list of user-defined rules against current drift results.

    Each rule is a dict:
        {
          "name":      "My rule name",
          "type":      one of RULE_TYPES,
          "threshold": float,
          "target":    str  (category or payment method name, if applicable)
        }

    Returns a list of result dicts:
        {
          "name":      str,
          "condition": str  (human-readable condition),
          "actual":    str  (actual value),
          "triggered": bool,
          "severity":  "warning" | "critical" | "ok"
        }
    """
    ss      = results.get("summary_stats", pd.DataFrame())
    psi     = results.get("psi", {})
    wss     = results.get("wasserstein", {})
    pc      = results.get("product_category", {})
    pm      = results.get("payment_method", {})

    b_mean  = float(ss.loc["mean","Baseline"]) if "mean" in ss.index else 0
    c_mean  = float(ss.loc["mean","Current"])  if "mean" in ss.index else 0
    pct_chg = ((c_mean / b_mean - 1) * 100) if b_mean > 0 else 0

    evaluated = []

    for rule in rules:
        rtype  = rule.get("type", "")
        thresh = float(rule.get("threshold", 0))
        target = rule.get("target", "")
        name   = rule.get("name", "Unnamed rule")

        triggered = False
        condition = ""
        actual    = ""

        if rtype == "Avg spend change >  X%":
            condition = f"Avg spend change > {thresh:.1f}%"
            actual    = f"{pct_chg:+.2f}%"
            triggered = pct_chg > thresh

        elif rtype == "Avg spend change <  X%":
            condition = f"Avg spend change < {thresh:.1f}%"
            actual    = f"{pct_chg:+.2f}%"
            triggered = pct_chg < thresh

        elif rtype == "Category share  >  X%  (current)":
            curr_share = pc.get("current_distribution", {}).get(target, 0)
            condition  = f"{target} current share > {thresh:.1f}%"
            actual     = f"{curr_share:.1f}%"
            triggered  = curr_share > thresh

        elif rtype == "Category share  <  X%  (current)":
            curr_share = pc.get("current_distribution", {}).get(target, 0)
            condition  = f"{target} current share < {thresh:.1f}%"
            actual     = f"{curr_share:.1f}%"
            triggered  = curr_share < thresh

        elif rtype == "Payment share   >  X%  (current)":
            curr_share = pm.get("current_distribution", {}).get(target, 0)
            condition  = f"{target} current share > {thresh:.1f}%"
            actual     = f"{curr_share:.1f}%"
            triggered  = curr_share > thresh

        elif rtype == "Payment share   <  X%  (current)":
            curr_share = pm.get("current_distribution", {}).get(target, 0)
            condition  = f"{target} current share < {thresh:.1f}%"
            actual     = f"{curr_share:.1f}%"
            triggered  = curr_share < thresh

        elif rtype == "PSI score       >  X":
            psi_val   = psi.get("psi", 0)
            condition = f"PSI score > {thresh:.3f}"
            actual    = f"{psi_val:.4f}"
            triggered = psi_val > thresh

        elif rtype == "Wasserstein     >  $X":
            wdist     = wss.get("distance", 0)
            condition = f"Wasserstein > ${thresh:.2f}"
            actual    = f"${wdist:.2f}"
            triggered = wdist > thresh

        elif rtype == "p-value (amount)<  X":
            pval      = results.get("purchase_amount", {}).get("p_value", 1.0)
            condition = f"p-value (amount) < {thresh:.4f}"
            actual    = f"{pval:.6f}"
            triggered = pval < thresh

        else:
            condition = rtype
            actual    = "—"

        severity = "ok"
        if triggered:
            severity = "critical" if (
                "PSI" in rtype and thresh > 0.25 or
                "spend" in rtype and abs(thresh) > 30
            ) else "warning"

        evaluated.append({
            "name":      name,
            "condition": condition,
            "actual":    actual,
            "triggered": triggered,
            "severity":  severity,
        })

    return evaluated


def get_categories(results: dict) -> list:
    """Return sorted list of all category names from results."""
    pc = results.get("product_category", {})
    cats = sorted(set(
        list(pc.get("baseline_distribution", {}).keys()) +
        list(pc.get("current_distribution",  {}).keys())
    ))
    return cats


def get_payment_methods(results: dict) -> list:
    """Return sorted list of all payment method names from results."""
    pm = results.get("payment_method", {})
    pays = sorted(set(
        list(pm.get("baseline_distribution", {}).keys()) +
        list(pm.get("current_distribution",  {}).keys())
    ))
    return pays