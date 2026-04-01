"""
report_generator.py  —  Standalone HTML report generator
Produces a single self-contained .html file with all Plotly charts embedded.
No Python / Streamlit needed to view — just open in any browser.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime


# ── Dark theme tokens matching the dashboard ──────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#080c14;color:#e2e8f0;font-family:'DM Sans',sans-serif;padding:2rem}
h1{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#3A86FF,#9B5DE5);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.3rem}
.sub{font-size:.85rem;color:#475569;margin-bottom:2rem}
.section{margin:2rem 0}
.sec-label{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
           color:#3A86FF;border-left:3px solid #3A86FF;padding-left:10px;margin-bottom:1.2rem;
           display:flex;align-items:center;gap:8px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.5rem}
.kpi{background:#0d1120;border:1px solid #1a1f35;border-radius:12px;padding:.9rem 1.1rem}
.kpi-label{font-size:.66rem;color:#3a4a6a;text-transform:uppercase;letter-spacing:.12em;font-weight:600;margin-bottom:.4rem}
.kpi-value{font-size:1.3rem;font-family:'IBM Plex Mono',monospace;font-weight:500}
.card-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:1.5rem}
.drift-card{background:linear-gradient(160deg,#0e1428,#111a30);border:1px solid #1e2545;
            border-radius:16px;padding:1.2rem 1.4rem;position:relative;overflow:hidden}
.drift-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;border-radius:16px 16px 0 0}
.drifted::before{background:linear-gradient(90deg,#EF476F,#ff6b8a)}
.stable::before{background:linear-gradient(90deg,#06D6A0,#00f0b0)}
.dc-feat{font-size:.68rem;color:#3a4a6a;text-transform:uppercase;letter-spacing:.12em;font-weight:600;margin-bottom:.3rem}
.dc-test{font-size:1rem;font-weight:600;color:#c8d4f0;margin-bottom:.7rem;font-family:'IBM Plex Mono',monospace}
.badge{display:inline-flex;align-items:center;gap:4px;padding:4px 11px;border-radius:20px;
       font-size:.73rem;font-weight:700;margin-bottom:.7rem}
.d-yes{background:rgba(239,71,111,.12);color:#ff7096;border:1px solid rgba(239,71,111,.25)}
.d-no {background:rgba(6,214,160,.10);color:#06D6A0;border:1px solid rgba(6,214,160,.2)}
.dc-stats{font-family:'IBM Plex Mono',monospace;font-size:.77rem;color:#3a4a6a;line-height:2;
          border-top:1px solid #1a1f35;padding-top:.6rem}
.dc-stats span{color:#94a3b8}
.chart-box{background:#0d1120;border:1px solid #1a1f35;border-radius:14px;
           padding:.5rem;margin-bottom:1.2rem;overflow:hidden}
.rules-table{width:100%;border-collapse:collapse;font-size:.84rem}
.rules-table th{background:#0d1120;color:#64748b;font-size:.7rem;font-weight:600;
                text-transform:uppercase;letter-spacing:.1em;padding:.6rem 1rem;
                text-align:left;border-bottom:1px solid #1a1f35}
.rules-table td{padding:.7rem 1rem;color:#94a3b8;border-bottom:1px solid #0f1220}
.rules-table tr:last-child td{border-bottom:none}
.v-drift{color:#ff7096;font-weight:700} .v-pass{color:#06D6A0;font-weight:700}
hr{border:none;border-top:1px solid #1a1f35;margin:2rem 0}
footer{font-size:.76rem;color:#3a4a6a;text-align:center;margin-top:3rem}
</style>
"""


def _fig_to_html(fig: go.Figure, height: int = 420) -> str:
    """Convert a Plotly figure to an embedded HTML div (no external CDN needed)."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False},
        default_height=height,
    )


def generate_html_report(
    results: dict,
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    charts: dict,          # {name: plotly_figure}
    custom_rules: list = None,
) -> str:
    """
    Build a complete standalone HTML report.

    Parameters
    ----------
    results      : drift detection results dict
    baseline_df  : baseline DataFrame
    current_df   : current DataFrame
    charts       : dict of {section_name: go.Figure}
    custom_rules : list of rule result dicts from evaluate_rules()

    Returns a full HTML string — write to .html and share freely.
    """
    now      = datetime.now().strftime("%B %d, %Y  %H:%M")
    psi      = results.get("psi", {})
    wss      = results.get("wasserstein", {})
    ss       = results.get("summary_stats", pd.DataFrame())
    b_mean   = float(ss.loc["mean","Baseline"]) if "mean" in ss.index else 0
    c_mean   = float(ss.loc["mean","Current"])  if "mean" in ss.index else 0
    pct_chg  = ((c_mean / b_mean - 1) * 100) if b_mean > 0 else 0

    feature_map = {
        "purchase_amount":  ("💵", "Purchase Amount",  "KS Test"),
        "product_category": ("🛒", "Product Category", "Chi-Square"),
        "payment_method":   ("💳", "Payment Method",   "Chi-Square"),
    }

    # ── KPI strip ─────────────────────────────────────────────────────────────
    psi_color = psi.get("color", "#64748b")
    kpi_html  = f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">PSI Score</div>
        <div class="kpi-value" style="color:{psi_color}">{psi.get('psi',0):.4f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">PSI Level</div>
        <div class="kpi-value" style="font-size:1rem;color:{psi_color}">{psi.get('level','—')}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Wasserstein Distance</div>
        <div class="kpi-value" style="color:#FFB703">${wss.get('distance',0):.2f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Mean Spend Change</div>
        <div class="kpi-value" style="color:{'#EF476F' if pct_chg>0 else '#06D6A0'}">{pct_chg:+.1f}%</div>
      </div>
    </div>"""

    # ── Feature cards ─────────────────────────────────────────────────────────
    cards_html = '<div class="card-row">'
    for key, (icon, label, test) in feature_map.items():
        if key not in results:
            continue
        r     = results[key]
        drift = r["drift_detected"]
        cls   = "drift-card " + ("drifted" if drift else "stable")
        badge = "d-yes" if drift else "d-no"
        txt   = "⚠ Drift Detected" if drift else "✓ Stable"
        pv_c  = "#ff7096" if drift else "#06D6A0"
        cards_html += f"""
        <div class="{cls}">
          <div class="dc-feat">{icon} {label}</div>
          <div class="dc-test">{test}</div>
          <span class="badge {badge}">{txt}</span>
          <div class="dc-stats">
            p-value &nbsp;&nbsp;: <span style="color:{pv_c}">{r['p_value']:.6f}</span><br>
            statistic: <span>{r['statistic']:.4f}</span>
          </div>
        </div>"""
    cards_html += "</div>"

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_html = ""
    for section_name, fig in charts.items():
        if fig is None:
            continue
        charts_html += f"""
        <div class="section">
          <div class="sec-label">📊 {section_name}</div>
          <div class="chart-box">
            {_fig_to_html(fig, 440)}
          </div>
        </div>"""

    # ── Custom rules ─────────────────────────────────────────────────────────
    rules_html = ""
    if custom_rules:
        rows = ""
        for rule in custom_rules:
            v_cls = "v-drift" if rule["triggered"] else "v-pass"
            v_txt = "⚠ TRIGGERED" if rule["triggered"] else "✓ Pass"
            rows += (
                f"<tr><td>{rule['name']}</td>"
                f"<td>{rule['condition']}</td>"
                f"<td style='font-family:monospace'>{rule['actual']}</td>"
                f"<td class='{v_cls}'>{v_txt}</td></tr>"
            )
        rules_html = f"""
        <div class="section">
          <div class="sec-label">📏 Custom Drift Rules</div>
          <table class="rules-table">
            <thead><tr>
              <th>Rule name</th><th>Condition</th>
              <th>Actual value</th><th>Status</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # ── Dataset info ──────────────────────────────────────────────────────────
    b_start = baseline_df["Date"].min().date() if "Date" in baseline_df else "—"
    b_end   = baseline_df["Date"].max().date() if "Date" in baseline_df else "—"
    c_start = current_df["Date"].min().date()  if "Date" in current_df  else "—"
    c_end   = current_df["Date"].max().date()  if "Date" in current_df  else "—"

    # ── Assemble full HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drift Detection Report — {now}</title>
{_CSS}
</head>
<body>

<h1>Consumer Behavior Drift Detection</h1>
<div class="sub">
  Generated: {now} &nbsp;|&nbsp;
  Baseline: {b_start} → {b_end} ({len(baseline_df):,} rows) &nbsp;|&nbsp;
  Current: {c_start} → {c_end} ({len(current_df):,} rows)
</div>

<div class="section">
  <div class="sec-label">🔬 Drift Detection Results</div>
  {kpi_html}
  {cards_html}
</div>

<hr>
{charts_html}
{rules_html}

<hr>
<footer>
  Consumer Behavior Drift Detection Dashboard &nbsp;·&nbsp;
  Built with Python · scipy · Plotly &nbsp;·&nbsp; {now}
</footer>

</body>
</html>"""

    return html