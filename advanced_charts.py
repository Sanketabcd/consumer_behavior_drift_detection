"""
advanced_charts.py  —  Violin+Box, Sankey, Correlation Heatmap,
                        PSI Chart, Rolling Window Chart, Anomaly Annotations
All charts use the same dark design tokens as visualization.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BG_PAPER   = "#0f1117"
BG_PLOT    = "#1a1d2e"
GRID_COLOR = "#2a2d3e"
TEXT_COLOR = "#e2e8f0"
TEXT_DIM   = "#64748b"
BASELINE_COLOR = "#3A86FF"
CURRENT_COLOR  = "#FFB703"
DRIFT_COLOR    = "#EF476F"
STABLE_COLOR   = "#06D6A0"

COLORS = ["#3A86FF","#FFB703","#06D6A0","#EF476F","#9B5DE5","#F15BB5","#00BBF9","#FB5607"]

def _rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def _base_layout(**kw):
    d = dict(
        paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT,
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_COLOR),
        hoverlabel=dict(bgcolor="#1e2235", bordercolor=BASELINE_COLOR, namelength=0,
                        font=dict(size=13, color="#f1f5f9", family="IBM Plex Mono, monospace")),
        legend=dict(bgcolor="rgba(15,17,23,0.7)", bordercolor=GRID_COLOR, borderwidth=1,
                    font=dict(color=TEXT_COLOR, size=11)),
        margin=dict(t=70, r=30, b=60, l=60),
    )
    d.update(kw)
    return d

def _title(text):
    return dict(text=text, font=dict(size=16, color=TEXT_COLOR, family="DM Sans, sans-serif"),
                x=0.5, xanchor="center", y=0.97)

def _axis(title="", **kw):
    d = dict(title=title, gridcolor=GRID_COLOR, gridwidth=1,
             linecolor=GRID_COLOR, tickcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
             tickfont=dict(color=TEXT_DIM, size=10), title_font=dict(color=TEXT_COLOR, size=12))
    d.update(kw)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 1. Violin + Box Plot
# ─────────────────────────────────────────────────────────────────────────────

def violin_box_plot(baseline_df, current_df):
    fig = go.Figure()

    for label, df, color in [("Baseline", baseline_df, BASELINE_COLOR),
                               ("Current",  current_df,  CURRENT_COLOR)]:
        vals = df["Purchase_Amount"].dropna()
        fig.add_trace(go.Violin(
            y=vals, name=label,
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            line_color=color,
            fillcolor=_rgba(color, 0.18),
            marker=dict(color=color, size=3, opacity=0.5),
            box=dict(fillcolor=_rgba(color, 0.35), line_color=color, line_width=1.5),
            meanline=dict(color=color, width=2),
            hovertemplate=(
                f"<b style='color:{color}'>{label}</b><br>"
                "Value: <b>$%{y:.2f}</b><extra></extra>"
            ),
            customdata=[[label]] * len(vals),
        ))

    fig.update_layout(**_base_layout(
        title=_title("Distribution Shape — Violin + Box Plot"),
        yaxis=_axis("Purchase Amount ($)", tickprefix="$"),
        xaxis=_axis(),
        height=500,
        violinmode="group",
        violingap=0.3,
        violingroupgap=0.1,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sankey Flow Diagram
# ─────────────────────────────────────────────────────────────────────────────

def sankey_diagram(baseline_df, current_df):
    cats = sorted(set(baseline_df["Product_Category"].unique()) |
                  set(current_df["Product_Category"].unique()))
    pays = sorted(set(baseline_df["Payment_Method"].unique()) |
                  set(current_df["Payment_Method"].unique()))

    # Nodes: [Baseline period, ...categories..., ...payments..., Current period]
    node_labels = ["Baseline Period"] + cats + pays + ["Current Period"]
    n_cats = len(cats)
    n_pays = len(pays)
    idx_baseline = 0
    idx_cats  = {c: i+1 for i, c in enumerate(cats)}
    idx_pays  = {p: i+1+n_cats for i, p in enumerate(pays)}
    idx_current = 1 + n_cats + n_pays

    sources, targets, values, colors = [], [], [], []

    # Baseline → categories
    for cat in cats:
        cnt = (baseline_df["Product_Category"] == cat).sum()
        if cnt > 0:
            sources.append(idx_baseline)
            targets.append(idx_cats[cat])
            values.append(int(cnt))
            colors.append(_rgba(BASELINE_COLOR, 0.45))

    # Categories → payments (using combined data for flow)
    combined = pd.concat([baseline_df, current_df], ignore_index=True)
    for cat in cats:
        for pay in pays:
            cnt = ((combined["Product_Category"]==cat) & (combined["Payment_Method"]==pay)).sum()
            if cnt > 0:
                sources.append(idx_cats[cat])
                targets.append(idx_pays[pay])
                values.append(int(cnt))
                colors.append(_rgba(COLORS[cats.index(cat) % len(COLORS)], 0.35))

    # Payments → current
    for pay in pays:
        cnt = (current_df["Payment_Method"] == pay).sum()
        if cnt > 0:
            sources.append(idx_pays[pay])
            targets.append(idx_current)
            values.append(int(cnt))
            colors.append(_rgba(CURRENT_COLOR, 0.45))

    node_colors = (
        [BASELINE_COLOR] +
        [COLORS[i % len(COLORS)] for i in range(n_cats)] +
        [COLORS[(i+3) % len(COLORS)] for i in range(n_pays)] +
        [CURRENT_COLOR]
    )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
            line=dict(color=BG_PAPER, width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Flow: <b>%{value}</b><extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=colors,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                "Transactions: <b>%{value}</b><extra></extra>"
            ),
        ),
    ))

    fig.update_layout(**_base_layout(
        title=_title("Customer Flow — Category to Payment Method (Sankey)"),
        height=560,
        font_size=11,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def correlation_heatmap(baseline_df, current_df):
    """
    Fixed version — two bugs repaired:
    1. pd.Categorical gives DIFFERENT codes to same category in each df.
       Fix: build a shared category map from both datasets combined, then
       apply the same integer mapping to both so codes are consistent.
    2. Items_Bought column may not exist — handled gracefully.
    """

    def encode(df, cat_maps):
        """Encode categoricals using a pre-built shared mapping."""
        d = df.copy()
        for col, mapping in cat_maps.items():
            if col in d.columns:
                # Map using the shared dictionary — unknown values get -1
                d[col] = d[col].map(mapping).fillna(-1).astype(int)
        num_cols = [c for c in ["Purchase_Amount", "Product_Category",
                                 "Payment_Method", "Items_Bought"]
                    if c in d.columns]
        if len(num_cols) < 2:
            return None, num_cols
        return d[num_cols].corr().round(3), num_cols

    # Build one consistent mapping for each categorical column
    # using values from BOTH datasets combined
    cat_maps = {}
    for col in ["Product_Category", "Payment_Method"]:
        if col in baseline_df.columns or col in current_df.columns:
            all_vals = sorted(set(
                list(baseline_df[col].dropna().unique()) +
                list(current_df[col].dropna().unique())
            ))
            cat_maps[col] = {v: i for i, v in enumerate(all_vals)}

    base_corr, cols = encode(baseline_df, cat_maps)
    curr_corr, _    = encode(current_df,  cat_maps)

    # Guard: if encoding failed return an empty error figure
    if base_corr is None or curr_corr is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numeric columns to compute correlation.<br>"
                 "Need at least: Purchase_Amount + one categorical column.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=TEXT_COLOR, size=14),
        )
        fig.update_layout(**_base_layout(
            title=_title("Feature Correlations — Insufficient Data"),
            height=300,
        ))
        return fig

    # Align both matrices to the same columns (in case one dataset
    # was missing a column the other had)
    shared_cols = [c for c in base_corr.columns if c in curr_corr.columns]
    base_corr   = base_corr.loc[shared_cols, shared_cols]
    curr_corr   = curr_corr.loc[shared_cols, shared_cols]
    delta_corr  = (curr_corr - base_corr).round(3)

    col_labels = {
        "Purchase_Amount":  "Purchase $",
        "Product_Category": "Category",
        "Payment_Method":   "Payment",
        "Items_Bought":     "Items",
    }
    labels = [col_labels.get(c, c) for c in shared_cols]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["  Baseline Correlations",
                        "  Current Correlations",
                        "  Delta  (Current − Baseline)"],
        horizontal_spacing=0.08,
    )

    for col_idx, (mat, cscale, show_scale) in enumerate([
        (base_corr,  "RdBu", False),
        (curr_corr,  "RdBu", False),
        (delta_corr, "PiYG", True),
    ], 1):
        z    = mat.values.tolist()
        text = [[f"{v:.2f}" for v in row] for row in mat.values]

        # Text color: white on dark cells, dark on light cells
        text_colors = []
        for row in mat.values:
            text_colors.append(
                ["#ffffff" if abs(v) > 0.4 else TEXT_DIM for v in row]
            )

        fig.add_trace(go.Heatmap(
            z=z, x=labels, y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale=cscale,
            zmid=0, zmin=-1, zmax=1,
            showscale=show_scale,
            colorbar=dict(
                title=dict(text="Δ corr", font=dict(color=TEXT_DIM, size=10)),
                tickfont=dict(color=TEXT_DIM, size=9),
                thickness=14,
            ),
            xgap=3, ygap=3,
            hovertemplate=(
                "<b>%{y}  ×  %{x}</b><br>"
                "Correlation: <b>%{z:.3f}</b><br>"
                "<i>Range: −1 (opposite) → 0 (no link) → +1 (same direction)</i>"
                "<extra></extra>"
            ),
        ), row=1, col=col_idx)

    fig.update_layout(**_base_layout(
        title=_title("Feature Correlations — How Relationships Changed Between Periods"),
        height=440,
    ))
    fig.update_annotations(font=dict(color=TEXT_DIM, size=12))
    fig.update_xaxes(tickfont=dict(color=TEXT_DIM, size=10), tickangle=-15)
    fig.update_yaxes(tickfont=dict(color=TEXT_DIM, size=10))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PSI Score Chart
# ─────────────────────────────────────────────────────────────────────────────

def psi_chart(psi_result):
    bin_data = psi_result["bin_data"]
    psi_val  = psi_result["psi"]
    level    = psi_result["level"]
    color    = psi_result["color"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["  Bin Distribution (Baseline vs Current)",
                        "  PSI Contribution per Bin"],
        horizontal_spacing=0.10,
    )

    # Left: grouped bar — baseline vs current %
    fig.add_trace(go.Bar(
        x=bin_data["Bin"], y=bin_data["Baseline"],
        name="Baseline", marker=dict(color=BASELINE_COLOR, opacity=0.8,
                                      line=dict(color=BG_PAPER, width=1)),
        hovertemplate="Bin: <b>%{x}</b><br>Baseline: <b>%{y:.2f}%</b><extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=bin_data["Bin"], y=bin_data["Current"],
        name="Current", marker=dict(color=CURRENT_COLOR, opacity=0.8,
                                     line=dict(color=BG_PAPER, width=1)),
        hovertemplate="Bin: <b>%{x}</b><br>Current: <b>%{y:.2f}%</b><extra></extra>",
    ), row=1, col=1)

    # Right: PSI contribution bars (colored by +/-)
    psi_colors = [_rgba(DRIFT_COLOR, 0.8) if v > 0 else _rgba(STABLE_COLOR, 0.8)
                  for v in bin_data["PSI_contribution"]]
    fig.add_trace(go.Bar(
        x=bin_data["Bin"], y=bin_data["PSI_contribution"],
        name="PSI contribution",
        marker=dict(color=psi_colors, line=dict(color=BG_PAPER, width=1)),
        hovertemplate="Bin: <b>%{x}</b><br>Contribution: <b>%{y:.4f}</b><extra></extra>",
    ), row=1, col=2)

    # Horizontal threshold lines on right panel
    for thresh, label, lcolor in [(0.10, "Moderate (0.10)", "#FFB703"),
                                   (0.25, "Significant (0.25)", "#EF476F")]:
        fig.add_hline(y=thresh/len(bin_data), row=1, col=2,
                      line=dict(color=lcolor, dash="dash", width=1),
                      annotation_text=label,
                      annotation_font=dict(color=lcolor, size=9))

    psi_badge = f"PSI = {psi_val:.4f}  [{level}]"
    fig.update_layout(**_base_layout(
        title=_title(f"Population Stability Index — {psi_badge}"),
        barmode="group", height=440,
        xaxis=_axis("Percentile Bin"),
        yaxis=_axis("Share (%)"),
        xaxis2=_axis("Percentile Bin"),
        yaxis2=_axis("PSI Contribution"),
    ))
    fig.update_annotations(font=dict(color=TEXT_DIM, size=12))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rolling Drift Window Chart (with Anomaly Annotations)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_window_chart(rolling_df, anomaly_notes=None):
    """
    anomaly_notes: list of dicts {"date": str, "text": str, "y": float}
    from the AI annotation engine.
    """
    if rolling_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=_title("Rolling Drift Window — No data available"),
            height=400,
        ))
        return fig

    x = rolling_df["window_mid"].astype(str).tolist()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["  KS Statistic over Time (rolling 30-day window)",
                        "  Mean Purchase Amount: Window Baseline vs Current"],
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
    )

    # ── Top: KS statistic line ───────────────────────────────────────────────
    drift_mask   = rolling_df["drift"]
    stable_color = [DRIFT_COLOR if d else STABLE_COLOR for d in drift_mask]

    fig.add_trace(go.Scatter(
        x=x, y=rolling_df["ks_stat"].tolist(),
        mode="lines+markers", name="KS Statistic",
        line=dict(color=BASELINE_COLOR, width=2),
        marker=dict(size=8, color=stable_color,
                    line=dict(color=BG_PAPER, width=1.5)),
        customdata=np.stack([
            rolling_df["p_value"],
            rolling_df["wasserstein"],
            rolling_df["n_base"],
            rolling_df["n_curr"],
        ], axis=-1),
        hovertemplate=(
            "<b>📅 %{x}</b><br>"
            "KS Stat    : <b>%{y:.4f}</b><br>"
            "p-value    : <b>%{customdata[0]:.6f}</b><br>"
            "Wasserstein: <b>$%{customdata[1]:.2f}</b><br>"
            "N baseline : <b>%{customdata[2]:.0f}</b><br>"
            "N current  : <b>%{customdata[3]:.0f}</b>"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    # Significance threshold line
    fig.add_hline(y=0.05, row=1, col=1,
                  line=dict(color=DRIFT_COLOR, dash="dash", width=1.2),
                  annotation_text="α = 0.05",
                  annotation_font=dict(color=DRIFT_COLOR, size=10))

    # Shade drift regions
    prev_drift = False
    start_x = None
    for i, (xi, d) in enumerate(zip(x, drift_mask)):
        if d and not prev_drift:
            start_x = xi
        if not d and prev_drift and start_x:
            fig.add_vrect(x0=start_x, x1=x[i-1], row=1, col=1,
                          fillcolor=_rgba(DRIFT_COLOR, 0.08),
                          line_width=0, layer="below")
            start_x = None
        prev_drift = d
    if prev_drift and start_x:
        fig.add_vrect(x0=start_x, x1=x[-1], row=1, col=1,
                      fillcolor=_rgba(DRIFT_COLOR, 0.08),
                      line_width=0, layer="below")

    # ── Bottom: mean comparison ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=rolling_df["base_mean"].tolist(),
        mode="lines+markers", name="Window Baseline Mean",
        line=dict(color=BASELINE_COLOR, width=2, dash="dot"),
        marker=dict(size=6, color=BASELINE_COLOR),
        hovertemplate="<b>%{x}</b><br>Baseline mean: <b>$%{y:.2f}</b><extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=rolling_df["curr_mean"].tolist(),
        mode="lines+markers", name="Window Current Mean",
        line=dict(color=CURRENT_COLOR, width=2),
        marker=dict(size=6, color=CURRENT_COLOR),
        fill="tonexty", fillcolor=_rgba(CURRENT_COLOR, 0.07),
        hovertemplate="<b>%{x}</b><br>Current mean: <b>$%{y:.2f}</b><extra></extra>",
    ), row=2, col=1)

    # ── Anomaly annotations from AI ──────────────────────────────────────────
    if anomaly_notes:
        for ann in anomaly_notes:
            fig.add_annotation(
                x=str(ann["date"]), y=ann["y"],
                text=f"🔍 {ann['text']}",
                showarrow=True, arrowhead=2, arrowcolor=DRIFT_COLOR,
                arrowwidth=1.5, ax=0, ay=-40,
                font=dict(color=DRIFT_COLOR, size=10, family="IBM Plex Mono"),
                bgcolor=_rgba(DRIFT_COLOR, 0.12),
                bordercolor=DRIFT_COLOR, borderwidth=1,
                row=1, col=1,
            )

    fig.update_layout(**_base_layout(
        title=_title("Rolling Drift Window Analysis (30-day sliding window)"),
        height=580,
        hovermode="x unified",
        xaxis2=_axis("Window Midpoint Date", tickangle=-30),
        yaxis=_axis("KS Statistic"),
        yaxis2=_axis("Avg Purchase ($)", tickprefix="$"),
    ))
    fig.update_annotations(font=dict(color=TEXT_DIM, size=12))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Wasserstein Distance Gauge
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein_gauge(wass_result, baseline_mean):
    dist = wass_result["distance"]
    pct  = wass_result["pct_shift"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dist,
        number=dict(prefix="$", font=dict(size=36, color=TEXT_COLOR,
                                           family="IBM Plex Mono")),
        delta=dict(reference=0, increasing=dict(color=DRIFT_COLOR),
                   valueformat=".2f"),
        title=dict(text="Wasserstein Distance<br><span style='font-size:13px;color:#64748b'>"
                        f"Distribution shifted by ~${dist:.2f} avg ({pct:.1f}% of baseline mean)"
                        "</span>",
                   font=dict(size=15, color=TEXT_COLOR)),
        gauge=dict(
            axis=dict(range=[0, baseline_mean * 0.6],
                      tickwidth=1, tickcolor=TEXT_DIM,
                      tickfont=dict(color=TEXT_DIM, size=10)),
            bar=dict(color=CURRENT_COLOR, thickness=0.25),
            bgcolor=BG_PLOT,
            borderwidth=1, bordercolor=GRID_COLOR,
            steps=[
                dict(range=[0, baseline_mean*0.10], color=_rgba(STABLE_COLOR, 0.15)),
                dict(range=[baseline_mean*0.10, baseline_mean*0.25], color=_rgba("#FFB703", 0.15)),
                dict(range=[baseline_mean*0.25, baseline_mean*0.60], color=_rgba(DRIFT_COLOR, 0.15)),
            ],
            threshold=dict(line=dict(color=DRIFT_COLOR, width=2),
                           thickness=0.75, value=baseline_mean*0.25),
        ),
    ))

    fig.update_layout(
        paper_bgcolor=BG_PAPER,
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_COLOR),
        margin=dict(t=80, r=30, b=30, l=30),
        height=320,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Waterfall Chart — category share changes baseline → current
# ─────────────────────────────────────────────────────────────────────────────

def waterfall_chart(results):
    """
    Waterfall showing which categories gained (+) and lost (−) share.
    Hover shows exact percentage point change and direction.
    """
    pc = results.get("product_category", {})
    pm = results.get("payment_method", {})

    if not pc or "baseline_distribution" not in pc:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=_title("Waterfall — no data"), height=300))
        return fig, go.Figure()

    def _build(dist_b, dist_c, title_txt):
        cats   = sorted(set(list(dist_b.keys()) + list(dist_c.keys())))
        deltas = [round(dist_c.get(c, 0) - dist_b.get(c, 0), 2) for c in cats]
        colors = [DRIFT_COLOR if d < 0 else STABLE_COLOR for d in deltas]
        total  = round(sum(deltas), 2)

        # Sort by absolute delta descending
        pairs  = sorted(zip(cats, deltas, colors), key=lambda x: abs(x[1]), reverse=True)
        cats, deltas, colors = zip(*pairs) if pairs else ([], [], [])

        fig = go.Figure(go.Bar(
            x=list(cats),
            y=list(deltas),
            marker=dict(
                color=list(colors),
                opacity=0.85,
                line=dict(color=BG_PAPER, width=1.5),
            ),
            customdata=[[dist_b.get(c,0), dist_c.get(c,0), d]
                        for c, d in zip(cats, deltas)],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Baseline : <b>%{customdata[0]:.1f}%</b><br>"
                "Current  : <b>%{customdata[1]:.1f}%</b><br>"
                "Change   : <b>%{customdata[2]:+.2f} pp</b>"
                "<extra></extra>"
            ),
            text=[f"{d:+.1f}" for d in deltas],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_COLOR),
        ))

        # Zero line
        fig.add_hline(y=0, line=dict(color=GRID_COLOR, width=1.5, dash="solid"))

        fig.update_layout(**_base_layout(
            title=_title(title_txt),
            height=420,
            xaxis=_axis("", tickangle=-25),
            yaxis=_axis("Change (percentage points)", ticksuffix=" pp"),
        ))
        return fig

    fig_cat = _build(
        pc["baseline_distribution"], pc["current_distribution"],
        "Category Share Change  (Baseline → Current)"
    )
    fig_pay = _build(
        pm.get("baseline_distribution", {}), pm.get("current_distribution", {}),
        "Payment Method Share Change  (Baseline → Current)"
    )
    return fig_cat, fig_pay


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Drill-Down chart — click a category bar to see its purchase distribution
# ─────────────────────────────────────────────────────────────────────────────

def drilldown_distribution(baseline_df, current_df, selected_category=None):
    """
    When a category is selected, shows the purchase amount KDE for that
    category only — baseline vs current. Reveals within-category drift.
    """
    from scipy.stats import gaussian_kde as _gkde
    import numpy as np

    if not selected_category:
        fig = go.Figure()
        fig.add_annotation(
            text="← Click a category bar above to drill into it",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=TEXT_DIM, size=14, family="DM Sans, sans-serif"),
        )
        fig.update_layout(**_base_layout(
            title=_title("Drill-Down — Purchase Distribution by Category"),
            height=320,
        ))
        return fig

    b_vals = baseline_df[baseline_df["Product_Category"] == selected_category]["Purchase_Amount"].dropna()
    c_vals = current_df[current_df["Product_Category"] == selected_category]["Purchase_Amount"].dropna()

    fig = go.Figure()

    for label, vals, color in [("Baseline", b_vals, BASELINE_COLOR),
                                ("Current",  c_vals, CURRENT_COLOR)]:
        if len(vals) < 3:
            continue
        kde  = _gkde(vals)
        xr   = np.linspace(vals.min(), np.percentile(vals, 99), 300)
        yr   = kde(xr)
        fig.add_trace(go.Scatter(
            x=xr, y=yr, name=label,
            mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=_rgba(color, 0.20),
            hovertemplate=(
                f"<b>{label} — {selected_category}</b><br>"
                "Amount : <b>$%{x:.2f}</b><br>"
                "Density: <b>%{y:.6f}</b><extra></extra>"
            ),
        ))

    b_mean = b_vals.mean() if len(b_vals) else 0
    c_mean = c_vals.mean() if len(c_vals) else 0
    pct    = ((c_mean / b_mean - 1) * 100) if b_mean > 0 else 0

    fig.update_layout(**_base_layout(
        title=_title(
            f"Drill-Down: {selected_category}  "
            f"(mean ${b_mean:.2f} → ${c_mean:.2f},  {pct:+.1f}%)"
        ),
        height=360,
        xaxis=_axis("Purchase Amount ($)", tickprefix="$"),
        yaxis=_axis("Density"),
        legend=dict(
            bgcolor="rgba(15,17,23,0.7)", bordercolor=GRID_COLOR,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=11),
            orientation="h", x=0.5, xanchor="center", y=-0.18,
        ),
    ))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ML Drift Charts
# ─────────────────────────────────────────────────────────────────────────────

def ml_model_comparison_chart(ml_results: dict) -> go.Figure:
    """Bar chart comparing drift scores from all 3 ML models + ensemble."""
    models = ["Isolation Forest", "Random Forest", "Gradient Boosting", "Ensemble"]
    scores = [
        ml_results["isolation_forest"]["drift_score"],
        ml_results["random_forest"]["drift_score"],
        ml_results["gradient_boosting"]["drift_score"],
        ml_results["ensemble_score"],
    ]
    colors_list = [
        ml_results["isolation_forest"]["color"],
        ml_results["random_forest"]["color"],
        ml_results["gradient_boosting"]["color"],
        ml_results["ensemble_color"],
    ]
    auc_labels = [
        f"Anomaly rate: {ml_results['isolation_forest']['anomaly_rate']:.2%}",
        f"AUC: {ml_results['random_forest']['auc']:.4f}",
        f"AUC: {ml_results['gradient_boosting']['auc']:.4f}",
        f"Score: {ml_results['ensemble_score']:.4f}",
    ]

    fig = go.Figure(go.Bar(
        x=models, y=scores,
        marker=dict(color=colors_list, opacity=0.85,
                    line=dict(color=BG_PAPER, width=1.5)),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=12, color=TEXT_COLOR),
        customdata=auc_labels,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Drift Score: <b>%{y:.4f}</b><br>"
            "%{customdata}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.5, line=dict(color="#FFB703", width=1.5, dash="dash"),
                  annotation_text="Drift threshold (0.5)",
                  annotation_font=dict(color="#FFB703", size=10))
    fig.update_layout(**_base_layout(
        title=_title("ML Model Drift Scores — All Models vs Ensemble"),
        height=400,
        yaxis=_axis("Drift Score (0 = stable, 1 = fully drifted)", range=[0, 1.1]),
        xaxis=_axis(""),
    ))
    return fig


def ml_feature_importance_chart(ml_results: dict) -> go.Figure:
    """Grouped bar showing feature importance from RF and GBT."""
    pf    = ml_results["per_feature"]
    feats = list(pf.keys())
    rf_scores  = [pf[f]["drift_score"] for f in feats]
    rf_aucs    = [pf[f]["auc"] for f in feats]
    colors_f   = [pf[f]["color"] for f in feats]

    # Also show RF classifier importances
    rf_imp = ml_results["random_forest"].get("feature_importance", {})
    gb_imp = ml_results["gradient_boosting"].get("feature_importance", {})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RF Drift Score", x=feats, y=rf_scores,
        marker=dict(color=colors_f, opacity=0.8),
        text=[f"{s:.3f}" for s in rf_scores],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Drift Score: <b>%{y:.4f}</b><br>AUC: <b>" +
                      "</b><extra></extra>",
    ))
    if rf_imp:
        fig.add_trace(go.Bar(
            name="RF Feature Weight", x=feats,
            y=[rf_imp.get(f, 0) for f in feats],
            marker=dict(color=BASELINE_COLOR, opacity=0.6),
            hovertemplate="<b>%{x}</b><br>RF Importance: <b>%{y:.4f}</b><extra></extra>",
        ))
    if gb_imp:
        fig.add_trace(go.Bar(
            name="GBT Feature Weight", x=feats,
            y=[gb_imp.get(f, 0) for f in feats],
            marker=dict(color=CURRENT_COLOR, opacity=0.6),
            hovertemplate="<b>%{x}</b><br>GBT Importance: <b>%{y:.4f}</b><extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title=_title("Feature-Level Drift (per-feature ML classifier AUC)"),
        height=420,
        barmode="group",
        yaxis=_axis("Score"),
        xaxis=_axis(""),
    ))
    return fig


def ml_anomaly_score_chart(ml_results: dict, baseline_df, current_df) -> go.Figure:
    """Scatter of Isolation Forest anomaly scores — baseline vs current."""
    row_scores = ml_results["isolation_forest"].get("row_scores", None)
    if row_scores is None or row_scores.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title=_title("Anomaly Scores — no data"), height=300))
        return fig

    for period, color in [("Baseline", BASELINE_COLOR), ("Current", CURRENT_COLOR)]:
        sub = row_scores[row_scores["period"] == period]
        fig_data = sub["anomaly_score"].values
        fig = None  # reset

    fig = go.Figure()
    for period, color, opacity in [("Baseline", BASELINE_COLOR, 0.5),
                                    ("Current",  CURRENT_COLOR, 0.8)]:
        sub = row_scores[row_scores["period"] == period]
        fig.add_trace(go.Histogram(
            x=sub["anomaly_score"], name=period,
            marker=dict(color=color, opacity=opacity,
                        line=dict(color=BG_PAPER, width=0.5)),
            nbinsx=40,
            hovertemplate=f"<b>{period}</b><br>Anomaly Score: <b>%{{x:.3f}}</b><br>Count: <b>%{{y}}</b><extra></extra>",
        ))

    # Threshold line
    threshold = row_scores[row_scores["period"]=="Baseline"]["anomaly_score"].quantile(0.95)
    fig.add_vline(x=threshold, line=dict(color="#EF476F", width=1.5, dash="dash"),
                  annotation_text="Anomaly threshold (95th pct baseline)",
                  annotation_font=dict(color="#EF476F", size=10))

    fig.update_layout(**_base_layout(
        title=_title("Isolation Forest Anomaly Score Distribution"),
        height=400,
        barmode="overlay",
        xaxis=_axis("Anomaly Score (higher = more anomalous)"),
        yaxis=_axis("Count"),
        legend=dict(bgcolor="rgba(15,17,23,0.7)", bordercolor=GRID_COLOR,
                    borderwidth=1, font=dict(color=TEXT_COLOR, size=11)),
    ))
    return fig


def ml_drift_gauge(ml_results: dict) -> go.Figure:
    """Gauge showing the ensemble drift score 0-100."""
    score = ml_results["ensemble_score"] * 100
    color = ml_results["ensemble_color"]
    level = ml_results["ensemble_level"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 1),
        title=dict(text=f"Drift Risk Score<br><span style='font-size:0.85em;color:{color}'>{level}</span>",
                   font=dict(color=TEXT_COLOR, size=14)),
        delta=dict(reference=50, increasing=dict(color=DRIFT_COLOR),
                   decreasing=dict(color=STABLE_COLOR)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_DIM,
                      tickfont=dict(color=TEXT_DIM, size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=BG_PAPER,
            bordercolor=GRID_COLOR,
            steps=[
                dict(range=[0,  50],  color=_rgba(STABLE_COLOR, 0.12)),
                dict(range=[50, 70],  color=_rgba("#FFB703", 0.12)),
                dict(range=[70, 85],  color=_rgba("#FF7B00", 0.12)),
                dict(range=[85, 100], color=_rgba(DRIFT_COLOR, 0.12)),
            ],
            threshold=dict(line=dict(color="#EF476F", width=2), value=70),
        ),
        number=dict(suffix="%", font=dict(color=TEXT_COLOR, size=36,
                                           family="IBM Plex Mono")),
    ))
    fig.update_layout(**_base_layout(
        title=_title("Ensemble ML Drift Risk Gauge"),
        height=380,
    ))
    return fig