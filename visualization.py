"""
visualization.py  —  Premium Plotly chart suite
All charts use a unified dark-glass design language:
  • Deep navy/slate background  (#0f1117 / #1a1d2e)
  • Neon accent palette  (electric blue #3A86FF, amber #FFB703, emerald #06D6A0, coral #EF476F)
  • Glassmorphism card feel via plot_bgcolor with subtle grid lines
  • Consistent font: "IBM Plex Mono" for numbers, "DM Sans" for labels
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde as _gaussian_kde

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_PAPER   = "#0f1117"
BG_PLOT    = "#1a1d2e"
GRID_COLOR = "#2a2d3e"
TEXT_COLOR = "#e2e8f0"
TEXT_DIM   = "#64748b"

BASELINE_COLOR = "#3A86FF"   # electric blue
CURRENT_COLOR  = "#FFB703"   # warm amber
COLORS = [
    "#3A86FF", "#FFB703", "#06D6A0", "#EF476F",
    "#9B5DE5", "#F15BB5", "#00BBF9", "#FB5607",
]

def _rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _base_layout(**kwargs) -> dict:
    """Return a fully styled layout dict, merging any extra kwargs."""
    layout = dict(
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_COLOR),
        hoverlabel=dict(
            bgcolor="#1e2235",
            bordercolor=BASELINE_COLOR,
            namelength=0,
            font=dict(size=13, color="#f1f5f9", family="IBM Plex Mono, monospace"),
        ),
        legend=dict(
            bgcolor="rgba(15,17,23,0.7)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11),
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR, gridwidth=1,
            linecolor=GRID_COLOR, tickcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR, size=12),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, gridwidth=1,
            linecolor=GRID_COLOR, tickcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR, size=12),
        ),
        margin=dict(t=70, r=30, b=60, l=60),
    )
    layout.update(kwargs)
    return layout

def _title(text: str) -> dict:
    return dict(
        text=text,
        font=dict(size=16, color=TEXT_COLOR, family="DM Sans, sans-serif"),
        x=0.5, xanchor="center", y=0.97,
    )

def _kde_curve(series, n_points=300):
    data = series.dropna().values
    kde  = _gaussian_kde(data)
    x    = np.linspace(data.min(), np.percentile(data, 99), n_points)
    return x, kde(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Purchase Amount Distribution
# ─────────────────────────────────────────────────────────────────────────────

def purchase_amount_distribution(baseline_df, current_df):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["  Density Curve (KDE)", "  Histogram"],
        horizontal_spacing=0.08,
    )

    for label, df, color in [("Baseline", baseline_df, BASELINE_COLOR),
                               ("Current",  current_df,  CURRENT_COLOR)]:
        x, y = _kde_curve(df["Purchase_Amount"])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=_rgba(color, 0.15),
            legendgroup=label,
            hovertemplate=(
                f"<b style='color:{color}'>{label}</b><br>"
                "Amount : <b>$%{x:.2f}</b><br>"
                "Density: <b>%{y:.6f}</b><extra></extra>"
            ),
        ), row=1, col=1)

    for label, df, color in [("Baseline", baseline_df, BASELINE_COLOR),
                               ("Current",  current_df,  CURRENT_COLOR)]:
        fig.add_trace(go.Histogram(
            x=df["Purchase_Amount"].dropna(), name=label,
            histnorm="probability density", nbinsx=40,
            marker=dict(color=color, opacity=0.5,
                        line=dict(color=BG_PAPER, width=0.5)),
            legendgroup=label, showlegend=False,
            hovertemplate=(
                f"<b style='color:{color}'>{label}</b><br>"
                "Range  : <b>$%{x}</b><br>"
                "Density: <b>%{y:.6f}</b><extra></extra>"
            ),
        ), row=1, col=2)

    layout = _base_layout(
        title=_title("Purchase Amount Distribution — Baseline vs Current"),
        barmode="overlay", height=440,
        legend=dict(
            bgcolor="rgba(15,17,23,0.7)", bordercolor=GRID_COLOR,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=11),
            orientation="h", x=0.5, xanchor="center", y=-0.15,
        ),
    )
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=TEXT_DIM, size=12))
    fig.update_xaxes(title_text="Purchase Amount ($)",
                     gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_DIM))
    fig.update_yaxes(title_text="Density",
                     gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_DIM))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Product Category Distribution
# ─────────────────────────────────────────────────────────────────────────────

def category_distribution(baseline_df, current_df):
    base_counts = baseline_df["Product_Category"].value_counts()
    curr_counts  = current_df["Product_Category"].value_counts()
    base_pct = (base_counts / base_counts.sum() * 100).rename("Baseline")
    curr_pct  = (curr_counts  / curr_counts.sum()  * 100).rename("Current")
    df_plot = pd.concat([base_pct, curr_pct], axis=1).fillna(0).sort_index()

    fig = go.Figure()
    for label, color, counts in [("Baseline", BASELINE_COLOR, base_counts),
                                   ("Current",  CURRENT_COLOR,  curr_counts)]:
        cats = df_plot.index.tolist()
        vals = df_plot[label].round(2).tolist()
        fig.add_trace(go.Bar(
            name=label, x=cats, y=vals,
            marker=dict(
                color=color, opacity=0.85,
                line=dict(color=BG_PAPER, width=1.5),
                pattern=dict(shape="" if label == "Baseline" else "/",
                             solidity=0.4),
            ),
            customdata=[int(counts.get(c, 0)) for c in cats],
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT_DIM, size=10),
            hovertemplate=(
                f"<b style='color:{color}'>%{{x}}</b><br>"
                f"Period : <b>{label}</b><br>"
                "Share  : <b>%{y:.2f}%</b><br>"
                "Count  : <b>%{customdata}</b><extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(
        title=_title("Product Category Distribution — Baseline vs Current"),
        barmode="group", height=460,
        xaxis=dict(title="", tickangle=-30, gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_DIM, size=10),
                   title_font=dict(color=TEXT_COLOR)),
        yaxis=dict(title="Share (%)", gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_DIM, size=10),
                   title_font=dict(color=TEXT_COLOR)),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Payment Method Distribution
# ─────────────────────────────────────────────────────────────────────────────

def payment_method_distribution(baseline_df, current_df):
    base_counts = baseline_df["Payment_Method"].value_counts()
    curr_counts  = current_df["Payment_Method"].value_counts()
    base_pct = (base_counts / base_counts.sum() * 100).rename("Baseline")
    curr_pct  = (curr_counts  / curr_counts.sum()  * 100).rename("Current")
    df_plot = pd.concat([base_pct, curr_pct], axis=1).fillna(0).sort_index()

    fig = go.Figure()
    for label, color, counts in [("Baseline", BASELINE_COLOR, base_counts),
                                   ("Current",  CURRENT_COLOR,  curr_counts)]:
        methods = df_plot.index.tolist()
        vals    = df_plot[label].round(2).tolist()
        fig.add_trace(go.Bar(
            name=label, x=methods, y=vals,
            marker=dict(
                color=color, opacity=0.85,
                line=dict(color=BG_PAPER, width=1.5),
                pattern=dict(shape="" if label == "Baseline" else "\\",
                             solidity=0.4),
            ),
            customdata=[int(counts.get(m, 0)) for m in methods],
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT_DIM, size=10),
            hovertemplate=(
                f"<b style='color:{color}'>%{{x}}</b><br>"
                f"Period : <b>{label}</b><br>"
                "Share  : <b>%{y:.2f}%</b><br>"
                "Count  : <b>%{customdata}</b><extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(
        title=_title("Payment Method Distribution — Baseline vs Current"),
        barmode="group", height=440,
        xaxis=dict(title="", tickangle=-15, gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_DIM, size=10),
                   title_font=dict(color=TEXT_COLOR)),
        yaxis=dict(title="Share (%)", gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_DIM, size=10),
                   title_font=dict(color=TEXT_COLOR)),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Monthly Average Purchase Amount Trend
# ─────────────────────────────────────────────────────────────────────────────

def purchase_amount_trend(combined_df):
    df = combined_df.copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly = (df.groupby(["Month", "Period"])["Purchase_Amount"]
                 .agg(Avg="mean", Min="min", Max="max", Count="count")
                 .reset_index())

    fig = go.Figure()

    for period, color in [("Baseline", BASELINE_COLOR), ("Current", CURRENT_COLOR)]:
        subset = monthly[monthly["Period"] == period].sort_values("Month")
        months = subset["Month"].tolist()
        avgs   = subset["Avg"].round(2).tolist()

        # Shaded fill (visual only)
        fig.add_trace(go.Scatter(
            x=months, y=avgs, mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=_rgba(color, 0.10),
            hoverinfo="skip", showlegend=False,
        ))

        # Range band (min–max envelope)
        fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=subset["Max"].round(2).tolist() + subset["Min"].round(2).tolist()[::-1],
            fill="toself", fillcolor=_rgba(color, 0.07),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))

        # Invisible large hit-target markers
        fig.add_trace(go.Scatter(
            x=months, y=avgs, name=period,
            mode="markers",
            marker=dict(size=20, opacity=0, color=color),
            customdata=np.stack([
                subset["Min"].round(2),
                subset["Max"].round(2),
                subset["Count"],
                avgs,
            ], axis=-1),
            hovertemplate=(
                f"<b style='color:{color}'>📅 %{{x}}</b><br>"
                "──────────────────<br>"
                f"Period      : <b>{period}</b><br>"
                "Avg Amount  : <b>$%{customdata[3]:.2f}</b><br>"
                "Min Amount  : <b>$%{customdata[0]:.2f}</b><br>"
                "Max Amount  : <b>$%{customdata[1]:.2f}</b><br>"
                "Transactions: <b>%{customdata[2]:.0f}</b>"
                "<extra></extra>"
            ),
        ))

        # Glowing dot on line
        fig.add_trace(go.Scatter(
            x=months, y=avgs, mode="markers",
            marker=dict(
                size=8, color=color,
                line=dict(color=BG_PAPER, width=2),
                symbol="circle",
            ),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(**_base_layout(
        title=_title("Monthly Average Purchase Amount — Time Series"),
        hovermode="closest",
        spikedistance=50,
        height=450,
        xaxis=dict(
            title="Month", tickangle=-40, gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR),
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor=GRID_COLOR,
            spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            title="Avg Purchase Amount ($)", gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR),
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor=GRID_COLOR,
            spikethickness=1, spikedash="dot",
            tickprefix="$",
        ),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Category Frequency Trend (stacked area)
# ─────────────────────────────────────────────────────────────────────────────

def category_frequency_trend(combined_df):
    df = combined_df.copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    pivot = (df.groupby(["Month", "Product_Category"])
               .size().unstack(fill_value=0).sort_index())
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_cum = pivot_pct.cumsum(axis=1)
    months    = pivot_pct.index.tolist()

    fig = go.Figure()

    for i, cat in enumerate(pivot_pct.columns):
        color     = COLORS[i % len(COLORS)]
        band_top  = pivot_cum[cat].round(2).tolist()
        band_h    = pivot_pct[cat].round(2).tolist()
        mid_y     = [(t - h / 2) for t, h in zip(band_top, band_h)]
        counts    = pivot[cat].tolist()

        # Stacked area band
        fig.add_trace(go.Scatter(
            x=months, y=band_h, name=cat,
            mode="lines", stackgroup="one",
            line=dict(color=color, width=0.5),
            fillcolor=_rgba(color, 0.75),
            hoverinfo="skip",
        ))

        # Invisible hit-target at midpoint of band
        fig.add_trace(go.Scatter(
            x=months, y=mid_y, name=cat,
            mode="markers",
            marker=dict(size=16, opacity=0, color=color),
            customdata=list(zip(band_h, counts)),
            hovertemplate=(
                f"<b style='color:{color}'>📅 %{{x}}</b><br>"
                "──────────────────<br>"
                f"Category : <b>{cat}</b><br>"
                "Share    : <b>%{customdata[0]:.2f}%</b><br>"
                "Count    : <b>%{customdata[1]}</b>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    layout = _base_layout(
        title=_title("Product Category Share — Monthly Trend"),
        hovermode="closest",
        spikedistance=60,
        height=470,
        margin=dict(t=70, r=180, b=60, l=60),
        xaxis=dict(
            title="Month", tickangle=-40, gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR),
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor=GRID_COLOR,
            spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            title="Share (%)", gridcolor=GRID_COLOR, range=[0, 100],
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_COLOR),
            ticksuffix="%",
        ),
        legend=dict(
            orientation="v", x=1.02, y=1,
            bgcolor="rgba(15,17,23,0.8)", bordercolor=GRID_COLOR,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=10),
        ),
    )
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Drift Summary Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def drift_summary_heatmap(drift_results):
    feature_labels = {
        "purchase_amount":  "Purchase Amount  (KS Test)",
        "product_category": "Product Category  (Chi-Square)",
        "payment_method":   "Payment Method  (Chi-Square)",
    }

    features, pvals, stats_vals, verdicts, tests = [], [], [], [], []
    for key, label in feature_labels.items():
        if key in drift_results:
            res = drift_results[key]
            features.append(label)
            pvals.append(res["p_value"])
            stats_vals.append(res["statistic"])
            verdicts.append("⚠ Drift Detected" if res["drift_detected"] else "✓ Stable")
            tests.append(res["test"])

    drift_colors  = ["#EF476F" if "Drift" in v else "#06D6A0" for v in verdicts]
    verdict_color = [["#3d0f1a" if "Drift" in v else "#0a2e20"] for v in verdicts]

    fig = go.Figure(data=[go.Table(
        columnwidth=[3.0, 1.2, 1.2, 1.5],
        header=dict(
            values=["<b>Feature</b>", "<b>p-value</b>",
                    "<b>Statistic</b>", "<b>Result</b>"],
            fill_color="#1e2235",
            font=dict(color=TEXT_COLOR, size=13, family="DM Sans"),
            align=["left", "center", "center", "center"],
            height=40,
            line=dict(color=GRID_COLOR, width=1),
        ),
        cells=dict(
            values=[
                features,
                [f"{p:.6f}" for p in pvals],
                [f"{s:.4f}" for s in stats_vals],
                verdicts,
            ],
            fill_color=[
                [BG_PLOT] * len(features),
                [BG_PLOT] * len(features),
                [BG_PLOT] * len(features),
                [c[0] for c in verdict_color],
            ],
            font=dict(
                size=12,
                family="IBM Plex Mono, monospace",
                color=[
                    [TEXT_COLOR] * len(features),
                    [TEXT_DIM]   * len(features),
                    [TEXT_DIM]   * len(features),
                    drift_colors,
                ],
            ),
            align=["left", "center", "center", "center"],
            height=36,
            line=dict(color=GRID_COLOR, width=1),
        ),
    )])

    fig.update_layout(
        paper_bgcolor=BG_PAPER,
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_COLOR),
        margin=dict(t=10, r=10, b=10, l=10),
        height=210,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Summary Statistics Table
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats_table(summary_df):
    STAT_INFO = {
        "count": ("Count",                             "Total number of transactions in this period."),
        "mean":  ("Mean  (Average)",                   "Sum of all amounts ÷ count. Shows the typical spend level."),
        "std":   ("Std  (Standard Deviation)",         "How spread out amounts are. High = wide variation from the mean."),
        "min":   ("Min  (Minimum)",                    "The smallest single purchase amount recorded."),
        "25%":   ("25th Percentile  (Q1)",             "25% of purchases fall below this value — lower end of spending."),
        "50%":   ("50th Percentile  (Median)",         "Middle value — better than mean for skewed data."),
        "75%":   ("75th Percentile  (Q3)",             "75% of purchases fall below this — upper end of spending."),
        "max":   ("Max  (Maximum)",                    "The largest single purchase amount recorded."),
    }

    rows       = summary_df.index.tolist()
    baseline   = summary_df["Baseline"].tolist()
    current    = summary_df["Current"].tolist()
    full_names = [STAT_INFO.get(r, (r, ""))[0] for r in rows]
    meanings   = [STAT_INFO.get(r, (r, ""))[1] for r in rows]

    base_fill  = ["#1a2744" if b > c else BG_PLOT for b, c in zip(baseline, current)]
    curr_fill  = ["#2a1e0a" if c > b else BG_PLOT for b, c in zip(baseline, current)]
    base_color = [BASELINE_COLOR if b > c else TEXT_DIM for b, c in zip(baseline, current)]
    curr_color = [CURRENT_COLOR  if c > b else TEXT_DIM for b, c in zip(baseline, current)]

    table_trace = go.Table(
        columnwidth=[2.8, 1.3, 1.3],
        header=dict(
            values=["<b>Statistic  ← hover for meaning</b>",
                    f"<b style='color:{BASELINE_COLOR}'>Baseline</b>",
                    f"<b style='color:{CURRENT_COLOR}'>Current</b>"],
            fill_color="#1e2235",
            font=dict(color=TEXT_COLOR, size=13, family="DM Sans"),
            align=["left", "center", "center"],
            height=40,
            line=dict(color=GRID_COLOR, width=1),
        ),
        cells=dict(
            values=[
                full_names,
                [f"{v:,.2f}" for v in baseline],
                [f"{v:,.2f}" for v in current],
            ],
            fill_color=[
                [BG_PLOT] * len(rows),
                base_fill,
                curr_fill,
            ],
            font=dict(
                size=12,
                family="IBM Plex Mono, monospace",
                color=[
                    [TEXT_COLOR] * len(rows),
                    base_color,
                    curr_color,
                ],
            ),
            align=["left", "center", "center"],
            height=36,
            line=dict(color=GRID_COLOR, width=1),
        ),
    )

    # Invisible scatter for accurate hover per row
    scatter_base = go.Scatter(
        x=baseline, y=full_names, mode="markers",
        marker=dict(size=22, opacity=0, color="rgba(0,0,0,0)"),
        customdata=list(zip(full_names, meanings,
                            [f"{v:,.2f}" for v in baseline],
                            [f"{v:,.2f}" for v in current])),
        hovertemplate=(
            f"<b style='color:{BASELINE_COLOR}'>%{{customdata[0]}}</b><br>"
            "<i style='color:#94a3b8'>%{customdata[1]}</i><br><br>"
            f"Baseline : <b style='color:{BASELINE_COLOR}'>%{{customdata[2]}}</b><br>"
            f"Current  : <b style='color:{CURRENT_COLOR}'>%{{customdata[3]}}</b>"
            "<extra></extra>"
        ),
        showlegend=False, xaxis="x", yaxis="y",
    )

    scatter_curr = go.Scatter(
        x=current, y=full_names, mode="markers",
        marker=dict(size=22, opacity=0, color="rgba(0,0,0,0)"),
        customdata=list(zip(full_names, meanings,
                            [f"{v:,.2f}" for v in baseline],
                            [f"{v:,.2f}" for v in current])),
        hovertemplate=(
            f"<b style='color:{CURRENT_COLOR}'>%{{customdata[0]}}</b><br>"
            "<i style='color:#94a3b8'>%{customdata[1]}</i><br><br>"
            f"Baseline : <b style='color:{BASELINE_COLOR}'>%{{customdata[2]}}</b><br>"
            f"Current  : <b style='color:{CURRENT_COLOR}'>%{{customdata[3]}}</b>"
            "<extra></extra>"
        ),
        showlegend=False, xaxis="x", yaxis="y",
    )

    fig = go.Figure(data=[table_trace, scatter_base, scatter_curr])
    fig.update_layout(
        paper_bgcolor=BG_PAPER,
        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT_COLOR),
        hoverlabel=dict(
            bgcolor="#1e2235", bordercolor=BASELINE_COLOR, namelength=0,
            font=dict(size=13, color="#f1f5f9", family="IBM Plex Mono, monospace"),
        ),
        margin=dict(t=10, r=10, b=10, l=10),
        height=340,
        hovermode="closest",
        xaxis=dict(
            visible=False,
            range=[min(min(baseline), min(current)) * 0.9,
                   max(max(baseline), max(current)) * 1.1],
            domain=[0, 1], fixedrange=True,
        ),
        yaxis=dict(
            visible=False, type="category",
            categoryarray=list(reversed(full_names)),
            domain=[0, 1], fixedrange=True,
        ),
    )
    return fig