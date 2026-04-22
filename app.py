"""
app.py  —  Consumer Behavior Drift Detection  |  Next-Level Dashboard
Features: PSI · Wasserstein · Rolling Window · Violin+Box · Sankey ·
Correlation Heatmap · PDF Export · CSV Upload · Date Picker + Threshold Slider
"""

import os, sys, io
import pandas as pd
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from drift_detection  import run_all_drift_checks, rolling_drift_window
from ml_drift_engine  import run_ml_drift_checks
from visualization   import (
    purchase_amount_distribution, category_distribution,
    payment_method_distribution, purchase_amount_trend,
    category_frequency_trend, drift_summary_heatmap, summary_stats_table,
)
# advanced_charts imported above
from pdf_report       import generate_pdf, REPORTLAB_OK
from smart_mapper     import smart_load, mapping_summary_html
from advanced_charts  import (violin_box_plot, sankey_diagram, correlation_heatmap,
                              ml_model_comparison_chart, ml_feature_importance_chart,
                              ml_anomaly_score_chart, ml_drift_gauge,
                              psi_chart, rolling_window_chart, wasserstein_gauge,
                              waterfall_chart, drilldown_distribution)
from custom_rules     import evaluate_rules, get_categories, get_payment_methods, RULE_TYPES
from report_generator import generate_html_report
from bulk_scanner     import run_bulk_scan
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Detection · Analytics",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme session state ──────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'
_IS_LIGHT = (st.session_state['theme'] == 'light')

# ── CSS ───────────────────────────────────────────────────────────────────────────
# Theme colour tokens — switched by session_state
if _IS_LIGHT:
    _BG       = "#f0f4fb";  _BG2   = "#e8edf8";  _BG3  = "#ffffff"
    _TEXT     = "#1a2035";  _DIM   = "#4a5a80";  _MID  = "#3a4a70"
    _BORDER   = "#c8d4ee";  _BOR2  = "#b8c8e4"
    _SB_BG    = "linear-gradient(180deg,#e0e8f8,#edf2fc)"
    _CARD_BG  = "linear-gradient(160deg,#f4f7fd,#ffffff)"
    _MET_BG   = "linear-gradient(135deg,#ffffff,#f0f4fc)"
    _TAB_BG   = "#e0e8f8";  _PANEL = "#f4f7fd";  _CHART = "#ffffff"
    _EXP_BG   = "#f4f7fd";  _KPI   = "#f0f4fc"
    _ABOUT_BG = "linear-gradient(135deg,#e8edf8,#f0f4fc,#e8f4ff)"
    _ABOUT_BR = "#c8d4ee";  _ABOUT_D= "#3a4a70"
    _FEAT_BG  = "#ffffff";  _H_COL = "#185FA5"
    _H_GRAD   = "linear-gradient(180deg,#185FA5,#534AB7)"
    _HR       = "#c8d4ee";  _H2    = "#1a2035";  _H3   = "#3a4a70"
    _DC_NAME  = "#1a2035";  _STATS = "#4a5a80";  _SPAN = "#2a3a60"
    _FBAR_BG  = "#d8e4f4";  _KS_V  = "#1a2035"
    _CHDR_BG  = "#e0e8f4";  _CHDR  = "#1a2035"
    _CROW     = "#edf2fc";  _COMP  = "#f8faff"
    _PILL_BG  = "#f0f4fc";  _PILL_B= "#c8d4ee"; _PILL_C= "#3a4a70"
    _UPLOAD   = "#e8edf8"
else:
    _BG       = "#080c14";  _BG2   = "#0d1120";  _BG3  = "#0e1428"
    _TEXT     = "#e2e8f0";  _DIM   = "#3a4a6a";  _MID  = "#94a3b8"
    _BORDER   = "#1a1f35";  _BOR2  = "#1e2545"
    _SB_BG    = "linear-gradient(180deg,#080c14,#0d1120)"
    _CARD_BG  = "linear-gradient(160deg,#0e1428,#111a30)"
    _MET_BG   = "linear-gradient(135deg,#10152a,#131828)"
    _TAB_BG   = "#0d1120";  _PANEL = "#0a0e1a";  _CHART = "#080c14"
    _EXP_BG   = "#0d1120";  _KPI   = "#0d1120"
    _ABOUT_BG = "linear-gradient(135deg,#08111f,#0d1628,#091420)"
    _ABOUT_BR = "#1a2d4a";  _ABOUT_D= "#7a8ab0"
    _FEAT_BG  = "#0e1428";  _H_COL = "#3A86FF"
    _H_GRAD   = "linear-gradient(180deg,#3A86FF,#9B5DE5)"
    _HR       = "#1a1f35";  _H2    = "#b8c8e8";  _H3   = "#7a8ab0"
    _DC_NAME  = "#c8d4f0";  _STATS = "#3a4a6a";  _SPAN = "#94a3b8"
    _FBAR_BG  = "#1a1f35";  _KS_V  = "#e2e8f0"
    _CHDR_BG  = "#0d1120";  _CHDR  = "#64748b"
    _CROW     = "#12182a";  _COMP  = "#080c14"
    _PILL_BG  = "#0d1120";  _PILL_B= "#1a1f35"; _PILL_C= "#4a5a80"
    _UPLOAD   = "#0d1120"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body {{ font-family:'DM Sans',sans-serif; background:{_BG}; color:{_TEXT}; }}
.stApp {{ background:{_BG}; }}
#MainMenu {{ visibility:hidden; }}
footer    {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}

[data-testid="collapsedControl"] {{
    visibility:visible !important; display:flex !important; opacity:1 !important;
    pointer-events:auto !important; position:fixed !important;
    left:0 !important; top:50vh !important; transform:translateY(-50%) !important;
    z-index:999999 !important; width:28px !important; height:54px !important;
    background:#12162a !important; border:1px solid #3A86FF !important;
    border-left:none !important; border-radius:0 10px 10px 0 !important;
    box-shadow:4px 0 20px rgba(58,134,255,0.35) !important;
    cursor:pointer !important; align-items:center !important; justify-content:center !important;
    transition:background .2s, width .2s !important;
}}
[data-testid="collapsedControl"]:hover {{ background:#3A86FF !important; width:34px !important; }}
[data-testid="collapsedControl"] svg {{ fill:#3A86FF !important; width:16px !important; height:16px !important; }}
[data-testid="collapsedControl"]:hover svg {{ fill:#ffffff !important; }}
[data-testid="stSidebarCollapseButton"] button {{
    visibility:visible !important; opacity:1 !important;
    background:{_BG2} !important; border:1px solid {_BORDER} !important; border-radius:8px !important;
}}
[data-testid="stSidebarCollapseButton"] button:hover {{ background:#3A86FF !important; border-color:#3A86FF !important; }}
[data-testid="stSidebarCollapseButton"] button svg {{ fill:#7a8ab0 !important; }}
[data-testid="stSidebarCollapseButton"] button:hover svg {{ fill:#ffffff !important; }}

.block-container {{ padding:1.8rem 2.5rem !important; max-width:100% !important; }}

[data-testid="stSidebar"] {{ background:{_SB_BG} !important; border-right:1px solid {_BORDER} !important; }}
[data-testid="stSidebar"] * {{ color:{_MID} !important; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color:{_TEXT} !important; }}
[data-testid="stSidebar"] .stMarkdown strong {{ color:{_TEXT} !important; }}
[data-testid="stSidebar"] hr {{ border-color:{_BORDER} !important; }}

[data-testid="metric-container"] {{ background:{_MET_BG}; border:1px solid {_BOR2}; border-radius:14px; padding:1.1rem 1.3rem; transition:border-color .2s,box-shadow .2s; }}
[data-testid="metric-container"]:hover {{ border-color:#3A86FF; box-shadow:0 0 20px rgba(58,134,255,0.12); }}
[data-testid="stMetricLabel"] {{ color:{_DIM} !important; font-size:.72rem !important; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }}
[data-testid="stMetricValue"] {{ color:{_TEXT} !important; font-family:'IBM Plex Mono',monospace !important; font-size:1.7rem !important; font-weight:500; }}

h1 {{ font-size:1.75rem !important; font-weight:700 !important; background:linear-gradient(135deg,#3A86FF,#9B5DE5); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em; }}
h2 {{ color:{_H2} !important; font-size:1.05rem !important; font-weight:600 !important; }}
h3 {{ color:{_H3} !important; font-size:.9rem !important; }}
hr {{ border-color:{_HR} !important; margin:1.8rem 0; }}
p  {{ color:{_MID}; line-height:1.7; }}

[data-baseweb="tab-list"] {{ background:{_TAB_BG}; border-radius:10px; padding:3px; border:1px solid {_BORDER}; gap:3px; }}
[data-baseweb="tab"] {{ border-radius:8px !important; color:{_DIM} !important; font-weight:500 !important; font-size:.85rem !important; padding:7px 18px !important; transition:all .15s !important; }}
[data-baseweb="tab"]:hover {{ color:{_MID} !important; background:rgba(58,134,255,0.06) !important; }}
[aria-selected="true"] {{ background:linear-gradient(135deg,#1e3a7a,#2a2a7a) !important; color:#7eb3ff !important; box-shadow:0 0 12px rgba(58,134,255,0.2) !important; }}
[data-baseweb="tab-panel"] {{ background:{_PANEL}; border-radius:0 0 14px 14px; border:1px solid {_BORDER}; border-top:none; padding:1.4rem !important; }}

[data-testid="stPlotlyChart"] {{ border:1px solid {_BORDER}; border-radius:16px; overflow:hidden; background:{_CHART}; box-shadow:0 4px 24px rgba(0,0,0,0.25); }}
[data-testid="stExpander"] {{ background:{_EXP_BG}; border:1px solid {_BORDER} !important; border-radius:12px; }}
[data-testid="stExpander"] summary {{ color:{_MID} !important; }}
[data-testid="stExpander"] summary:hover {{ color:{_TEXT} !important; }}
.stCaption {{ color:{_DIM} !important; font-size:.76rem !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {_BORDER} !important; border-radius:10px !important; overflow:hidden; }}

.drift-card {{ background:{_CARD_BG}; border:1px solid {_BOR2}; border-radius:16px; padding:1.3rem 1.5rem; transition:all .25s; height:100%; position:relative; overflow:hidden; }}
.drift-card::before {{ content:""; position:absolute; top:0; left:0; right:0; height:2px; border-radius:16px 16px 0 0; }}
.drift-card.drifted::before {{ background:linear-gradient(90deg,#EF476F,#ff6b8a); }}
.drift-card.stable::before  {{ background:linear-gradient(90deg,#06D6A0,#00f0b0); }}
.drift-card:hover {{ border-color:#2a3560; transform:translateY(-3px); box-shadow:0 12px 32px rgba(0,0,0,0.3); }}
.drift-card.drifted:hover {{ box-shadow:0 12px 32px rgba(239,71,111,0.15); }}
.drift-card.stable:hover  {{ box-shadow:0 12px 32px rgba(6,214,160,0.12); }}
.dc-icon-row {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:.7rem; }}
.dc-test-pill {{ font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; padding:2px 9px; border-radius:20px; background:rgba(58,134,255,0.1); color:#3A86FF; border:1px solid rgba(58,134,255,0.2); }}
.dc-label {{ font-size:.68rem; color:{_DIM}; text-transform:uppercase; letter-spacing:.12em; font-weight:600; margin-bottom:.4rem; }}
.dc-test-name {{ font-size:1.1rem; font-weight:600; color:{_DC_NAME}; margin-bottom:.9rem; font-family:'IBM Plex Mono',monospace; }}
.drift-badge {{ display:inline-flex; align-items:center; gap:5px; padding:5px 13px; border-radius:20px; font-size:.74rem; font-weight:700; margin-bottom:.9rem; }}
.drift-yes {{ background:rgba(239,71,111,.12); color:#ff7096; border:1px solid rgba(239,71,111,.25); }}
.drift-no  {{ background:rgba(6,214,160,.10); color:#06D6A0; border:1px solid rgba(6,214,160,.2); }}
.dc-stats {{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:{_STATS}; line-height:2; border-top:1px solid {_BORDER}; padding-top:.7rem; margin-top:.3rem; }}
.dc-stats span {{ color:{_SPAN}; font-weight:500; }}

.section-heading {{ display:flex; align-items:center; gap:10px; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:{_H_COL}; margin-bottom:1.2rem; margin-top:.3rem; }}
.section-heading::before {{ content:""; display:block; width:3px; height:16px; background:{_H_GRAD}; border-radius:2px; flex-shrink:0; }}
.sh-icon {{ font-size:.9rem; }}

.kpi-strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1rem 0 1.4rem; }}
.kpi-strip-card {{ background:{_KPI}; border:1px solid {_BORDER}; border-radius:12px; padding:.9rem 1.1rem; transition:border-color .2s; }}
.kpi-strip-card:hover {{ border-color:#3A86FF; }}
.ks-label {{ font-size:.66rem; color:{_DIM}; text-transform:uppercase; letter-spacing:.12em; font-weight:600; margin-bottom:.4rem; }}
.ks-value {{ font-size:1.35rem; font-family:'IBM Plex Mono',monospace; color:{_KS_V}; font-weight:500; }}
.ks-sub   {{ font-size:.7rem; color:{_DIM}; margin-top:.2rem; }}

.about-hero {{ background:{_ABOUT_BG}; border:1px solid {_ABOUT_BR}; border-radius:18px; padding:1.8rem 2rem; margin-bottom:1.6rem; }}
.about-tag {{ display:inline-flex; align-items:center; gap:6px; font-size:.66rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; color:#3A86FF; background:rgba(58,134,255,.08); border:1px solid rgba(58,134,255,.2); border-radius:20px; padding:3px 12px; margin-bottom:.9rem; }}
.about-desc {{ font-size:.93rem; color:{_ABOUT_D}; line-height:1.8; max-width:880px; margin-bottom:1.2rem; }}
.about-desc b {{ color:{_DC_NAME}; font-weight:600; }}
.about-divider {{ width:100%; height:1px; background:linear-gradient(90deg,transparent,{_BORDER} 30%,{_BORDER} 70%,transparent); margin:1rem 0; }}
.about-pills {{ display:flex; flex-wrap:wrap; gap:8px; }}
.about-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:.74rem; font-weight:500; color:{_PILL_C}; background:{_PILL_BG}; border:1px solid {_PILL_B}; border-radius:8px; padding:5px 12px; transition:all .2s; }}
.about-pill:hover {{ border-color:#3A86FF; color:{_TEXT}; }}
.pill-dot {{ width:6px; height:6px; border-radius:50%; flex-shrink:0; }}

.feat-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:.8rem 0 1.4rem; }}
.feat-card {{ background:{_FEAT_BG}; border:1px solid {_BOR2}; border-radius:14px; padding:1.1rem 1.2rem; transition:all .22s; }}
.feat-card:hover {{ border-color:#3A86FF; transform:translateY(-2px); }}
.feat-rank {{ font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:{_H_COL}; margin-bottom:.4rem; }}
.feat-name {{ font-size:1rem; font-weight:600; color:{_DC_NAME}; margin-bottom:.8rem; }}
.feat-bar-bg {{ background:{_FBAR_BG}; border-radius:20px; height:6px; margin-bottom:.7rem; overflow:hidden; }}
.feat-bar-fill {{ height:6px; border-radius:20px; transition:width .6s ease; }}
.feat-stats {{ font-family:'IBM Plex Mono',monospace; font-size:.74rem; color:{_STATS}; line-height:1.9; }}
.feat-stats span {{ color:{_SPAN}; }}
.feat-badge {{ display:inline-flex; align-items:center; gap:4px; font-size:.66rem; font-weight:700; padding:2px 9px; border-radius:20px; margin-bottom:.6rem; }}
.feat-drift  {{ background:rgba(239,71,111,.12); color:#ff7096; border:1px solid rgba(239,71,111,.25); }}
.feat-stable {{ background:rgba(6,214,160,.10); color:#06D6A0; border:1px solid rgba(6,214,160,.2); }}

.comp-wrap {{ border:1px solid {_BORDER}; border-radius:14px; overflow:hidden; margin:.8rem 0; }}
.comp-table {{ width:100%; border-collapse:collapse; font-family:'DM Sans',sans-serif; background:{_COMP}; }}
.comp-header {{ background:{_CHDR_BG}; color:{_CHDR}; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; padding:.7rem 1rem; text-align:left; border-bottom:1px solid {_BORDER}; }}
.comp-row {{ border-bottom:1px solid {_BORDER}; transition:background .15s; }}
.comp-row:last-child {{ border-bottom:none; }}
.comp-row:hover {{ background:{_CROW}; }}
.comp-cell {{ padding:.75rem 1rem; font-size:.84rem; color:{_MID}; vertical-align:middle; }}
.comp-cell.feat-n {{ color:{_DC_NAME}; font-weight:600; }}
.comp-cell.num {{ text-align:right; font-family:'IBM Plex Mono',monospace; font-size:.8rem; }}
.comp-mini-bar {{ display:inline-block; height:4px; border-radius:4px; vertical-align:middle; margin-left:8px; }}
.comp-verdict-drift  {{ color:#ff7096; font-weight:700; }}
.comp-verdict-stable {{ color:#06D6A0; font-weight:700; }}
.upload-zone {{ background:{_UPLOAD}; border:2px dashed {_BORDER}; border-radius:14px; padding:2rem; text-align:center; margin:.5rem 0; }}
</style>
""", unsafe_allow_html=True)





# ── Helpers ───────────────────────────────────────────────────────────────────
def show_chart(fig):
    if isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d","select2d","autoScale2d"],
            "displaylogo": False,
            "toImageButtonOptions": {"format":"svg","scale":2},
        })
    else:
        st.pyplot(fig)

@st.cache_data(show_spinner="Loading dataset…")
def load_default_data():
    data_dir = os.path.join(ROOT, "data")
    for p in ["baseline_data.csv","current_data.csv","combined_data.csv"]:
        if not os.path.exists(os.path.join(data_dir, p)):
            import subprocess
            subprocess.run([sys.executable, os.path.join(data_dir,"generate_data.py")],
                           check=True, cwd=ROOT)
            break
    baseline = pd.read_csv(os.path.join(data_dir,"baseline_data.csv"), parse_dates=["Date"])
    current  = pd.read_csv(os.path.join(data_dir,"current_data.csv"),  parse_dates=["Date"])
    combined = pd.read_csv(os.path.join(data_dir,"combined_data.csv"), parse_dates=["Date"])
    return baseline, current, combined

def show_upload_guide():
    """Show a helper card — now just says any CSV works."""
    st.markdown("""
    <div style="background:#0d1a2e;border:1px solid #1e3a5f;border-radius:12px;
                padding:.9rem 1.1rem;margin:.4rem 0">
      <div style="font-size:.7rem;color:#3A86FF;font-weight:700;letter-spacing:.1em;
                  text-transform:uppercase;margin-bottom:.5rem">✨ Any CSV Works</div>
      <div style="font-size:.8rem;color:#94a3b8;line-height:1.8">
        Upload <b style="color:#e2e8f0">any CSV file</b> — columns are auto-detected.<br>
        The system finds your date, numeric, and category columns automatically.<br>
        <span style="color:#4a5a80;font-size:.75rem">
          Works best with: sales, transactions, orders, surveys, logs.
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header" style="display:flex;align-items:center;gap:14px;padding:0 0 1.2rem;border-bottom:1px solid #1e2235;margin-bottom:.5rem">
      <span style="font-size:2rem">📡</span>
      <div>
        <div style="font-size:1.75rem;font-weight:700;background:linear-gradient(135deg,#3A86FF,#9B5DE5);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Consumer Behavior Drift Detection</div>
        <div style="font-size:.85rem;color:#475569;margin-top:2px">Real-time statistical monitoring · 12 advanced features</div>
      </div>
    </div>
    <div class="about-hero">
      <div class="about-tag">⬡ About This Model</div>
      <div class="about-desc">
        This model detects changes in <b>consumer purchasing behavior</b> using statistical drift tests.
        <b>KS tests</b> identify shifts in feature distributions. <b>Time-series analysis</b> highlights
        evolving customer interests. <b>AI-powered reports</b> explain what drifted and why.
        <b>Alerts notify businesses</b> of significant behavioral changes so they can
        <b>adapt marketing strategies proactively</b>.
      </div>
      <div class="about-divider"></div>
      <div class="about-pills">
        <div class="about-pill"><span class="pill-dot" style="background:#3A86FF"></span>KS + Chi-Square Tests</div>
        <div class="about-pill"><span class="pill-dot" style="background:#9B5DE5"></span>PSI Severity Score</div>
        <div class="about-pill"><span class="pill-dot" style="background:#FFB703"></span>Wasserstein Distance</div>
        <div class="about-pill"><span class="pill-dot" style="background:#06D6A0"></span>Rolling 30-day Window</div>
        <div class="about-pill"><span class="pill-dot" style="background:#00BBF9"></span>PDF Export · CSV Upload</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        app_mode = st.radio("App Mode", ["📡 Drift Dashboard", "🔮 Manual Prediction"], label_visibility="collapsed")
        st.markdown("---")
        
        st.markdown("### ⚙️ Configuration")

        # -- Upload CSVs --
        st.markdown("**📂 Upload Your Data**")
        st.caption("Upload two separate CSV files — one for each period to compare.")

        upload_mode = st.radio(
            "Data source",
            ["Use sample data", "Upload two CSVs", "Bulk Scanner"],
            horizontal=False,
            help="Sample data · Two-file compare · Bulk scan multiple files at once",
        )

        uploaded_base = None
        uploaded_curr = None
        if upload_mode == "Upload two CSVs":
            st.caption("**Baseline CSV** — the reference/old period")
            uploaded_base = st.file_uploader(
                "Baseline period CSV",
                type=["csv"], key="base_upload",
                help="Required columns: Date, Purchase_Amount, Product_Category, Payment_Method",
            )
            st.caption("**Current CSV** — the new period to compare against baseline")
            uploaded_curr = st.file_uploader(
                "Current period CSV",
                type=["csv"], key="curr_upload",
                help="Same columns as baseline CSV",
            )
            if uploaded_base and uploaded_curr:
                st.success("✅ Both files uploaded")
            elif uploaded_base or uploaded_curr:
                st.warning("⚠️ Upload both files to enable comparison")
            show_upload_guide()

            # ── Sample CSV downloads ──────────────────────────────────────
            st.markdown("**📥 Download Sample CSVs**")
            st.caption("Not sure what format to use? Download these sample files to see the expected structure.")
            import os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))

            _sc1, _sc2 = st.columns(2)
            with _sc1:
                _sbase_path = _os.path.join(_here, "sample_baseline.csv")
                if _os.path.exists(_sbase_path):
                    with open(_sbase_path) as _f:
                        st.download_button(
                            "⬇️ sample_baseline.csv",
                            data=_f.read(),
                            file_name="sample_baseline.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="30-row baseline sample — Jan 2024",
                        )
            with _sc2:
                _scurr_path = _os.path.join(_here, "sample_current.csv")
                if _os.path.exists(_scurr_path):
                    with open(_scurr_path) as _f:
                        st.download_button(
                            "⬇️ sample_current.csv",
                            data=_f.read(),
                            file_name="sample_current.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="30-row current sample with drift — Jul 2024",
                        )
            st.markdown("""
            <div style="background:#0d1a2e;border:1px solid #1e3a5f;border-radius:10px;
                        padding:.7rem 1rem;margin:.4rem 0;font-size:.78rem;color:#94a3b8">
              <b style="color:#3A86FF">Required columns:</b>
              <span style="font-family:monospace;color:#FFB703"> Date</span> ·
              <span style="font-family:monospace;color:#FFB703"> Purchase_Amount</span> ·
              <span style="font-family:monospace;color:#FFB703"> Product_Category</span> ·
              <span style="font-family:monospace;color:#FFB703"> Payment_Method</span><br>
              Any column names work — the system auto-detects them.
              Extra columns are ignored.
            </div>
            """, unsafe_allow_html=True)

        # ── Bulk Scanner uploader ────────────────────────────────────────────
        bulk_files = []
        if upload_mode == "Bulk Scanner":
            st.caption("Upload **3 or more** CSV files. Auto-detection runs on each.")
            bulk_uploads = st.file_uploader(
                "Upload multiple CSV files",
                type=["csv"],
                accept_multiple_files=True,
                key="bulk_upload",
                help="Upload any CSV files — columns are auto-detected",
            )
            if bulk_uploads:
                st.success(f"✅ {len(bulk_uploads)} file{'s' if len(bulk_uploads)>1 else ''} uploaded")
                bulk_files = bulk_uploads
            if len(bulk_uploads) < 2 if bulk_uploads else True:
                st.info("Upload at least 2 CSV files to start scanning")

            bulk_mode = st.selectbox(
                "Comparison mode",
                ["Consecutive (1→2, 2→3, 3→4…)",
                 "First vs All (file 1 as fixed baseline)",
                 "All Pairs (every combination)"],
                help="How to pair files for comparison",
            )

        st.markdown("---")
        st.markdown("**📅 Date Range Filter**")
        use_date_filter = st.checkbox(
            "Filter by date range",
            value=False,
            help="Only available when using sample data",
            disabled=(upload_mode in ["Upload two CSVs", "Bulk Scanner"]),
        )

        st.markdown("---")
        st.markdown("**🎯 Detection Threshold**")
        alpha = st.slider(
            "p-value threshold (α)",
            min_value=0.01, max_value=0.10,
            value=0.05, step=0.01,
            help="Lower = stricter. Drift detected when p-value < α",
        )
        st.caption(f"Current threshold: **α = {alpha:.2f}**")

        st.markdown("---")

        # ── Theme toggle ──────────────────────────────────────────────────────
        st.markdown("**🎨 Theme**")
        # Use session_state to remember theme
        if "theme" not in st.session_state:
            st.session_state["theme"] = "dark"
        theme_label = "☀️ Switch to Light Mode" if st.session_state["theme"] == "dark" else "🌙 Switch to Dark Mode"
        if st.button(theme_label, use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()



        st.markdown("---")
        st.caption("🔄 Refresh page to reset")
        st.caption("Built with Streamlit · scipy · Plotly")

    # ── Load / prepare data ───────────────────────────────────────────────────
    baseline_df, current_df, combined_df = load_default_data()
    using_upload = False

    if upload_mode == "Upload two CSVs" and uploaded_base and uploaded_curr:
        try:
            base_raw = pd.read_csv(uploaded_base)
            curr_raw = pd.read_csv(uploaded_curr)

            # Auto-detect columns — no required format
            baseline_df, mapping_b, report_b = smart_load(base_raw, "Baseline")
            current_df,  mapping_c, report_c = smart_load(curr_raw,  "Current")
            combined_df = pd.concat([baseline_df, current_df], ignore_index=True)
            using_upload = True

            # Show what was detected
            st.markdown(
                mapping_summary_html(mapping_b, mapping_c, base_raw, curr_raw),
                unsafe_allow_html=True,
            )

            # Show data preview
            with st.expander("📋 Preview mapped data", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"**Baseline** — {len(baseline_df):,} rows  |  "
                               f"{baseline_df['Date'].min().date()} → {baseline_df['Date'].max().date()}")
                    st.dataframe(baseline_df[["Date","Purchase_Amount",
                                              "Product_Category","Payment_Method"]].head(8),
                                 use_container_width=True)
                with c2:
                    st.caption(f"**Current** — {len(current_df):,} rows  |  "
                               f"{current_df['Date'].min().date()} → {current_df['Date'].max().date()}")
                    st.dataframe(current_df[["Date","Purchase_Amount",
                                             "Product_Category","Payment_Method"]].head(8),
                                 use_container_width=True)

            if len(baseline_df) < 10 or len(current_df) < 10:
                st.warning("⚠️ Very few rows detected — results may not be meaningful. Check the mapping above.")

        except Exception as e:
            st.error(f"Could not process CSV: {e}")
            st.info("Try checking: does the file have column headers? Is it a valid CSV?")

    # ── Bulk Scanner mode ────────────────────────────────────────────────────
    if upload_mode == "Bulk Scanner" and len(bulk_files) >= 2:
        st.markdown('<div class="section-heading"><span class="sh-icon">🔎</span> Bulk Scanner Results</div>',
                    unsafe_allow_html=True)
        st.caption(f"Scanning {len(bulk_files)} files — auto-detecting columns in each.")

        mode_map = {
            "Consecutive (1→2, 2→3, 3→4…)": "consecutive",
            "First vs All (file 1 as fixed baseline)": "first_vs_all",
            "All Pairs (every combination)": "all_pairs",
        }
        scan_mode = mode_map.get(bulk_mode, "consecutive")

        with st.spinner(f"Running bulk scan across {len(bulk_files)} files…"):
            # Auto-map each uploaded file
            file_map = {}
            failed   = []
            for uf in bulk_files:
                try:
                    raw = pd.read_csv(uf)
                    mapped, _, _ = smart_load(raw, uf.name)
                    file_map[uf.name] = mapped
                except Exception as ex:
                    failed.append(f"{uf.name}: {ex}")

            if failed:
                for msg in failed:
                    st.warning(f"⚠️ Skipped — {msg}")

            if len(file_map) >= 2:
                scan_df = run_bulk_scan(file_map, alpha=alpha, mode=scan_mode)
            else:
                scan_df = pd.DataFrame()

        if scan_df.empty:
            st.error("Could not produce any comparisons. Check that files have valid data.")
        else:
            # Summary KPIs
            n_drift    = int(scan_df["Drift"].sum())
            n_total    = len(scan_df)
            n_critical = int((scan_df["Severity"] == "🔴 Critical").sum())
            n_moderate = int((scan_df["Severity"] == "🟡 Moderate").sum())

            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric("Total Pairs Scanned", n_total)
            bk2.metric("Pairs with Drift",    n_drift)
            bk3.metric("Critical",            n_critical)
            bk4.metric("Moderate",            n_moderate)

            st.markdown("<br>", unsafe_allow_html=True)

            # Colour-coded table
            def _colour_row(row):
                if row["Severity"] == "🔴 Critical":
                    return ["background:rgba(239,71,111,0.10)"] * len(row)
                if row["Severity"] == "🟡 Moderate":
                    return ["background:rgba(255,183,3,0.08)"] * len(row)
                return [""] * len(row)

            display_cols = [
                "Baseline File","Current File","Severity",
                "KS Stat","PSI","KS p-value",
                "Base Mean $","Curr Mean $","Mean Change %",
                "Cat Chi2 p","Pay Chi2 p",
            ]
            display_df = scan_df[[c for c in display_cols if c in scan_df.columns]]

            st.dataframe(
                display_df.style.apply(_colour_row, axis=1),
                use_container_width=True,
                height=min(400, 55 + len(display_df) * 38),
            )

            # Worst pair details
            worst = scan_df.iloc[0]
            if worst["Drift"]:
                st.markdown(f"""
                <div style="background:#1a0a10;border:1px solid #EF476F44;border-radius:12px;
                            padding:.9rem 1.2rem;margin:.8rem 0">
                  <div style="font-size:.7rem;color:#EF476F;font-weight:700;letter-spacing:.1em;
                              text-transform:uppercase;margin-bottom:.4rem">Most Drifted Pair</div>
                  <div style="font-size:.95rem;color:#c8d4f0;font-weight:600">
                    {worst['Baseline File']}  →  {worst['Current File']}
                  </div>
                  <div style="font-size:.82rem;color:#94a3b8;margin-top:.3rem">
                    KS stat: {worst.get('KS Stat','—')}  ·
                    PSI: {worst.get('PSI','—')}  ·
                    Mean change: {worst.get('Mean Change %','—')}%
                  </div>
                </div>""", unsafe_allow_html=True)

            # Bulk scan CSV download
            st.download_button(
                "⬇️ Download Bulk Scan Results CSV",
                data=scan_df.drop(columns=["_sort_key"], errors="ignore").to_csv(index=False),
                file_name=f"bulk_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Column mapping summary per file
            with st.expander("🔍 Column mapping per file", expanded=False):
                for fname, fdf in file_map.items():
                    st.caption(f"**{fname}** — {len(fdf):,} rows")
                    cols_found = {
                        c: "✅" for c in
                        ["Date","Purchase_Amount","Product_Category","Payment_Method"]
                        if c in fdf.columns
                    }
                    st.write(cols_found)

        st.divider()

    # Date range filter — only for sample data
    if use_date_filter and not using_upload and not combined_df.empty:
        col1, col2 = st.columns(2)
        min_d = combined_df["Date"].min().date()
        max_d = combined_df["Date"].max().date()
        mid_d = combined_df["Date"].median().date()
        with col1:
            base_end = st.date_input("Baseline ends", value=mid_d,
                                      min_value=min_d, max_value=max_d)
        with col2:
            curr_start = st.date_input("Current starts", value=mid_d,
                                        min_value=min_d, max_value=max_d)
        baseline_df = combined_df[combined_df["Date"].dt.date <= base_end].copy()
        current_df  = combined_df[combined_df["Date"].dt.date >= curr_start].copy()
        if baseline_df.empty or current_df.empty:
            st.warning("Date filter produced empty dataset — using defaults.")
            baseline_df, current_df, combined_df = load_default_data()

    # ── Show dataset summary in sidebar after data is loaded ─────────────────
    with st.sidebar:
        st.markdown("---")
        src_label = "📤 Uploaded" if using_upload else "📦 Sample Data"
        st.markdown(f"**{src_label}**")
        c1, c2 = st.columns(2)
        c1.metric("Baseline", f"{len(baseline_df):,}")
        c2.metric("Current",  f"{len(current_df):,}")
        st.caption(f"Baseline: {baseline_df['Date'].min().date()} → {baseline_df['Date'].max().date()}")
        st.caption(f"Current : {current_df['Date'].min().date()} → {current_df['Date'].max().date()}")

    if app_mode == "🔮 Manual Prediction":
        st.markdown('<div class="section-heading"><span class="sh-icon">🔮</span> Manual Prediction Module</div>',
                    unsafe_allow_html=True)
        st.caption("Enter a single transaction's details to predict its Purchase Amount based on the baseline data patterns.")

        with st.form("manual_prediction_form"):
            pred_col1, pred_col2, pred_col3 = st.columns([2, 2, 1])
            with pred_col1:
                all_cat_options = list(baseline_df["Product_Category"].dropna().unique())
                user_cat = st.selectbox("Product Category", options=all_cat_options, key="pred_cat")
            with pred_col2:
                all_pay_options = list(baseline_df["Payment_Method"].dropna().unique())
                user_pay = st.selectbox("Payment Method", options=all_pay_options, key="pred_pay")
            with pred_col3:
                st.markdown("<br>", unsafe_allow_html=True) # align with dropdowns
                submit_pred = st.form_submit_button("Predict Amount", use_container_width=True)

        if submit_pred:
            if baseline_df.empty:
                st.warning("Baseline data is empty. Cannot train prediction model.")
            else:
                with st.spinner("Training Random Forest & building explanation…"):
                    try:
                        from prediction_engine import train_and_predict
                        predicted_amt, expected_std, explain = train_and_predict(
                            baseline_df, user_cat, user_pay
                        )
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        explain = None

                if explain:
                    # ── Headline prediction card ──────────────────────────
                    _rank    = explain["rank"]
                    _total   = explain["total_combos"]
                    _rank_txt = f"#{_rank} most expensive of {_total} combos"
                    _vs_avg  = explain["predicted_amt"] - explain["global_mean"]
                    _vs_txt  = f"{_vs_avg:+.2f} vs baseline average"

                    st.markdown(f"""
                    <div style="background:{_MET_BG};border:1px solid {_BOR2};border-radius:16px;
                                padding:1.4rem 1.6rem;margin:.8rem 0">
                      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;align-items:center">
                        <div style="text-align:center">
                          <div style="font-size:.72rem;color:{_DIM};text-transform:uppercase;
                               letter-spacing:.1em;font-weight:600;margin-bottom:.3rem">
                            Predicted Amount
                          </div>
                          <div style="font-size:2.4rem;font-family:'IBM Plex Mono',monospace;
                               color:#3A86FF;font-weight:700">${explain['predicted_amt']:.2f}</div>
                          <div style="font-size:.78rem;color:{_MID};margin-top:.2rem">{_vs_txt}</div>
                        </div>
                        <div style="text-align:center;border-left:1px solid {_BORDER};
                                    border-right:1px solid {_BORDER};padding:0 1rem">
                          <div style="font-size:.72rem;color:{_DIM};text-transform:uppercase;
                               letter-spacing:.1em;font-weight:600;margin-bottom:.3rem">
                            90% Confidence Range
                          </div>
                          <div style="font-size:1.3rem;font-family:'IBM Plex Mono',monospace;
                               color:#FFB703;font-weight:600">
                            ${explain['ci_low']:.2f} – ${explain['ci_high']:.2f}
                          </div>
                          <div style="font-size:.78rem;color:{_MID};margin-top:.2rem">
                            Based on {explain['n_trees']} decision trees
                          </div>
                        </div>
                        <div style="text-align:center">
                          <div style="font-size:.72rem;color:{_DIM};text-transform:uppercase;
                               letter-spacing:.1em;font-weight:600;margin-bottom:.3rem">
                            Combo Rank
                          </div>
                          <div style="font-size:1.3rem;font-family:'IBM Plex Mono',monospace;
                               color:#9B5DE5;font-weight:600">#{_rank} / {_total}</div>
                          <div style="font-size:.78rem;color:{_MID};margin-top:.2rem">
                            {_rank_txt}
                          </div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-heading"><span class="sh-icon">🔍</span> Why this prediction?</div>',
                                unsafe_allow_html=True)

                    # ── Explanation tabs ──────────────────────────────────
                    ex_tab1, ex_tab2, ex_tab3, ex_tab4 = st.tabs([
                        "📊 Feature Impact", "🛒 Category Breakdown",
                        "💳 Payment Breakdown", "🏆 All Combos"
                    ])

                    with ex_tab1:
                        st.caption("How much each input influenced the prediction.")
                        # Feature importance bars
                        cat_imp = explain["cat_importance"]
                        pay_imp = explain["pay_importance"]
                        st.markdown(f"""
                        <div style="margin:.8rem 0">
                          <div style="font-size:.78rem;color:{_MID};margin-bottom:.3rem">
                            Product Category &nbsp;<b style="color:{_TEXT}">{user_cat}</b>
                          </div>
                          <div style="background:{_BG};border-radius:6px;height:18px;overflow:hidden;margin-bottom:.8rem">
                            <div style="background:#3A86FF;width:{min(cat_imp,100)}%;height:100%;border-radius:6px;
                                 display:flex;align-items:center;padding-left:8px;
                                 font-size:.72rem;font-weight:700;color:#fff">{cat_imp:.1f}%</div>
                          </div>
                          <div style="font-size:.78rem;color:{_MID};margin-bottom:.3rem">
                            Payment Method &nbsp;<b style="color:{_TEXT}">{user_pay}</b>
                          </div>
                          <div style="background:{_BG};border-radius:6px;height:18px;overflow:hidden">
                            <div style="background:#9B5DE5;width:{min(pay_imp,100)}%;height:100%;border-radius:6px;
                                 display:flex;align-items:center;padding-left:8px;
                                 font-size:.72rem;font-weight:700;color:#fff">{pay_imp:.1f}%</div>
                          </div>
                        </div>
                        <div style="font-size:.76rem;color:{_DIM};margin-top:.8rem;padding:.6rem .8rem;
                             background:{_BG2};border-radius:8px">
                          <b style="color:{_TEXT}">How to read this:</b> The Random Forest model learned from
                          {explain['n_training_rows']:,} baseline transactions.
                          Product Category drove <b style="color:#3A86FF">{cat_imp:.1f}%</b> of the
                          prediction and Payment Method drove <b style="color:#9B5DE5">{pay_imp:.1f}%</b>.
                          Higher % = that feature had more influence on the predicted amount.
                        </div>
                        """, unsafe_allow_html=True)

                        # Actual vs predicted for this combo
                        if explain["combo_actual_mean"] is not None:
                            _act = explain["combo_actual_mean"]
                            _diff = explain["predicted_amt"] - _act
                            st.metric(
                                f"Actual avg for {user_cat} + {user_pay} in baseline",
                                f"${_act:.2f}",
                                delta=f"Prediction off by ${_diff:+.2f}",
                                delta_color="off",
                                help=f"Based on {explain['combo_n']} real transactions in baseline data"
                            )

                    with ex_tab2:
                        st.caption(f"Average spend per category in baseline data. Your selected category: **{user_cat}**")
                        cat_df = explain["cat_avgs"].reset_index()
                        cat_df.columns = ["Category", "Avg Spend ($)", "Transactions", "Std Dev ($)"]
                        cat_df["Avg Spend ($)"] = cat_df["Avg Spend ($)"].apply(lambda x: f"${x:.2f}")
                        cat_df["Std Dev ($)"]   = cat_df["Std Dev ($)"].apply(lambda x: f"±${x:.2f}")
                        # Highlight selected row
                        def _highlight(row):
                            if row["Category"] == user_cat:
                                return [f"background:rgba(58,134,255,0.15)"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            cat_df.style.apply(_highlight, axis=1),
                            use_container_width=True, hide_index=True
                        )
                        st.caption(f"**{user_cat}** transactions = {explain['combo_n']} in baseline · "
                                   f"Global baseline mean = ${explain['global_mean']:.2f}")

                    with ex_tab3:
                        st.caption(f"Average spend per payment method in baseline data. Your selected method: **{user_pay}**")
                        pay_df = explain["pay_avgs"].reset_index()
                        pay_df.columns = ["Payment Method", "Avg Spend ($)", "Transactions", "Std Dev ($)"]
                        pay_df["Avg Spend ($)"] = pay_df["Avg Spend ($)"].apply(lambda x: f"${x:.2f}")
                        pay_df["Std Dev ($)"]   = pay_df["Std Dev ($)"].apply(lambda x: f"±${x:.2f}")
                        def _highlight2(row):
                            if row["Payment Method"] == user_pay:
                                return [f"background:rgba(155,93,229,0.15)"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            pay_df.style.apply(_highlight2, axis=1),
                            use_container_width=True, hide_index=True
                        )

                    with ex_tab4:
                        st.caption("Predicted amounts for every category + payment combination, ranked highest to lowest.")
                        combos_df = explain["all_combos"].copy()
                        combos_df["Predicted_Amount"] = combos_df["Predicted_Amount"].apply(
                            lambda x: f"${x:.2f}"
                        )
                        combos_df["Rank"] = range(1, len(combos_df)+1)
                        combos_df = combos_df[["Rank","Product_Category","Payment_Method","Predicted_Amount"]]
                        def _highlight3(row):
                            if (row["Product_Category"] == user_cat and
                                row["Payment_Method"]   == user_pay):
                                return ["background:rgba(58,134,255,0.2)"] * len(row)
                            return [""] * len(row)
                        st.dataframe(
                            combos_df.style.apply(_highlight3, axis=1),
                            use_container_width=True, hide_index=True
                        )
                        st.caption(f"Your selection ({user_cat} + {user_pay}) is highlighted in blue — ranked #{_rank} of {_total}.")

                elif explain is None and not baseline_df.empty:
                    st.error("Not enough baseline data to generate a prediction.")
        st.stop()

    # ── Run drift detection — statistical + ML ───────────────────────────────
    with st.spinner("Running statistical tests…"):
        results = run_all_drift_checks(baseline_df, current_df, alpha=alpha)
    with st.spinner("Training ML models…"):
        ml_results = run_ml_drift_checks(baseline_df, current_df)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Drift Detection Results
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">🔬</span> Drift Detection Results</div>',
                unsafe_allow_html=True)

    feature_map = {
        "purchase_amount":  ("💵","Purchase Amount",  "KS + RF AUC"),
        "product_category": ("🛒","Product Category", "Chi-Sq + RF AUC"),
        "payment_method":   ("💳","Payment Method",   "Chi-Sq + RF AUC"),
    }
    key_map = {
        "purchase_amount":  "Purchase_Amount",
        "product_category": "Product_Category",
        "payment_method":   "Payment_Method",
    }
    cols = st.columns(3)
    for col, (key, (icon, label, test)) in zip(cols, feature_map.items()):
        r = results[key]
        drift = r["drift_detected"]
        card_cls  = "drifted" if drift else "stable"
        badge_cls = "drift-yes" if drift else "drift-no"
        badge_txt = "⚠ Drift Detected" if drift else "✓ Stable"
        pval_color = "#ff7096" if drift else "#06D6A0"
        # Pre-compute ML values outside the f-string to avoid {{}} hashability bug
        _feat_key  = key_map.get(key, "")
        _pf_empty  = {"auc": 0, "drift_score": 0}
        _pf        = ml_results["per_feature"].get(_feat_key, _pf_empty)
        _ml_auc    = _pf.get("auc", 0)
        _ml_score  = _pf.get("drift_score", 0)
        with col:
            st.markdown(f"""
            <div class="drift-card {card_cls}">
              <div class="dc-icon-row">
                <span class="dc-icon">{icon}</span>
                <span class="dc-test-pill">{test}</span>
              </div>
              <div class="dc-label">{label}</div>
              <div class="dc-test-name">{test}</div>
              <span class="drift-badge {badge_cls}">{badge_txt}</span>
              <div class="dc-stats">
                p-value &nbsp;&nbsp;: <span class="hi" style="color:{pval_color}">{r['p_value']:.6f}</span><br>
                statistic: <span>{r['statistic']:.4f}</span><br>
                ML AUC&nbsp;&nbsp;&nbsp;: <span style="color:#7eb3ff">{_ml_auc:.4f}</span><br>
                ML Score&nbsp;&nbsp;: <span style="color:#7eb3ff">{_ml_score:.4f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PSI + Wasserstein KPI row
    psi = results.get("psi", {})
    wss = results.get("wasserstein", {})
    psi_val   = psi.get("psi", 0)
    psi_level = psi.get("level", "—")
    psi_color = psi.get("color", "#64748b")
    wss_dist  = wss.get("distance", 0)
    wss_pct   = wss.get("pct_shift", 0)
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi-strip-card">
        <div class="ks-label">PSI Score</div>
        <div class="ks-value" style="color:{psi_color}">{psi_val:.4f}</div>
        <div class="ks-sub">Population Stability Index</div>
      </div>
      <div class="kpi-strip-card">
        <div class="ks-label">PSI Level</div>
        <div class="ks-value" style="font-size:1.05rem;color:{psi_color}">{psi_level}</div>
        <div class="ks-sub">&lt;0.10 Stable · 0.10–0.25 Moderate · &gt;0.25 Significant</div>
      </div>
      <div class="kpi-strip-card">
        <div class="ks-label">Wasserstein Distance</div>
        <div class="ks-value" style="color:#FFB703">${wss_dist:.2f}</div>
        <div class="ks-sub">Distribution shift in dollar units</div>
      </div>
      <div class="kpi-strip-card">
        <div class="ks-label">Distribution Shift</div>
        <div class="ks-value" style="color:#FFB703">{wss_pct:.1f}%</div>
        <div class="ks-sub">Relative to baseline mean purchase</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span class="sh-icon">📋</span> Drift Summary Table</div>',
                unsafe_allow_html=True)
    show_chart(drift_summary_heatmap(results))

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — ML Drift Analysis
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">🤖</span> ML Drift Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Three ML models independently detect drift. Score 0 = identical, 1 = fully different.")

    ml_kpi1, ml_kpi2, ml_kpi3, ml_kpi4 = st.columns(4)
    ml_kpi1.metric("Ensemble Score",  f"{ml_results['ensemble_score']:.4f}")
    ml_kpi2.metric("Drift Verdict",   "⚠ Drift" if ml_results["ensemble_drift"] else "✓ Stable")
    ml_kpi3.metric("Severity",        ml_results["ensemble_level"])
    ml_kpi4.metric("Best RF AUC",     f"{ml_results['random_forest']['auc']:.4f}",
                   help="AUC >0.65 means models can separate baseline from current = drift detected")

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🎯 Model Comparison", "🔍 Feature Importance",
        "🚨 Anomaly Scores",   "⚡ Drift Gauge"
    ])
    with ml_tab1:
        st.caption("All three ML models independently score drift. Ensemble bar = majority vote result.")
        show_chart(ml_model_comparison_chart(ml_results))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Isolation Forest",  f"{ml_results['isolation_forest']['drift_score']:.4f}",
                   help=f"Anomaly rate: {ml_results['isolation_forest']['anomaly_rate']:.2%}")
        mc2.metric("Random Forest AUC", f"{ml_results['random_forest']['auc']:.4f}")
        mc3.metric("Gradient Boosting", f"{ml_results['gradient_boosting']['auc']:.4f}")

    with ml_tab2:
        st.caption("Per-feature RF classifier AUC. Higher = that feature drifted more between periods.")
        show_chart(ml_feature_importance_chart(ml_results))
        pf = ml_results["per_feature"]
        fi1, fi2, fi3 = st.columns(3)
        for col_w, fname in zip([fi1,fi2,fi3],
                                ["Purchase_Amount","Product_Category","Payment_Method"]):
            fd = pf.get(fname, {})
            col_w.metric(fname.replace("_"," "),
                         f"AUC {fd.get('auc',0):.4f}",
                         delta=fd.get("level",""))

    with ml_tab3:
        st.caption("Isolation Forest trains on baseline. Current rows scored as anomalous = most different from baseline pattern.")
        show_chart(ml_anomaly_score_chart(ml_results, baseline_df, current_df))
        iso = ml_results["isolation_forest"]
        ai1, ai2, ai3 = st.columns(3)
        ai1.metric("Anomalies Detected", f"{iso['n_anomalies']:,}")
        ai2.metric("Anomaly Rate",        f"{iso['anomaly_rate']:.2%}")
        ai3.metric("IF Drift Score",      f"{iso['drift_score']:.4f}")

    with ml_tab4:
        st.caption("Ensemble ML risk gauge. Above 70 = action recommended.")
        show_chart(ml_drift_gauge(ml_results))

    st.divider()

    # Rolling window
    st.markdown('<div class="section-heading"><span class="sh-icon">📈</span> Rolling Drift Window</div>',
                unsafe_allow_html=True)
    st.caption("30-day sliding window across the combined timeline — shows when drift started.")
    with st.spinner("Computing rolling windows…"):
        rolling_df = rolling_drift_window(combined_df, window_days=30, step_days=7)
    show_chart(rolling_window_chart(rolling_df))
    if not rolling_df.empty:
        n_drift_w = rolling_df["drift"].sum()
        st.caption(f"**{n_drift_w}** of **{len(rolling_df)}** windows show drift (p < α={alpha:.2f})")

    st.divider()

        # SECTION 3 — Distribution Comparisons
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">📈</span> Distribution Comparisons</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💵  Purchase Amount", "🛒  Product Category",
        "💳  Payment Method",  "🎻  Violin + Box",
        "📉  Waterfall"
    ])

    with tab1:
        st.caption("KDE and histogram compare how purchase amounts are spread across both periods.")
        show_chart(purchase_amount_distribution(baseline_df, current_df))
        st.markdown("**Summary Statistics**")
        st.caption("💡 Hover the Statistic column to see full name and meaning.")
        show_chart(summary_stats_table(results["summary_stats"]))

    with tab2:
        st.caption("Grouped bars reveal shifts in category popularity between baseline and current.")
        show_chart(category_distribution(baseline_df, current_df))
        res_cat = results["product_category"]
        df_cat  = pd.DataFrame({
            "Baseline (%)": res_cat["baseline_distribution"],
            "Current (%)":  res_cat["current_distribution"],
        }).fillna(0)
        df_cat["Δ (pp)"] = (df_cat["Current (%)"] - df_cat["Baseline (%)"]).round(1)
        st.dataframe(df_cat.style.background_gradient(subset=["Δ (pp)"], cmap="RdYlGn"),
                     use_container_width=True)

    with tab3:
        st.caption("Tracks whether customers are switching between payment methods over time.")
        show_chart(payment_method_distribution(baseline_df, current_df))
        res_pay = results["payment_method"]
        df_pay  = pd.DataFrame({
            "Baseline (%)": res_pay["baseline_distribution"],
            "Current (%)":  res_pay["current_distribution"],
        }).fillna(0)
        df_pay["Δ (pp)"] = (df_pay["Current (%)"] - df_pay["Baseline (%)"]).round(1)
        st.dataframe(df_pay.style.background_gradient(subset=["Δ (pp)"], cmap="RdYlGn"),
                     use_container_width=True)

    with tab4:
        st.caption("Violin plot shows full distribution shape; box shows IQR, median, and outliers. Hover any point for exact value.")
        show_chart(violin_box_plot(baseline_df, current_df))

    with tab5:
        st.caption("Bars show which categories and payment methods gained (+) or lost (−) share. Sorted by magnitude of change.")
        wf_cat, wf_pay = waterfall_chart(results)
        st.markdown("**Category Share Change**")
        show_chart(wf_cat)
        st.markdown("**Payment Method Share Change**")
        show_chart(wf_pay)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3b — Drill-Down Click Explorer
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">🔍</span> Drill-Down Explorer</div>',
                unsafe_allow_html=True)
    st.caption("Select a category below to drill into its purchase amount distribution — see within-category drift.")

    all_cats = sorted(set(
        list(baseline_df["Product_Category"].dropna().unique()) +
        list(current_df["Product_Category"].dropna().unique())
    ))

    dd_col1, dd_col2 = st.columns([1, 3])
    with dd_col1:
        selected_cat = st.selectbox(
            "Select category",
            options=["— All categories —"] + all_cats,
            key="drilldown_cat",
        )
    with dd_col2:
        if selected_cat and selected_cat != "— All categories —":
            b_n = (baseline_df["Product_Category"] == selected_cat).sum()
            c_n = (current_df["Product_Category"]  == selected_cat).sum()
            st.caption(f"Baseline: **{b_n:,}** transactions  ·  Current: **{c_n:,}** transactions")
        else:
            st.caption("Choose a specific category to see its purchase amount distribution in both periods")

    cat_for_drill = None if (not selected_cat or selected_cat == "— All categories —") else selected_cat
    show_chart(drilldown_distribution(baseline_df, current_df, cat_for_drill))

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Relationship Analysis
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">🔗</span> Relationship Analysis</div>',
                unsafe_allow_html=True)

    rel_tab1, rel_tab2 = st.tabs(["🌊  Sankey Flow Diagram", "🔥  Correlation Heatmap"])

    with rel_tab1:
        st.caption("Visualizes how customers flow from categories to payment methods. Width of bands = volume of transactions.")
        show_chart(sankey_diagram(baseline_df, current_df))

    with rel_tab2:
        st.caption("Shows how correlations between features changed. Delta panel (right) highlights which relationships strengthened or weakened.")
        show_chart(correlation_heatmap(baseline_df, current_df))

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Time-Series Analysis
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">📅</span> Time-Series Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("**Monthly Average Purchase Amount**")
    st.caption("Hover any point for exact avg, min, max, and transaction count. Shaded band = min/max range.")
    show_chart(purchase_amount_trend(combined_df))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Category Share Over Time**")
    st.caption("Hover stacked bands for exact share % and count per category per month.")
    show_chart(category_frequency_trend(combined_df))

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Feature Importance Ranking
    # ───────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">🏆</span> Feature Importance Ranking</div>',
                unsafe_allow_html=True)
    st.caption("Which feature drifted the most? Ranked by a combined score of p-value strength, test statistic, and PSI contribution.")

    # Build feature data list
    feat_data = []
    ss = results.get("summary_stats", pd.DataFrame())
    base_mean = float(ss.loc["mean","Baseline"]) if "mean" in ss.index else 0
    curr_mean = float(ss.loc["mean","Current"])  if "mean" in ss.index else 0
    pct_change = ((curr_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0

    pa = results["purchase_amount"]
    pa_score = round((1 - pa["p_value"]) * 0.4 + min(pa["statistic"], 1.0) * 0.4 + min(psi.get("psi",0)/0.5,1.0) * 0.2, 4)
    feat_data.append({
        "icon": "💵", "name": "Purchase Amount", "test": "KS Test",
        "score": pa_score, "p_value": pa["p_value"], "statistic": pa["statistic"],
        "drift": pa["drift_detected"],
        "detail": f"Mean: ${base_mean:.2f} → ${curr_mean:.2f} ({pct_change:+.1f}%)",
        "bar_color": "#EF476F" if pa["drift_detected"] else "#06D6A0",
    })

    pc = results["product_category"]
    pc_score = round((1 - pc["p_value"]) * 0.5 + min(pc["statistic"]/300.0,1.0) * 0.5, 4)
    if "baseline_distribution" in pc:
        deltas = {k: abs(pc["current_distribution"].get(k,0)-pc["baseline_distribution"].get(k,0)) for k in pc["baseline_distribution"]}
        top_cat = max(deltas, key=deltas.get) if deltas else "N/A"
        cat_detail = f"Biggest shift: {top_cat} ({max(deltas.values()) if deltas else 0:+.1f} pp)"
    else:
        cat_detail = "Category distribution shifted"
    feat_data.append({
        "icon": "🛒", "name": "Product Category", "test": "Chi-Square",
        "score": pc_score, "p_value": pc["p_value"], "statistic": pc["statistic"],
        "drift": pc["drift_detected"], "detail": cat_detail,
        "bar_color": "#EF476F" if pc["drift_detected"] else "#06D6A0",
    })

    pm = results["payment_method"]
    pm_score = round((1 - pm["p_value"]) * 0.5 + min(pm["statistic"]/200.0,1.0) * 0.5, 4)
    if "baseline_distribution" in pm:
        deltas = {k: abs(pm["current_distribution"].get(k,0)-pm["baseline_distribution"].get(k,0)) for k in pm["baseline_distribution"]}
        top_pay = max(deltas, key=deltas.get) if deltas else "N/A"
        pay_detail = f"Biggest shift: {top_pay} ({max(deltas.values()) if deltas else 0:+.1f} pp)"
    else:
        pay_detail = "Payment mix shifted"
    feat_data.append({
        "icon": "💳", "name": "Payment Method", "test": "Chi-Square",
        "score": pm_score, "p_value": pm["p_value"], "statistic": pm["statistic"],
        "drift": pm["drift_detected"], "detail": pay_detail,
        "bar_color": "#EF476F" if pm["drift_detected"] else "#06D6A0",
    })

    feat_data.sort(key=lambda x: x["score"], reverse=True)
    rank_labels = ["🥇 #1 Highest Drift", "🥈 #2 Medium Drift", "🥉 #3 Lowest Drift"]
    rank_colors = ["#FFB703", "#94a3b8", "#CD7F32"]

    feat_cols = st.columns(3)
    for i, (col, feat) in enumerate(zip(feat_cols, feat_data)):
        bar_pct = int(feat["score"] * 100)
        badge_cls = "feat-drift" if feat["drift"] else "feat-stable"
        badge_txt = "⚠ Drift" if feat["drift"] else "✓ Stable"
        with col:
            st.markdown(
                f'''<div class="feat-card">
  <div class="feat-rank" style="color:{rank_colors[i]}">{rank_labels[i]}</div>
  <div class="feat-name">{feat["icon"]} {feat["name"]}</div>
  <span class="feat-badge {badge_cls}">{badge_txt}</span>
  <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{bar_pct}%;background:{feat["bar_color"]}"></div></div>
  <div class="feat-stats">
    Importance : <span>{feat["score"]:.4f}</span><br>
    Test&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span>{feat["test"]}</span><br>
    p-value&nbsp;&nbsp;&nbsp;: <span>{feat["p_value"]:.6f}</span><br>
    Statistic&nbsp;: <span>{feat["statistic"]:.4f}</span><br>
    <span style="color:#4a5a80;font-size:.7rem">{feat["detail"]}</span>
  </div>
</div>''', unsafe_allow_html=True)

    st.divider()

    # SECTION 6b — Multi-Feature Comparison Table
    # ───────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">📋</span> Multi-Feature Comparison Table</div>',
                unsafe_allow_html=True)
    st.caption("All features side-by-side sorted by drift importance score.")

    max_stat = max(f["statistic"] for f in feat_data) or 1
    rows_html = ""
    for i, feat in enumerate(feat_data):
        bar_w  = int((feat["statistic"] / max_stat) * 70)
        pv_col = "#ff7096" if feat["drift"] else "#06D6A0"
        verd_c = "comp-verdict-drift" if feat["drift"] else "comp-verdict-stable"
        verd_t = "⚠ Drift Detected" if feat["drift"] else "✓ Stable"
        rows_html += (
            f'<tr class="comp-row">'
            f'<td class="comp-cell" style="color:{rank_colors[i]};font-weight:700">#{i+1}</td>'
            f'<td class="comp-cell feat-n">{feat["icon"]} {feat["name"]}</td>'
            f'<td class="comp-cell num" style="color:#7eb3ff">{feat["test"]}</td>'
            f'<td class="comp-cell num" style="color:{pv_col}">{feat["p_value"]:.6f}</td>'
            f'<td class="comp-cell num">{feat["statistic"]:.4f}'
            f'<span class="comp-mini-bar" style="width:{bar_w}px;background:{feat["bar_color"]}"></span></td>'
            f'<td class="comp-cell num" style="color:#7eb3ff">{feat["score"]:.4f}</td>'
            f'<td class="comp-cell"><span class="{verd_c}">{verd_t}</span></td>'
            f'<td class="comp-cell" style="color:#4a5a80;font-size:.76rem">{feat["detail"]}</td>'
            f'</tr>'
        )

    st.markdown(
        f'''<div class="comp-wrap">
<table class="comp-table">
<thead><tr>
  <th class="comp-header" style="width:38px">Rank</th>
  <th class="comp-header">Feature</th>
  <th class="comp-header">Test</th>
  <th class="comp-header num">p-value</th>
  <th class="comp-header num">Statistic</th>
  <th class="comp-header num">Importance</th>
  <th class="comp-header">Verdict</th>
  <th class="comp-header">Key Change</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>
<div style="font-size:.72rem;color:#3a4a6a;margin-top:.5rem">
  Importance = p-value strength (40%) · test statistic magnitude (40%) · PSI contribution (20%)
</div>''', unsafe_allow_html=True)

    st.divider()

        # SECTION 7 — Custom Drift Rules
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">📏</span> Custom Drift Rules</div>',
                unsafe_allow_html=True)
    st.caption("Define your own alert rules — triggered when specific metrics cross your thresholds.")

    # Init rules list in session state
    if "custom_rules_list" not in st.session_state:
        st.session_state["custom_rules_list"] = []

    # Add new rule UI
    with st.expander("➕ Add a new rule", expanded=len(st.session_state["custom_rules_list"]) == 0):
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 2, 1.5, 1])
        with r_col1:
            r_name = st.text_input("Rule name", placeholder="e.g. High spend alert", key="r_name")
        with r_col2:
            r_type = st.selectbox("Condition type", RULE_TYPES, key="r_type")
        with r_col3:
            needs_target = any(k in r_type for k in ["Category", "Payment"])
            if needs_target:
                if "Category" in r_type:
                    r_target = st.selectbox("Category", get_categories(results), key="r_target")
                else:
                    r_target = st.selectbox("Payment method", get_payment_methods(results), key="r_target_pay")
            else:
                r_target = ""
                st.text_input("Target (n/a)", value="—", disabled=True, key="r_target_na")
        with r_col4:
            r_thresh = st.number_input("Threshold", value=20.0, step=1.0, key="r_thresh")

        if st.button("Add Rule", use_container_width=True):
            if r_name.strip():
                st.session_state["custom_rules_list"].append({
                    "name": r_name.strip(), "type": r_type,
                    "threshold": r_thresh, "target": r_target,
                })
                st.rerun()
            else:
                st.warning("Please enter a rule name.")

    # Evaluate and display existing rules
    if st.session_state["custom_rules_list"]:
        evaluated_rules = evaluate_rules(
            st.session_state["custom_rules_list"],
            results, baseline_df, current_df
        )

        triggered_count = sum(1 for r in evaluated_rules if r["triggered"])
        if triggered_count:
            st.error(f"⚠️ {triggered_count} rule{'s' if triggered_count > 1 else ''} triggered!")
        else:
            st.success("✅ All rules passing")

        for i, (rule_def, rule_res) in enumerate(
                zip(st.session_state["custom_rules_list"], evaluated_rules)):
            rc1, rc2, rc3, rc4, rc5 = st.columns([2, 3, 1.5, 1.2, 0.5])
            status_color = "#ff7096" if rule_res["triggered"] else "#06D6A0"
            status_txt   = "⚠ Triggered" if rule_res["triggered"] else "✓ Pass"
            with rc1:
                st.markdown(f"<span style='font-weight:600;color:#c8d4f0'>{rule_res['name']}</span>",
                            unsafe_allow_html=True)
            with rc2:
                st.caption(rule_res["condition"])
            with rc3:
                st.markdown(f"<span style='font-family:monospace;font-size:.85rem;color:#94a3b8'>"
                            f"Actual: {rule_res['actual']}</span>", unsafe_allow_html=True)
            with rc4:
                st.markdown(f"<span style='color:{status_color};font-weight:700'>{status_txt}</span>",
                            unsafe_allow_html=True)
            with rc5:
                if st.button("✕", key=f"del_rule_{i}", help="Remove this rule"):
                    st.session_state["custom_rules_list"].pop(i)
                    st.rerun()
    else:
        st.info("No rules defined yet. Add your first rule above.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
        # SECTION 7 — Export
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading"><span class="sh-icon">📤</span> Export</div>',
                unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**📄 Download PDF Report**")
        if not REPORTLAB_OK:
            st.warning("Install `reportlab` to enable PDF export: `pip install reportlab`")
            st.caption("Run: `pip install reportlab` then restart the app.")
        else:
            # Generate PDF immediately — single click download
            with st.spinner("Preparing PDF…"):
                pdf_bytes = generate_pdf(results, baseline_df, current_df)

            if pdf_bytes:
                from datetime import datetime
                fname = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    help="Click to download the full drift detection report as PDF",
                )
                st.caption("Includes: drift results · PSI · Wasserstein · summary stats")
            else:
                st.error("PDF generation failed — check reportlab installation.")

    with exp_col2:
        st.markdown("**🌐 Shareable HTML Report**")
        st.caption("Self-contained HTML file — open in any browser, no Python needed. Share with anyone.")
        if st.button("⚙️ Generate HTML Report", use_container_width=True, key="gen_html"):
            with st.spinner("Building shareable report…"):
                # Collect the key charts to embed
                _charts = {
                    "Purchase Amount Distribution": purchase_amount_distribution(baseline_df, current_df),
                    "Monthly Spend Trend":          purchase_amount_trend(combined_df),
                    "Category Distribution":        category_distribution(baseline_df, current_df),
                    "Payment Method Distribution":  payment_method_distribution(baseline_df, current_df),
                    "Category Share Over Time":     category_frequency_trend(combined_df),
                    "PSI Score":                    psi_chart(psi),
                }
                # Add waterfall
                _wf_cat, _wf_pay = waterfall_chart(results)
                _charts["Waterfall — Category"] = _wf_cat
                _charts["Waterfall — Payment"]  = _wf_pay

                _rules_for_report = evaluate_rules(
                    st.session_state.get("custom_rules_list", []),
                    results, baseline_df, current_df
                ) if st.session_state.get("custom_rules_list") else []

                html_str = generate_html_report(
                    results=results,
                    baseline_df=baseline_df,
                    current_df=current_df,
                    charts=_charts,
                    custom_rules=_rules_for_report,
                )

            from datetime import datetime as _dt
            html_fname = f"drift_report_{_dt.now().strftime('%Y%m%d_%H%M')}.html"
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_str.encode("utf-8"),
                file_name=html_fname,
                mime="text/html",
                use_container_width=True,
                help="Open this file in any browser — no Python required",
            )
            st.caption("📎 Share via email, Slack, Google Drive — anyone can open it")

    _exp_col3, _exp_col4 = st.columns(2)
    with _exp_col3:
        st.markdown("**📊 Download Results CSV**")
        summary_rows = []
        for key, (_, label, test) in feature_map.items():
            r = results[key]
            summary_rows.append({
                "Feature": label, "Test": test,
                "Statistic": r["statistic"], "p_value": r["p_value"],
                "Drift_Detected": r["drift_detected"], "Alpha": alpha,
                "PSI": psi.get("psi",""), "Wasserstein": wss.get("distance",""),
            })
        csv_df = pd.DataFrame(summary_rows)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_df.to_csv(index=False),
            file_name="drift_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with _exp_col4:
        st.markdown("**📥 Download Mapped Data**")
        st.caption("Download the auto-mapped baseline + current data as CSV.")
        st.download_button(
            "⬇️ Baseline CSV",
            data=baseline_df.to_csv(index=False),
            file_name="baseline_mapped.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Current CSV",
            data=current_df.to_csv(index=False),
            file_name="current_mapped.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # ── Raw data explorer ─────────────────────────────────────────────────────
    with st.expander("🗃  Explore Raw Data"):
        period = st.radio("Select period:", ["Baseline","Current","Combined"], horizontal=True)
        df_show = {"Baseline":baseline_df,"Current":current_df,"Combined":combined_df}[period]
        st.dataframe(df_show, use_container_width=True)
        st.caption(f"{len(df_show):,} rows · {len(df_show.columns)} columns")


if __name__ == "__main__":
    main()