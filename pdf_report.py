"""
pdf_report.py  —  PDF Report Generator using reportlab
Generates a branded, multi-page PDF with all drift results and charts.
"""

import io
import pandas as pd
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
    BRAND_BLUE   = colors.HexColor("#3A86FF")
    BRAND_AMBER  = colors.HexColor("#FFB703")
    DRIFT_RED    = colors.HexColor("#EF476F")
    STABLE_GREEN = colors.HexColor("#06D6A0")
    BG_DARK      = colors.HexColor("#0f1117")
    BG_CARD      = colors.HexColor("#1a1d2e")
    TEXT_MAIN    = colors.HexColor("#e2e8f0")
    TEXT_DIM     = colors.HexColor("#64748b")
    GRID         = colors.HexColor("#2a2d3e")
except ImportError:
    REPORTLAB_OK = False
    BRAND_BLUE = BRAND_AMBER = DRIFT_RED = STABLE_GREEN = None
    BG_DARK = BG_CARD = TEXT_MAIN = TEXT_DIM = GRID = None


def generate_pdf(results: dict, baseline_df: pd.DataFrame,
                 current_df: pd.DataFrame,
                 ai_report: str = "",
                 root_cause: str = "") -> bytes:
    """
    Generate the full PDF report.
    Returns raw bytes suitable for st.download_button.
    """
    if not REPORTLAB_OK:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Consumer Behavior Drift Detection Report",
    )

    styles = getSampleStyleSheet()

    def sty(name, **kw):
        s = ParagraphStyle(name, parent=styles["Normal"], **kw)
        return s

    H1    = sty("H1",    fontSize=22, textColor=BRAND_BLUE,    spaceAfter=6,  fontName="Helvetica-Bold")
    H2    = sty("H2",    fontSize=14, textColor=TEXT_MAIN,      spaceAfter=4,  fontName="Helvetica-Bold",  spaceBefore=14)
    H3    = sty("H3",    fontSize=11, textColor=BRAND_BLUE,     spaceAfter=3,  fontName="Helvetica-Bold")
    BODY  = sty("BODY",  fontSize=9,  textColor=TEXT_DIM,       spaceAfter=4,  leading=14)
    SMALL = sty("SMALL", fontSize=8,  textColor=TEXT_DIM,       spaceAfter=2)
    META  = sty("META",  fontSize=8,  textColor=TEXT_DIM,       alignment=TA_RIGHT)

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("Consumer Behavior", H1))
    story.append(Paragraph("Drift Detection Report", H1))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_BLUE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}  |  "
        f"Baseline n={len(baseline_df):,}  |  Current n={len(current_df):,}",
        META
    ))
    story.append(Spacer(1, 1*cm))

    # ── Section 1: Drift Results ──────────────────────────────────────────────
    story.append(Paragraph("1. Drift Detection Results", H2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
    story.append(Spacer(1, 0.3*cm))

    feature_map = {
        "purchase_amount":  ("Purchase Amount",  "Kolmogorov-Smirnov"),
        "product_category": ("Product Category", "Chi-Square"),
        "payment_method":   ("Payment Method",   "Chi-Square"),
    }

    tdata = [["Feature", "Test", "Statistic", "p-value", "Result"]]
    for key, (label, test) in feature_map.items():
        if key in results:
            r = results[key]
            tdata.append([
                label, test,
                f"{r['statistic']:.4f}",
                f"{r['p_value']:.6f}",
                "DRIFT" if r["drift_detected"] else "STABLE",
            ])

    t = Table(tdata, colWidths=[4*cm, 4.5*cm, 3*cm, 3*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BG_CARD),
        ("TEXTCOLOR",     (0,0), (-1,0),  BRAND_BLUE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#111827"),
                                           colors.HexColor("#0f1117")]),
        ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_DIM),
        ("ALIGN",         (2,0), (-1,-1), "CENTER"),
        ("GRID",          (0,0), (-1,-1), 0.3, GRID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))

    # Color the result column
    for i, key in enumerate(feature_map, 1):
        if key in results:
            cell_color = DRIFT_RED if results[key]["drift_detected"] else STABLE_GREEN
            t.setStyle(TableStyle([("TEXTCOLOR", (4, i), (4, i), cell_color),
                                    ("FONTNAME",  (4, i), (4, i), "Helvetica-Bold")]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))

    # ── Section 2: Advanced Metrics ───────────────────────────────────────────
    story.append(Paragraph("2. Advanced Drift Metrics", H2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
    story.append(Spacer(1, 0.3*cm))

    psi = results.get("psi", {})
    wss = results.get("wasserstein", {})

    adv_data = [
        ["Metric", "Value", "Interpretation"],
        ["PSI Score",          f"{psi.get('psi', 'N/A'):.4f}" if psi else "N/A",
         psi.get("level", "N/A")],
        ["Wasserstein Distance", f"${wss.get('distance', 'N/A'):.2f}" if wss else "N/A",
         wss.get("interpretation", "N/A")],
    ]
    t2 = Table(adv_data, colWidths=[4.5*cm, 3*cm, 9.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BG_CARD),
        ("TEXTCOLOR",     (0,0), (-1,0),  BRAND_BLUE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#111827"),
                                           colors.HexColor("#0f1117")]),
        ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_DIM),
        ("GRID",          (0,0), (-1,-1), 0.3, GRID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.6*cm))

    # ── Section 3: Summary Statistics ────────────────────────────────────────
    story.append(Paragraph("3. Purchase Amount Statistics", H2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
    story.append(Spacer(1, 0.3*cm))

    ss = results.get("summary_stats", pd.DataFrame())
    if not ss.empty:
        stat_labels = {
            "count": "Count", "mean": "Mean", "std": "Std Dev",
            "min": "Minimum", "25%": "Q1 (25th pct)", "50%": "Median",
            "75%": "Q3 (75th pct)", "max": "Maximum",
        }
        ss_data = [["Statistic", "Baseline", "Current", "Change"]]
        for idx in ss.index:
            b = float(ss.loc[idx, "Baseline"])
            c = float(ss.loc[idx, "Current"])
            chg = c - b
            sign = "+" if chg >= 0 else ""
            ss_data.append([
                stat_labels.get(idx, idx),
                f"${b:,.2f}" if idx != "count" else f"{b:,.0f}",
                f"${c:,.2f}" if idx != "count" else f"{c:,.0f}",
                f"{sign}{chg:,.2f}",
            ])
        t3 = Table(ss_data, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 5.5*cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  BG_CARD),
            ("TEXTCOLOR",     (0,0), (-1,0),  BRAND_BLUE),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#111827"),
                                               colors.HexColor("#0f1117")]),
            ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_DIM),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("GRID",          (0,0), (-1,-1), 0.3, GRID),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t3)
    story.append(Spacer(1, 0.6*cm))

    # ── Section 4: AI Report ──────────────────────────────────────────────────
    if ai_report and not ai_report.startswith("["):
        story.append(PageBreak())
        story.append(Paragraph("4. AI-Generated Analysis", H2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
        story.append(Spacer(1, 0.3*cm))

        for line in ai_report.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2*cm))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], H3))
            elif line.startswith("- "):
                story.append(Paragraph(f"• {line[2:]}", BODY))
            else:
                story.append(Paragraph(line, BODY))
        story.append(Spacer(1, 0.5*cm))

    # ── Section 5: Root Cause ────────────────────────────────────────────────
    if root_cause and not root_cause.startswith("["):
        story.append(Paragraph("5. Root Cause Analysis", H2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
        story.append(Spacer(1, 0.3*cm))
        for line in root_cause.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2*cm))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], H3))
            elif line.startswith("- "):
                story.append(Paragraph(f"• {line[2:]}", BODY))
            else:
                story.append(Paragraph(line, BODY))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRID))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Generated by Consumer Behavior Drift Detection Dashboard  "
        f"|  {datetime.now().strftime('%Y-%m-%d')}  "
        "|  Powered by scipy + Claude AI",
        SMALL
    ))

    doc.build(story)
    return buf.getvalue()