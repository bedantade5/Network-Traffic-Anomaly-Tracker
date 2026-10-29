"""Streamlit dashboard for scored UNSW-NB15 connections."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Network Anomaly Tracker", layout="wide")

# ---------------------------------------------------------------------------
# Design system — palette, tokens, typography
# ---------------------------------------------------------------------------
# A warm-toned, high-contrast palette inspired by industrial monitoring
# consoles. Off-black base with amber signal accents, avoiding cold
# corporate blues and generic SaaS aesthetics.

_BASE          = "#111110"     # near-black with warm undertone
_SURFACE       = "#1a1918"     # primary surface — panels, cards
_SURFACE_2     = "#232221"     # secondary surface — raised elements
_SURFACE_3     = "#2c2a28"     # tertiary — hover states, insets
_RULE          = "#302e2b"     # dividers and borders
_RULE_SUBTLE   = "#262422"     # very faint structural lines

_TEXT_HI       = "#e8e4de"     # primary text — high contrast
_TEXT_MID      = "#b5afa5"     # secondary text — labels, descriptions
_TEXT_LO       = "#7a756d"     # tertiary text — captions, muted info
_TEXT_DIM      = "#524e48"     # very low priority — footnotes

_AMBER         = "#d4a54a"     # primary accent — signal, emphasis
_AMBER_DIM     = "#8a7034"     # muted amber — borders, subtle markers
_RED           = "#c44b4b"     # anomaly / flagged — intentionally muted
_RED_DIM       = "#6b3030"     # muted red — backgrounds, borders
_TEAL          = "#4a8f8a"     # secondary accent — normal traffic, safe
_TEAL_DIM      = "#2d5553"     # muted teal — backgrounds

# Typography: Google Fonts loaded via HTML.
# IBM Plex Mono for data and numbers, Inter for UI text,
# Instrument Serif for editorial headings.
_FONT_DISPLAY  = "'Instrument Serif', Georgia, serif"
_FONT_BODY     = "'Inter', -apple-system, sans-serif"
_FONT_MONO     = "'IBM Plex Mono', 'SF Mono', monospace"


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Instrument+Serif&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  /* ================================================================
     BASE LAYER — page, scrollbar, native Streamlit elements
     ================================================================ */
  .stApp {{
      background-color: {_BASE};
      font-family: {_FONT_BODY};
  }}
  header[data-testid="stHeader"] {{
      background-color: {_BASE};
      border-bottom: 1px solid {_RULE_SUBTLE};
  }}
  footer {{ visibility: hidden; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {_BASE}; }}
  ::-webkit-scrollbar-thumb {{ background: {_RULE}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {_TEXT_LO}; }}

  /* Default text overrides */
  h1, h2, h3, h4, h5, h6 {{
      font-family: {_FONT_BODY} !important;
      color: {_TEXT_HI} !important;
  }}
  p, li, span, div {{ color: {_TEXT_MID}; }}
  a {{ color: {_AMBER}; text-decoration: none; }}
  a:hover {{ color: {_TEXT_HI}; }}


  /* ================================================================
     SIDEBAR
     ================================================================ */
  section[data-testid="stSidebar"] {{
      background-color: {_SURFACE};
      border-right: 1px solid {_RULE_SUBTLE};
  }}
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] label {{
      color: {_TEXT_MID} !important;
      font-size: 0.8rem;
      font-family: {_FONT_BODY};
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stTextInput label {{
      font-weight: 600;
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {_TEXT_LO} !important;
      margin-bottom: 3px;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: {_RULE};
      margin: 1rem 0;
      opacity: 0.5;
  }}
  /* Sidebar input styling */
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stMultiSelect > div > div {{
      background-color: {_SURFACE_2} !important;
      border-color: {_RULE} !important;
      color: {_TEXT_HI} !important;
      font-size: 0.82rem;
  }}


  /* ================================================================
     MASTHEAD — top header bar
     ================================================================ */
  .masthead {{
      display: flex;
      align-items: baseline;
      gap: 1rem;
      padding: 0.3rem 0 0.8rem 0;
      border-bottom: 1px solid {_RULE};
      margin-bottom: 1.2rem;
  }}
  .masthead-title {{
      font-family: {_FONT_DISPLAY};
      font-size: 1.65rem;
      font-weight: 400;
      color: {_TEXT_HI};
      letter-spacing: -0.01em;
      line-height: 1.2;
  }}
  .masthead-badge {{
      font-family: {_FONT_MONO};
      font-size: 0.6rem;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {_AMBER};
      background: {_SURFACE_2};
      border: 1px solid {_AMBER_DIM};
      border-radius: 2px;
      padding: 0.2rem 0.55rem;
      white-space: nowrap;
  }}


  /* ================================================================
     KPI STRIP — top metrics row
     ================================================================ */
  .kpi-strip {{
      display: flex;
      gap: 1px;
      background: {_RULE_SUBTLE};
      border: 1px solid {_RULE_SUBTLE};
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 0.4rem;
  }}
  .kpi-cell {{
      flex: 1;
      background: {_SURFACE};
      padding: 0.9rem 1.1rem;
      min-width: 0;
  }}
  .kpi-cell:first-child {{ border-radius: 3px 0 0 3px; }}
  .kpi-cell:last-child {{ border-radius: 0 3px 3px 0; }}
  .kpi-label {{
      font-family: {_FONT_MONO};
      font-size: 0.62rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {_TEXT_LO};
      margin-bottom: 0.35rem;
  }}
  .kpi-value {{
      font-family: {_FONT_MONO};
      font-size: 1.4rem;
      font-weight: 600;
      color: {_TEXT_HI};
      line-height: 1.15;
      font-variant-numeric: tabular-nums;
  }}
  .kpi-value.alert {{ color: {_AMBER}; }}
  .kpi-unit {{
      font-family: {_FONT_MONO};
      font-size: 0.7rem;
      font-weight: 400;
      color: {_TEXT_LO};
      margin-left: 0.2rem;
  }}


  /* ================================================================
     SECTION ANATOMY — headings, captions, dividers
     ================================================================ */
  .section-rule {{
      border: none;
      border-top: 1px solid {_RULE};
      margin: 1.8rem 0 1.2rem 0;
  }}
  .section-hd {{
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
      margin-bottom: 0.45rem;
  }}
  .section-hd-title {{
      font-family: {_FONT_DISPLAY};
      font-size: 1.2rem;
      font-weight: 400;
      color: {_TEXT_HI};
      letter-spacing: -0.005em;
  }}
  .section-hd-tag {{
      font-family: {_FONT_MONO};
      font-size: 0.58rem;
      font-weight: 500;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: {_TEXT_DIM};
  }}
  .section-caption {{
      font-size: 0.76rem;
      color: {_TEXT_LO};
      line-height: 1.55;
      margin-bottom: 0.6rem;
      max-width: 72ch;
  }}


  /* ================================================================
     EVAL PANEL — Precision/Recall/F1 + TP/FP/TN/FN
     ================================================================ */
  .eval-row {{
      display: flex;
      gap: 1px;
      background: {_RULE_SUBTLE};
      border: 1px solid {_RULE_SUBTLE};
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 0.5rem;
  }}
  .eval-cell {{
      flex: 1;
      background: {_SURFACE};
      padding: 0.75rem 0.9rem;
      text-align: center;
  }}
  .eval-cell-label {{
      font-family: {_FONT_MONO};
      font-size: 0.58rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {_TEXT_LO};
      margin-bottom: 0.25rem;
  }}
  .eval-cell-value {{
      font-family: {_FONT_MONO};
      font-size: 1.25rem;
      font-weight: 600;
      color: {_TEXT_HI};
      font-variant-numeric: tabular-nums;
  }}
  .eval-cell-value.dim {{ color: {_TEXT_MID}; font-size: 1.1rem; }}


  /* ================================================================
     CHART WRAPPER
     ================================================================ */
  .chart-container {{
      background: {_SURFACE};
      border: 1px solid {_RULE_SUBTLE};
      border-radius: 4px;
      padding: 0.6rem 0.5rem 0.2rem 0.5rem;
      margin-bottom: 0.2rem;
  }}
  .stPlotlyChart {{ border-radius: 4px; overflow: hidden; }}


  /* ================================================================
     DATAFRAME OVERRIDES
     ================================================================ */
  .stDataFrame {{
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid {_RULE_SUBTLE};
  }}


  /* ================================================================
     DOWNLOAD BUTTON
     ================================================================ */
  .stDownloadButton > button {{
      background-color: {_SURFACE} !important;
      border: 1px solid {_RULE} !important;
      color: {_TEXT_MID} !important;
      font-family: {_FONT_MONO} !important;
      font-size: 0.72rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.04em !important;
      text-transform: uppercase !important;
      padding: 0.5rem 1.2rem !important;
      border-radius: 3px !important;
      transition: all 0.15s ease;
  }}
  .stDownloadButton > button:hover {{
      border-color: {_AMBER_DIM} !important;
      color: {_AMBER} !important;
      background-color: {_SURFACE_2} !important;
  }}


  /* ================================================================
     EXPANDER
     ================================================================ */
  .stExpander {{
      border-color: {_RULE_SUBTLE} !important;
      background: {_SURFACE} !important;
      border-radius: 4px !important;
  }}
  .stExpander summary {{
      font-family: {_FONT_MONO};
      font-size: 0.72rem;
      font-weight: 500;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: {_TEXT_LO} !important;
  }}
  .stExpander summary:hover {{
      color: {_TEXT_MID} !important;
  }}


  /* ================================================================
     UTILITY
     ================================================================ */
  .spacer-xs {{ height: 0.3rem; }}
  .spacer-sm {{ height: 0.6rem; }}
  .spacer-md {{ height: 1rem; }}

  .footnote {{
      font-size: 0.7rem;
      color: {_TEXT_DIM};
      line-height: 1.5;
      max-width: 80ch;
  }}

  /* Sidebar brand mark */
  .sidebar-brand {{
      font-family: {_FONT_DISPLAY};
      font-size: 0.95rem;
      color: {_TEXT_LO};
      padding-bottom: 0.6rem;
      border-bottom: 1px solid {_RULE};
      margin-bottom: 0.8rem;
  }}
  .sidebar-section-label {{
      font-family: {_FONT_MONO};
      font-size: 0.58rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {_TEXT_DIM};
      margin-bottom: 0.3rem;
      margin-top: 0.2rem;
  }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: human-readable byte formatting
# ---------------------------------------------------------------------------
def _fmt_bytes(value: float) -> tuple[str, str]:
    """Format byte count into (number, unit) for split rendering."""
    if value == 0:
        return "0", "B"
    if not math.isfinite(value):
        return "—", ""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(abs(value), 1024)))
    i = min(i, len(units) - 1)
    scaled = value / (1024 ** i)
    if scaled >= 100:
        return f"{scaled:,.0f}", units[i]
    elif scaled >= 10:
        return f"{scaled:,.1f}", units[i]
    else:
        return f"{scaled:,.2f}", units[i]


# ---------------------------------------------------------------------------
# Plotly layout defaults
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=_SURFACE,
    plot_bgcolor="#14140f",
    font=dict(family="Inter, -apple-system, sans-serif", size=11, color=_TEXT_MID),
    margin=dict(l=55, r=18, t=28, b=48),
    hoverlabel=dict(bgcolor=_SURFACE_3, font_size=11, font_color=_TEXT_HI,
                    bordercolor=_RULE),
)

_GRID_COLOR = "#252320"
_ZERO_COLOR = "#252320"


# ---------------------------------------------------------------------------
# MASTHEAD
# ---------------------------------------------------------------------------
st.markdown("""
<div class="masthead">
    <span class="masthead-title">Network Anomaly Tracker</span>
    <span class="masthead-badge">Data Exfiltration Analysis</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"""
<div class="sidebar-brand">Anomaly Tracker</div>
<div class="sidebar-section-label">Data Source</div>
""", unsafe_allow_html=True)

directory = Path(st.sidebar.text_input("Output directory", "outputs"))

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">Filters</div>',
                    unsafe_allow_html=True)

parquet = directory / "processed_scored_logs.parquet"
csv = directory / "processed_scored_logs.csv"
source = parquet if parquet.exists() else csv
if not source.exists():
    st.info("Run main.py first. The dashboard reads processed_scored_logs.parquet "
            "(or .csv) and metrics.json from the output directory.")
    st.stop()


@st.cache_data
def read_outputs(data_path, metrics_path, data_mtime, metrics_mtime):
    flows = (pd.read_parquet(data_path) if data_path.suffix == ".parquet"
             else pd.read_csv(data_path))
    metrics = (json.loads(metrics_path.read_text(encoding="utf-8"))
               if metrics_path.exists() else {})
    return flows, metrics


metrics_path = directory / "metrics.json"
df, metrics = read_outputs(source, metrics_path, source.stat().st_mtime,
                           metrics_path.stat().st_mtime if metrics_path.exists() else 0)
if df.empty:
    st.warning("No scored connections in this output.")
    st.stop()

status = st.sidebar.selectbox("Show", ["All", "Flagged", "Unflagged"])
if status == "Flagged":
    df = df[df["anomaly_prediction"] == -1]
elif status == "Unflagged":
    df = df[df["anomaly_prediction"] == 1]
if "attack_cat" in df:
    selected = st.sidebar.multiselect("Attack category (evaluation metadata)",
                                      sorted(df["attack_cat"].astype(str).unique()))
    if selected:
        df = df[df["attack_cat"].astype(str).isin(selected)]
if "service" in df:
    services = st.sidebar.multiselect("Service",
                                      sorted(df["service"].astype(str).unique()))
    if services:
        df = df[df["service"].astype(str).isin(services)]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div style="font-size:0.68rem;color:{_TEXT_DIM};line-height:1.6;padding:0.2rem 0;">'
    'Anomalies are investigation leads, not proof of exfiltration. '
    'The processed splits do not support IP or timestamp forensics.'
    '</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI STRIP
# ---------------------------------------------------------------------------
flagged = df[df["anomaly_prediction"] == -1]
bytes_val, bytes_unit = _fmt_bytes(flagged["sbytes"].sum())
first_id = str(flagged["id"].min()) if len(flagged) else "—"

st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-cell">
        <div class="kpi-label">Visible Flows</div>
        <div class="kpi-value">{len(df):,}</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Flagged Flows</div>
        <div class="kpi-value alert">{len(flagged):,}</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">Flagged Src Bytes</div>
        <div class="kpi-value">{bytes_val}<span class="kpi-unit">{bytes_unit}</span></div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-label">First Flagged ID</div>
        <div class="kpi-value">{first_id}</div>
    </div>
</div>
<div class="footnote">
    Source bytes are the dataset's <em>sbytes</em> summed for flagged rows.
    ID order is not a timeline.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ANOMALY SCATTER PLOT
# ---------------------------------------------------------------------------
if not df.empty:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hd">
        <span class="section-hd-title">Traffic Analysis</span>
        <span class="section-hd-tag">Connection ID vs Transfer Rate</span>
    </div>
    <div class="section-caption">
        Top 5,000 connections by anomaly score. Y-axis uses a logarithmic scale
        to preserve readability across several orders of magnitude.
    </div>
    """, unsafe_allow_html=True)

    # Plot a bounded slice for responsive dashboards on large datasets.
    plot = df.sort_values("anomaly_score", ascending=False).head(5000).copy()
    plot["status"] = plot["anomaly_prediction"].map({-1: "Flagged", 1: "Normal"})

    # Clamp zero / negative to a small positive for log scale
    plot["bps_plot"] = plot["bytes_per_second"].clip(lower=0.1)

    fig = px.scatter(
        plot, x="id", y="bps_plot", color="status",
        hover_data={"anomaly_score": ":.4f", "sbytes": ":,", "dbytes": ":,",
                    "bytes_per_second": ":,.0f", "bps_plot": False},
        color_discrete_map={"Flagged": _RED, "Normal": _TEAL},
        labels={"id": "Connection ID", "bps_plot": "Bytes/s", "status": "Status"},
        category_orders={"status": ["Flagged", "Normal"]},
    )
    fig.update_traces(
        marker=dict(size=4.5, opacity=0.7,
                    line=dict(width=0.4, color="rgba(0,0,0,0.3)")),
        selector=dict(name="Normal"),
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.9,
                    line=dict(width=0.5, color=_RED_DIM)),
        selector=dict(name="Flagged"),
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        yaxis=dict(type="log", gridcolor=_GRID_COLOR, zerolinecolor=_ZERO_COLOR,
                   title=dict(text="Bytes / Second", font=dict(size=11, color=_TEXT_LO)),
                   tickfont=dict(family="IBM Plex Mono", size=10, color=_TEXT_LO)),
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_ZERO_COLOR,
                   title=dict(text="Connection ID", font=dict(size=11, color=_TEXT_LO)),
                   tickfont=dict(family="IBM Plex Mono", size=10, color=_TEXT_LO)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10.5, color=_TEXT_MID),
            itemsizing="constant",
        ),
        height=420,
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SUSPICIOUS CONNECTIONS TABLE
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown("""
<div class="section-hd">
    <span class="section-hd-title">Suspicious Connections</span>
    <span class="section-hd-tag">Top 100 by anomaly score</span>
</div>
""", unsafe_allow_html=True)

shown = [c for c in ["id", "anomaly_score", "anomaly_prediction", "sbytes", "dbytes",
                     "total_bytes", "bytes_per_second", "outbound_fraction", "proto",
                     "service", "state", "attack_cat", "label"] if c in df]
table_df = df.sort_values("anomaly_score", ascending=False)[shown].head(100).copy()

_COL_LABELS = {
    "id": "ID", "anomaly_score": "Anomaly Score", "anomaly_prediction": "Prediction",
    "sbytes": "Src Bytes", "dbytes": "Dst Bytes", "total_bytes": "Total Bytes",
    "bytes_per_second": "Bytes/s", "outbound_fraction": "Outbound Frac",
    "proto": "Protocol", "service": "Service", "state": "State",
    "attack_cat": "Attack Category", "label": "Label",
}

_COL_FORMAT = {}
for col in table_df.columns:
    if col == "anomaly_score":
        _COL_FORMAT[col] = "%.4f"
    elif col == "outbound_fraction":
        _COL_FORMAT[col] = "%.3f"
    elif col in ("sbytes", "dbytes", "total_bytes", "bytes_per_second"):
        _COL_FORMAT[col] = "%,.0f"

table_display = table_df.rename(columns=_COL_LABELS)

column_config = {}
for orig_col, fmt in _COL_FORMAT.items():
    display_col = _COL_LABELS.get(orig_col, orig_col)
    column_config[display_col] = st.column_config.NumberColumn(format=fmt)

st.dataframe(table_display, use_container_width=True, hide_index=True,
             column_config=column_config)


# ---------------------------------------------------------------------------
# EVALUATION SECTION
# ---------------------------------------------------------------------------
evaluation = metrics.get("evaluation", {})
if evaluation.get("available"):
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-hd">
        <span class="section-hd-title">Model Evaluation</span>
        <span class="section-hd-tag">Complete held-out test split</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Precision / Recall / F1 ---
    prec = f"{evaluation['precision']:.3f}"
    rec  = f"{evaluation['recall']:.3f}"
    f1   = f"{evaluation['f1']:.3f}"

    st.markdown(f"""
    <div class="eval-row">
        <div class="eval-cell">
            <div class="eval-cell-label">Precision</div>
            <div class="eval-cell-value">{prec}</div>
        </div>
        <div class="eval-cell">
            <div class="eval-cell-label">Recall</div>
            <div class="eval-cell-value">{rec}</div>
        </div>
        <div class="eval-cell">
            <div class="eval-cell-label">F1 Score</div>
            <div class="eval-cell-value">{f1}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TP / FP / TN / FN ---
    tp, fp = evaluation["tp"], evaluation["fp"]
    tn, fn = evaluation["tn"], evaluation["fn"]

    st.markdown(f"""
    <div class="eval-row">
        <div class="eval-cell">
            <div class="eval-cell-label">True Pos</div>
            <div class="eval-cell-value dim">{tp:,}</div>
        </div>
        <div class="eval-cell">
            <div class="eval-cell-label">False Pos</div>
            <div class="eval-cell-value dim">{fp:,}</div>
        </div>
        <div class="eval-cell">
            <div class="eval-cell-label">True Neg</div>
            <div class="eval-cell-value dim">{tn:,}</div>
        </div>
        <div class="eval-cell">
            <div class="eval-cell-label">False Neg</div>
            <div class="eval-cell-value dim">{fn:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Confusion Matrix ---
    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-caption" style="margin-bottom:0.35rem;">
        Confusion matrix — rows are actual labels, columns are predictions.
    </div>
    """, unsafe_allow_html=True)

    cm = evaluation["confusion_matrix"]

    # Build an annotated heatmap with Plotly
    x_labels = ["Predicted Normal", "Predicted Anomaly"]
    y_labels = ["Actual Normal", "Actual Attack"]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        text=[[f"{v:,}" for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=14, family="IBM Plex Mono", color=_TEXT_HI),
        colorscale=[
            [0.0, "#1a1918"],
            [0.3, _TEAL_DIM],
            [1.0, _TEAL],
        ],
        showscale=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Count: %{text}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig_cm.update_layout(
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k != "margin"},
        margin=dict(l=12, r=12, t=12, b=12),
        height=210,
        xaxis=dict(side="bottom", tickfont=dict(family="IBM Plex Mono", size=10.5, color=_TEXT_LO)),
        yaxis=dict(autorange="reversed", tickfont=dict(family="IBM Plex Mono", size=10.5, color=_TEXT_LO)),
    )

    # Two-column layout: matrix left, per-category right (if available)
    if "per_category" in evaluation:
        cm_col, cat_col = st.columns([2, 3])
    else:
        cm_col = st.columns([1])[0]
        cat_col = None

    with cm_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Per-category detection ---
    if cat_col is not None:
        with cat_col:
            st.markdown(f"""
            <div class="section-caption" style="margin-bottom:0.35rem;">
                Detection rates by attack category across the full test split.
            </div>
            """, unsafe_allow_html=True)

            cat_df = pd.DataFrame(evaluation["per_category"])
            cat_display = cat_df.rename(columns={
                "attack_cat": "Category", "total": "Total",
                "attack_rows": "Attack Rows", "flagged": "Flagged",
                "detected_attacks": "Detected", "attack_recall": "Recall",
            })

            cat_col_config = {
                "Total": st.column_config.NumberColumn(format="%d"),
                "Attack Rows": st.column_config.NumberColumn(format="%d"),
                "Flagged": st.column_config.NumberColumn(format="%d"),
                "Detected": st.column_config.NumberColumn(format="%d"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
            }
            st.dataframe(cat_display, hide_index=True, use_container_width=True,
                         column_config=cat_col_config, height=250)

    st.markdown(f'<div class="footnote" style="margin-top:0.5rem;">'
                f'Evaluation metrics cover the complete held-out test split '
                f'and are unaffected by sidebar filters.</div>',
                unsafe_allow_html=True)
else:
    st.info("Evaluation requires labels in the held-out test split.")


# ---------------------------------------------------------------------------
# DOWNLOAD & NOTES
# ---------------------------------------------------------------------------
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

dl_col, spacer_col, note_col = st.columns([1.2, 0.3, 3])
with dl_col:
    st.download_button("Export Visible Connections (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "visible_scored_connections.csv", "text/csv")
with note_col:
    with st.expander("Dataset limitations & notes"):
        st.markdown(f"""
<div style="font-size:0.76rem;color:{_TEXT_LO};line-height:1.65;">
<ul style="padding-left:1.1rem;margin:0;">
<li>Anomalies are investigation leads, not proof of exfiltration.</li>
<li>The processed splits do not support IP or timestamp forensics.</li>
<li>Source bytes (<em>sbytes</em>) are the dataset's recorded values, summed for flagged rows.</li>
<li>ID order is an ordering proxy, not a timeline or causal sequence.</li>
<li>Evaluation metrics cover the complete held-out test split and are unaffected by sidebar filters.</li>
</ul>
</div>
""", unsafe_allow_html=True)
