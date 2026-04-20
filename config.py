"""
config.py — Constants, chart helpers, and light-only CSS theme.
"""

# ── Physics constants (UNSCEAR 1988) ──────────────────────────────────────────
TH_ACTIVITY    = 75_000.0
U_ACTIVITY     =  4_000.0
K40_DEFAULT    =    350.0
HBRA_THRESHOLD =    370.0
CF_TH, CF_U, CF_K = 0.604, 0.462, 0.0417
TISSUE_WT      = 0.7
RISK_CRITICAL  = 740.0

MINERAL_FEATURES = [
    "Ilmenite","Rutile","Zircon","Garnet",
    "Sillimanite","Kyanite","Leucoxene","Others","Heavies",
]

# ── Colours ────────────────────────────────────────────────────────────────────
ACCENT      = "#1a73e8"   # Google blue  — interactive elements
RIVER       = "#006d77"   # Deep teal    — nature accent
DANGER_COL  = "#c5221f"
SUCCESS_COL = "#137333"
WARN_COL    = "#f9ab00"

# ── Plotly — base has NO xaxis/yaxis to avoid duplicate-kwarg errors ──────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, system-ui, sans-serif", size=14, color="#202124"),
    legend=dict(font=dict(size=13, color="#202124"), bgcolor="rgba(255,255,255,.9)"),
    margin=dict(t=46, b=36, l=12, r=12),
)

_G = dict(
    gridcolor="rgba(180,180,180,.22)",
    zerolinecolor="rgba(180,180,180,.30)",
    tickfont=dict(color="#5f6368", size=12),
    title_font=dict(color="#5f6368", size=13),
)


def chart_layout(**extra) -> dict:
    """Build update_layout dict with PLOT_BASE + default axes + caller extras.
    Deep-merges xaxis/yaxis so grid style is always preserved."""
    layout = {**PLOT_BASE, "xaxis": dict(**_G), "yaxis": dict(**_G)}
    for k, v in extra.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            layout[k] = {**_G, **v}
        else:
            layout[k] = v
    return layout


# ── CSS ────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ──────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  background-color: #ffffff !important;
  color: #202124 !important;
  font-size: 18px !important;
  line-height: 1.6 !important;
}

/* Heading scale */
h1 { font-size: 38px !important; font-weight: 700 !important; color: #202124 !important; line-height:1.2 !important; }
h2 { font-size: 28px !important; font-weight: 600 !important; color: #202124 !important; }
h3 { font-size: 22px !important; font-weight: 600 !important; color: #202124 !important; }
h4 { font-size: 18px !important; font-weight: 600 !important; color: #202124 !important; }
h5, h6 { font-size: 16px !important; font-weight: 500 !important; color: #5f6368 !important; }

/* Mobile heading scale */
@media (max-width: 768px) {
  h1 { font-size: 28px !important; }
  h2 { font-size: 22px !important; }
  h3 { font-size: 18px !important; }
}

/* Streamlit header bar */
header[data-testid="stHeader"], [data-testid="stToolbar"] {
  background: #ffffff !important;
  border-bottom: 1px solid #dadce0 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #f8f9fa !important;
  border-right: 1px solid #dadce0 !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p { color: #202124 !important; font-size: 15px !important; }

/* ── Selectbox ────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div {
  background: #ffffff !important; border: 1px solid #dadce0 !important;
  border-radius: 8px !important; color: #202124 !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] span { color: #202124 !important; font-size: 15px !important; }

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #ffffff !important; border: 1px solid #dadce0 !important;
  border-radius: 8px !important; padding: 14px 18px !important;
  box-shadow: 0 1px 3px rgba(60,64,67,.12) !important;
}
[data-testid="stMetricValue"] { color: #202124 !important; font-weight: 600 !important; font-size: 24px !important; }
[data-testid="stMetricLabel"] { color: #5f6368 !important; font-size: 12px !important; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid #dadce0 !important;
}
.stTabs [data-baseweb="tab"] {
  color: #5f6368 !important; font-size: 15px !important; font-weight: 500 !important;
  padding: 10px 20px !important; border-bottom: 2px solid transparent !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] { color: #1a73e8 !important; border-bottom-color: #1a73e8 !important; }

/* ── Expander ─────────────────────────────────────────────────── */
div[data-testid="stExpander"],
div[data-testid="stExpander"] > div,
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
  background: #ffffff !important; color: #202124 !important;
  border: 1px solid #dadce0 !important; border-radius: 8px !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  background: #1a73e8 !important; color: #fff !important; border: none !important;
  border-radius: 6px !important; font-size: 14px !important; font-weight: 500 !important;
  padding: 8px 22px !important; transition: opacity .15s !important;
}
[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ── Clean HTML data table ────────────────────────────────────── */
.rt { width:100%; border-collapse:collapse; font-size:15px; }
.rt th {
  background:#f0f4f8; border:1px solid #b0bec5; padding:9px 12px;
  text-align:left; font-weight:600; color:#202124; font-size:13px;
  text-transform:uppercase; letter-spacing:.5px;
}
.rt td { border:1px solid #dadce0; padding:8px 12px; color:#202124; background:#ffffff; }
.rt tr:nth-child(even) td { background:#f8fafb; }
.rt tr:hover td { background:#e8f0fe; }

/* ── Hero banner ──────────────────────────────────────────────── */
.nature-hero {
  background: linear-gradient(135deg, #004e89 0%, #006d77 45%, #1a936f 100%);
  border-radius: 12px; padding: 28px 32px; margin-bottom: 20px; color: #fff;
  position: relative; overflow: hidden;
}
.nature-hero::after {
  content: '';
  position: absolute; bottom: -20px; right: -20px;
  width: 200px; height: 200px;
  background: rgba(255,255,255,.06);
  border-radius: 50%;
}
.nature-hero h1 { font-size: 32px !important; font-weight: 700 !important;
                  color: #fff !important; margin: 0 0 6px; }
.nature-hero p  { font-size: 16px !important; color: rgba(255,255,255,.85) !important;
                  margin: 0; line-height: 1.5; }

/* ── Custom components ────────────────────────────────────────── */
.stat-card {
  background: #ffffff; border: 1px solid #dadce0; border-radius: 8px;
  padding: 16px 20px; box-shadow: 0 1px 3px rgba(60,64,67,.12);
}
.stat-label {
  font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
  color: #5f6368; margin-bottom: 4px;
}
.stat-value { font-size: 30px; font-weight: 700; color: #202124; font-variant-numeric: tabular-nums; }
.stat-sub   { font-size: 15px; color: #5f6368; margin-top: 4px; }

.section-label {
  font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
  color: #5f6368; margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #dadce0;
}

.badge-danger  { background:#fce8e6; color:#c5221f; padding:4px 12px; border-radius:4px; font-size:15px; font-weight:600; display:inline-block; }
.badge-warning { background:#fef7e0; color:#b06000; padding:4px 12px; border-radius:4px; font-size:15px; font-weight:600; display:inline-block; }
.badge-success { background:#e6f4ea; color:#137333; padding:4px 12px; border-radius:4px; font-size:15px; font-weight:600; display:inline-block; }

.detail-row { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #f1f3f4; font-size:16px; }
.detail-key { color:#5f6368; }
.detail-val { font-weight:500; color:#202124; font-family:'JetBrains Mono',monospace; font-size:14px; }

/* Nature link card */
.nature-link {
  display: inline-flex; align-items: center; gap: 8px;
  background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 8px;
  padding: 10px 16px; text-decoration: none; color: #137333;
  font-size: 13px; font-weight: 500; margin: 4px 4px 4px 0;
  transition: background .15s;
}
.nature-link:hover { background: #c8e6c9; }

@media (max-width:768px) {
  .main .block-container { padding:.8rem !important; }
  .stat-value { font-size: 22px !important; }
}

/* Author credit — fixed bottom-right corner */
.author-credit {
  position: fixed; bottom: 14px; right: 18px; z-index: 9999;
  background: rgba(255,255,255,.92); backdrop-filter: blur(6px);
  border: 1px solid #dadce0; border-radius: 20px;
  padding: 5px 14px; font-size: 13px; color: #5f6368;
  font-family: 'DM Sans', sans-serif; font-weight: 500;
  pointer-events: none;
  box-shadow: 0 1px 4px rgba(60,64,67,.15);
}
</style>
"""