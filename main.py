"""
main.py — Thamirabarani Basin Monazite Radioactivity Analysis
UI only — all physics/ML delegated to config, data, radiation, models.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from config import (
    HBRA_THRESHOLD, K40_DEFAULT, MINERAL_FEATURES,
    ACCENT, RIVER, DANGER_COL, SUCCESS_COL, WARN_COL,
    THEME_CSS, chart_layout,
)
from data import load_basin, load_irel
from radiation import compute as rad_compute, risk_label, sweep as rad_sweep
from models import (
    augment, train_models, loo_evaluate, bootstrap_ci,
    pca_project, cluster_stations, spatial_grid,
    anomaly_scores, permutation_importance, fit_regression,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thamirabarani Radioactivity Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
basin_df = load_basin()
irel_df  = load_irel()


# ── UI helpers ─────────────────────────────────────────────────────────────────
def stat_card(label: str, value: str, sub: str = "", col=None) -> None:
    html = (f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div>'
            + (f'<div class="stat-sub">{sub}</div>' if sub else "")
            + "</div>")
    (col or st).markdown(html, unsafe_allow_html=True)


def detail_row(key: str, val: str) -> str:
    return (f'<div class="detail-row"><span class="detail-key">{key}</span>'
            f'<span class="detail-val">{val}</span></div>')


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def badge(rc: str, label: str) -> None:
    st.markdown(f'<span class="badge-{rc}" style="margin:8px 0">{label}</span>',
                unsafe_allow_html=True)


def html_table(df: pd.DataFrame) -> None:
    """Render a DataFrame as a clean, styled HTML table — always visible."""
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df.values
    )
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    st.markdown(
        f'<div style="overflow-x:auto"><table class="rt"><thead><tr>{headers}</tr>'
        f'</thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Thamirabarani Basin**")
    st.markdown("Monazite Radioactivity Analysis")
    st.divider()
    MODE = st.radio("Navigation", [
        "Dashboard", "Live Calculator", "Regression Analysis",
        "ML Prediction", "Spatial Heatmap", "Reference",
    ])
    st.divider()
    station_filter = st.selectbox("Station Set",
                                  ["All Stations", "Main Channel", "Branch Only"])
    occ_factor = st.slider("Occupancy Factor", 0.10, 1.0, 0.20, 0.05,
                           help="Fraction of time spent at the sampling site.")
    st.divider()
    st.caption(
        "Method: Stoichiometric (UNSCEAR 1988)  \n"
        "Threshold: Ra-eq ≤ 370 Bq/kg"
    )


# ── Filter + compute ───────────────────────────────────────────────────────────
_mask = {"Main Channel": basin_df["type"] == "Main",
         "Branch Only":  basin_df["type"] == "Branch"}.get(
    station_filter, pd.Series([True] * len(basin_df)))
fdf = basin_df[_mask].reset_index(drop=True)
m   = rad_compute(fdf["monazite"].values, fdf["k40"].values, occ_factor)
for k, v in m.items():
    fdf[k] = v.tolist() if hasattr(v, "tolist") else v


# ── Nature hero banner ─────────────────────────────────────────────────────────
st.markdown("""
<div class="nature-hero">
  <h1>Thamirabarani Basin — Monazite Risk Assessment</h1>
  <p>Environmental radioactivity study of river sediments from Pechiparai reservoir
  to Thengapattanam estuary, Kanyakumari District, Tamil Nadu &nbsp;|&nbsp;
  Mineralogical data: IREL (India) Limited &nbsp;|&nbsp; UNSCEAR 1988 framework</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
section_label("Basin Summary")
k1, k2, k3, k4, k5 = st.columns(5)
hbra_n  = int((fdf["ra_eq"] > HBRA_THRESHOLD).sum())
max_idx = fdf["gamma"].idxmax()
for col, lbl, val, sub in [
    (k1, "Max Gamma",  f"{fdf['gamma'].max():.4f} µSv/h", fdf.loc[max_idx,"station"]),
    (k2, "Max Ra-eq",  f"{fdf['ra_eq'].max():.1f} Bq/kg", "HBRA > 370"),
    (k3, "Mean Ra-eq", f"{fdf['ra_eq'].mean():.1f} Bq/kg","Basin average"),
    (k4, "HBRA Sites", f"{hbra_n} / {len(fdf)}",          "Exceeds threshold"),
    (k5, "Max ELCR",   f"{fdf['elcr'].max():.2e}",         "WHO limit < 1e-5"),
]:
    stat_card(lbl, val, sub, col)
st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if MODE == "Dashboard":
    section_label("Geospatial Distribution")

    with st.container():
        fmap = folium.Map(location=[fdf["lat"].mean(), fdf["lon"].mean()],
                          zoom_start=11, tiles="CartoDB positron")
        path_ids = [s for s in ["S1","S2","S3","S4","S5","S6","S7","S8"]
                    if s in fdf["id"].values]
        path_coords = [[fdf.loc[fdf["id"]==s,"lat"].values[0],
                        fdf.loc[fdf["id"]==s,"lon"].values[0]] for s in path_ids]
        if len(path_coords) > 1:
            folium.PolyLine(path_coords, color=ACCENT, weight=2.5,
                            opacity=0.6, dash_array="5 4").add_to(fmap)

        for _, row in fdf.iterrows():
            rl, rc = risk_label(row["ra_eq"])
            col_map = {"danger": DANGER_COL, "warning": WARN_COL, "success": ACCENT}
            m_color = col_map[rc]
            popup_html = f"""<div style="font-family:DM Sans,sans-serif;font-size:13px;min-width:190px;padding:4px">
              <strong style="font-size:15px">{row['station']}</strong>
              <hr style="border:none;border-top:1px solid #dadce0;margin:6px 0">
              <table style="width:100%;border-collapse:collapse">
                <tr><td style="color:#5f6368;padding:2px 0">Geology</td><td style="text-align:right">{row['geology']}</td></tr>
                <tr><td style="color:#5f6368;padding:2px 0">Monazite</td><td style="text-align:right"><strong>{row['monazite']:.2f}%</strong></td></tr>
                <tr><td style="color:#5f6368;padding:2px 0">Ra-eq</td><td style="text-align:right"><strong>{row['ra_eq']:.1f} Bq/kg</strong></td></tr>
                <tr><td style="color:#5f6368;padding:2px 0">Gamma</td><td style="text-align:right">{row['gamma']:.5f} µSv/h</td></tr>
                <tr><td style="color:#5f6368;padding:2px 0">AED</td><td style="text-align:right">{row['aed']:.4f} µSv/yr</td></tr>
              </table>
              <div style="margin-top:6px;padding:3px 8px;border-radius:4px;background:{m_color}22;color:{m_color};font-weight:600;font-size:12px">{rl}</div>
            </div>"""
            folium.CircleMarker(
                [row["lat"], row["lon"]], radius=11,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"{row['station']} — Ra-eq {row['ra_eq']:.0f} Bq/kg",
                color=m_color, weight=2, fill_color=m_color,
                fill=True, fill_opacity=0.75,
            ).add_to(fmap)

        st_folium(fmap, height=500, use_container_width=True)
        st.markdown('<p style="font-size:15px;color:#5f6368;margin-top:6px">Marker colour: Blue = safe &nbsp;·&nbsp; Orange = HBRA (Ra-eq &gt; 370 Bq/kg) &nbsp;·&nbsp; Red = critical (&gt; 740 Bq/kg). Click any marker for full details.</p>', unsafe_allow_html=True)

    section_label("Analysis")
    t1, t2, t3, t4 = st.tabs(["Radiation Profile","Risk Correlation",
                               "Activity Concentrations","Cluster Analysis"])

    with t1:
        risk_colors = [DANGER_COL if r > HBRA_THRESHOLD else WARN_COL
                       if r > 200 else SUCCESS_COL for r in fdf["ra_eq"]]
        fig = go.Figure(go.Bar(
            x=fdf["station"], y=fdf["gamma"], marker_color=risk_colors,
            hovertemplate="<b>%{x}</b><br>Gamma = %{y:.6f} µSv/h<extra></extra>"))
        fig.add_hline(y=0.1, line_dash="dot", line_color=WARN_COL,
                      annotation_text="Moderate risk (0.1)")
        fig.add_hline(y=0.5, line_dash="dot", line_color=DANGER_COL,
                      annotation_text="High risk (0.5)")
        fig.update_layout(**chart_layout(
            title="Effective Gamma Radiation per Station (µSv/h)", height=380))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        # Build traces individually so each point gets alternating text position
        # preventing overlap of closely positioned stations (e.g. S3/S10)
        fig_sc = go.Figure()
        positions = ["top center","bottom center","top right","bottom left",
                     "top left","bottom right","top center","bottom center",
                     "top right","bottom left"]
        for i, (_, row_s) in enumerate(fdf.sort_values("monazite").reset_index(drop=True).iterrows()):
            rc_s = risk_label(row_s["ra_eq"])[1]
            dot_c = {"danger": DANGER_COL, "warning": WARN_COL, "success": SUCCESS_COL}[rc_s]
            fig_sc.add_trace(go.Scatter(
                x=[row_s["monazite"]], y=[row_s["ra_eq"]],
                mode="markers+text", text=[row_s["id"]],
                textposition=positions[i % len(positions)],
                textfont=dict(size=13, color="#202124"),
                marker=dict(size=max(8, row_s["gamma"]*40+8), color=dot_c, opacity=0.85,
                            line=dict(width=1.5, color="white")),
                hovertemplate=(f"<b>{row_s['station']}</b><br>"
                               f"Monazite: {row_s['monazite']:.2f}%<br>"
                               f"Ra-eq: {row_s['ra_eq']:.1f} Bq/kg<br>"
                               f"Gamma: {row_s['gamma']:.5f} µSv/h<extra></extra>"),
                showlegend=False,
            ))
        fig_sc.add_hline(y=HBRA_THRESHOLD, line_dash="dash", line_color=DANGER_COL,
                         annotation_text="HBRA threshold (370 Bq/kg)",
                         annotation_font_color=DANGER_COL)
        fig_sc.update_layout(**chart_layout(
            title="Monazite % vs Radium Equivalent Activity",
            xaxis={"title":"Monazite (%)"}, yaxis={"title":"Ra-eq (Bq/kg)"}, height=460))
        st.plotly_chart(fig_sc, use_container_width=True)

        # Cancer risk contextual information
        st.markdown("""
<div style="background:#fff8e1;border-left:4px solid #f9ab00;border-radius:8px;
            padding:18px 22px;margin:12px 0">
  <div style="font-size:16px;font-weight:600;color:#5f4700;margin-bottom:10px">
    Radioactivity and Cancer Risk — What do these numbers mean?
  </div>
  <div style="font-size:15px;color:#5f6368;line-height:1.7">
    <strong style="color:#202124">Radium Equivalent (Ra-eq)</strong> measures the combined
    external gamma radiation hazard from all three natural radionuclides (Th-232, U-238, K-40).
    Values below <strong>370 Bq/kg</strong> are considered safe by IAEA; values between
    370–740 Bq/kg are classified as High Background Natural Radiation Areas (HBRA).
    <br><br>
    <strong style="color:#202124">Excess Lifetime Cancer Risk (ELCR)</strong> estimates the
    additional probability of developing cancer due to gamma radiation exposure over a 70-year
    lifetime. The WHO acceptable threshold is <strong>1 × 10⁻⁵</strong> (1 extra case per
    100,000 people). All stations in this study fall below this threshold, indicating that
    soil-pathway radiation alone does not pose an unacceptable cancer risk.
    <br><br>
    <strong style="color:#c5221f">Station S7 (Parakanni)</strong> is the only HBRA site in this
    corridor, with Ra-eq = 716.7 Bq/kg driven by its high monazite content (0.62%).
    Long-term residents and agricultural workers near this zone should be considered for
    periodic radiation monitoring as part of precautionary public health practice.
  </div>
</div>""", unsafe_allow_html=True)

    with t3:
        melt = fdf[["station","a_th","a_u","k40"]].melt(
            id_vars="station", var_name="nuclide", value_name="activity")
        melt["nuclide"] = melt["nuclide"].map({"a_th":"Th-232","a_u":"U-238","k40":"K-40"})
        fig = px.bar(melt, x="station", y="activity", color="nuclide", barmode="group",
                     color_discrete_map={"Th-232":ACCENT,"U-238":SUCCESS_COL,"K-40":WARN_COL},
                     labels={"activity":"Activity (Bq/kg)","station":""},
                     title="Activity Concentrations by Radionuclide")
        fig.update_layout(**chart_layout(height=400))
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        ctrl_l, ctrl_r = st.columns([3, 1])
        with ctrl_r:
            n_cl = st.slider("Clusters", 2, 5, 2, key="dash_clust")
            link = st.selectbox("Linkage", ["ward","complete","average"], key="dash_link")
        with ctrl_l:
            cl    = cluster_stations(occ_factor, n_cl, link)
            fig_d, ax = plt.subplots(figsize=(10, 4))
            fig_d.patch.set_alpha(0); ax.set_facecolor("none")
            cut = cl["Z"][-(n_cl - 1), 2] - 1e-6
            dendrogram(cl["Z"], labels=cl["stations"], color_threshold=cut,
                       leaf_rotation=40, leaf_font_size=9, ax=ax,
                       above_threshold_color="#9aa0a6")
            ax.axhline(cut, color=DANGER_COL, linestyle="--", lw=1)
            ax.tick_params(colors="#5f6368")
            for sp in ax.spines.values(): sp.set_edgecolor("#dadce0")
            ax.set_title(f"Dendrogram — {link.capitalize()} linkage",
                         fontsize=12, color="#202124")
            plt.tight_layout(); st.pyplot(fig_d, clear_figure=True)

        cl_df = pd.DataFrame({
            "Station":        cl["stations"],
            "Cluster":        [f"C{c}" for c in cl["cluster_ids"]],
            "Monazite (%)":   np.round(cl["monazite"], 3),
            "Gamma (µSv/h)":  np.round(cl["gamma"], 6),
            "Ra-eq (Bq/kg)":  np.round(cl["ra_eq"], 1),
            "Classification": [risk_label(r)[0] for r in cl["ra_eq"]],
        }).sort_values("Cluster")
        html_table(cl_df)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif MODE == "Live Calculator":
    section_label("Stoichiometric Calculator — UNSCEAR 1988")
    st.caption("All outputs update in real time as you adjust the sliders.")

    c1, c2, c3 = st.columns(3)
    mon = c1.slider("Monazite (%)", 0.00, 2.00, 0.50, 0.01)
    ak  = c2.slider("K-40 (Bq/kg)", 100.0, 600.0, K40_DEFAULT, 10.0)
    of  = c3.slider("Occupancy Factor", 0.10, 1.00, occ_factor, 0.05)

    res = rad_compute(mon, ak, of)
    rl, rc = risk_label(res["ra_eq"])

    gauge_color = {"success": SUCCESS_COL, "warning": WARN_COL, "danger": DANGER_COL}[rc]
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=res["ra_eq"],
        delta={"reference": HBRA_THRESHOLD, "valueformat": ".1f",
               "increasing": {"color": DANGER_COL}, "decreasing": {"color": SUCCESS_COL}},
        number={"suffix": " Bq/kg", "font": {"size": 30, "family": "DM Sans, sans-serif"}},
        title={"text": "Radium Equivalent Activity",
               "font": {"size": 13, "color": "#5f6368"}},
        gauge={"axis": {"range": [0, 900], "tickwidth": 1, "tickcolor": "#dadce0"},
               "bar":  {"color": gauge_color, "thickness": 0.3},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range":[0,200],  "color":"rgba(19,115,51,.08)"},
                         {"range":[200,370], "color":"rgba(249,171,0,.08)"},
                         {"range":[370,740], "color":"rgba(197,34,31,.08)"},
                         {"range":[740,900], "color":"rgba(197,34,31,.15)"}],
               "threshold": {"line":{"color":DANGER_COL,"width":2},"value":HBRA_THRESHOLD}},
    ))
    fig_g.update_layout(height=230, **{k:v for k,v in chart_layout().items()
                                       if k not in ("xaxis","yaxis")})
    st.plotly_chart(fig_g, use_container_width=True)
    badge(rc, rl)

    cols = st.columns(6)
    for col, lbl, val in [
        (cols[0],"A-Th (Bq/kg)",  f"{res['a_th']:.2f}"),
        (cols[1],"A-U (Bq/kg)",   f"{res['a_u']:.2f}"),
        (cols[2],"Dose (nGy/h)",  f"{res['dose_ngy']:.2f}"),
        (cols[3],"Gamma (µSv/h)", f"{res['gamma']:.6f}"),
        (cols[4],"AED (µSv/yr)",  f"{res['aed']:.4f}"),
        (cols[5],"ELCR",          f"{res['elcr']:.3e}"),
    ]:
        col.metric(lbl, val)

    with st.expander("Step-by-step derivation"):
        st.markdown(f"""
| Step | Expression | Result |
|------|-----------|--------|
| Th activity  | ({mon:.2f}/100)×75000 | **{res['a_th']:.3f} Bq/kg** |
| U activity   | ({mon:.2f}/100)×4000  | **{res['a_u']:.3f} Bq/kg** |
| Dose rate    | 0.604×{res['a_th']:.1f}+0.462×{res['a_u']:.1f}+0.0417×{ak:.0f} | **{res['dose_ngy']:.3f} nGy/h** |
| Gamma        | ({res['dose_ngy']:.3f}/1000)×0.7 | **{res['gamma']:.6f} µSv/h** |
| Ra-eq        | {res['a_u']:.2f}+1.43×{res['a_th']:.1f}+0.077×{ak:.0f} | **{res['ra_eq']:.3f} Bq/kg** |
| AED          | {res['gamma']:.5f}×8760×{of}×0.7 | **{res['aed']:.4f} µSv/yr** |
| ELCR         | {res['aed']:.4f}×0.05×70/1e6 | **{res['elcr']:.3e}** |
""")

    section_label("What-If Sensitivity Analysis")
    sw = rad_sweep((0.0, 2.0), ak, of)
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=sw["monazite"], y=sw["ra_eq"], name="Ra-eq (Bq/kg)",
                               line=dict(color=ACCENT, width=2)))
    fig_s.add_trace(go.Scatter(x=sw["monazite"], y=sw["aed"], name="AED (µSv/yr)",
                               yaxis="y2", line=dict(color=SUCCESS_COL, width=2, dash="dot")))
    fig_s.add_vline(x=mon, line_color=WARN_COL, line_dash="dash",
                    annotation_text=f"Current ({mon:.2f}%)", annotation_font_color=WARN_COL)
    fig_s.add_hline(y=HBRA_THRESHOLD, line_color=DANGER_COL, line_dash="dot",
                    annotation_text="HBRA (370 Bq/kg)", annotation_font_color=DANGER_COL)
    fig_s.update_layout(**chart_layout(
        title="Ra-eq and AED vs Monazite %",
        yaxis={"title": "Ra-eq (Bq/kg)"},
        yaxis2=dict(title="AED (µSv/yr)", overlaying="y", side="right",
                    tickfont=dict(color="#5f6368",size=12),
                    title_font=dict(color="#5f6368",size=13)),
        height=360,
        legend=dict(orientation="h", y=1.08, font=dict(color="#202124",size=13)),
    ))
    st.plotly_chart(fig_s, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif MODE == "Regression Analysis":
    section_label("Predictive Regression — Monazite % from Radiometric Proxies")
    st.caption("Ra-eq, dose rate, and gamma are deterministic linear functions of Monazite % in "
               "the stoichiometric model, so R² = 1.0 confirms internal consistency.")

    ctrl1, ctrl2 = st.columns(2)
    xcol_choice = ctrl1.selectbox("Predictor variable", ["ra_eq","dose_ngy","gamma"],
                                  format_func={"ra_eq":"Ra-eq (Bq/kg)",
                                               "dose_ngy":"Dose rate (nGy/h)",
                                               "gamma":"Gamma (µSv/h)"}.get)
    reg_model = ctrl2.radio("Model", ["OLS","Ridge (α=0.1)","Lasso (α=0.01)"],
                            horizontal=True)

    mdl, r2, slope, intercept, y_hat = fit_regression(
        fdf[xcol_choice].values, fdf["monazite"].values, reg_model)
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.6f}")
    m2.metric("Slope", f"{slope:.8f}")
    m3.metric("Intercept", f"{intercept:.6f}")

    rt1, rt2, rt3 = st.tabs(["Regression Fit","Residuals","Predict"])

    with rt1:
        xline = np.linspace(fdf[xcol_choice].min(), fdf[xcol_choice].max(), 300)
        pt_colors = [DANGER_COL if r > HBRA_THRESHOLD else ACCENT for r in fdf["ra_eq"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fdf[xcol_choice], y=fdf["monazite"],
                                 mode="markers+text", text=fdf["id"],
                                 textposition="top center",
                                 marker=dict(size=11, color=pt_colors),
                                 hovertext=fdf["station"], showlegend=False))
        fig.add_trace(go.Scatter(x=xline, y=mdl.predict(xline.reshape(-1,1)),
                                 mode="lines", name=f"{reg_model} (R²={r2:.4f})",
                                 line=dict(color=WARN_COL, width=2, dash="dot")))
        fig.update_layout(**chart_layout(title=f"Monazite % vs {xcol_choice}", height=420))
        st.plotly_chart(fig, use_container_width=True)

    with rt2:
        residuals = fdf["monazite"].values - y_hat
        fig = px.bar(x=fdf["id"], y=residuals, color=residuals,
                     color_continuous_scale="RdBu",
                     labels={"x":"Station","y":"Residual (Actual − Predicted)"},
                     title="Regression Residuals")
        fig.add_hline(y=0, line_color="#5f6368", line_width=1)
        fig.update_layout(**chart_layout(height=360))
        st.plotly_chart(fig, use_container_width=True)

    with rt3:
        new_x = st.number_input(f"Enter {xcol_choice}:",
                                value=float(fdf[xcol_choice].mean()))
        pred  = max(0.0, float(mdl.predict([[new_x]])[0]))
        rp    = rad_compute(pred, K40_DEFAULT, occ_factor)
        rl_p, rc_p = risk_label(rp["ra_eq"])
        st.metric("Predicted Monazite %", f"{pred:.4f} %")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ra-eq", f"{rp['ra_eq']:.2f} Bq/kg")
        c2.metric("Gamma", f"{rp['gamma']:.6f} µSv/h")
        c3.metric("ELCR",  f"{rp['elcr']:.3e}")
        badge(rc_p, rl_p)


# ═══════════════════════════════════════════════════════════════════════════════
# ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif MODE == "ML Prediction":
    section_label("Machine Learning — Monazite % Prediction")

    with st.sidebar:
        st.divider()
        st.markdown("**Augmentation Controls**")
        n_aug    = st.slider("Synthetic samples", 90, 190, 140, 10)
        noise_sc = st.slider("Noise scale (σ)", 0.05, 0.40, 0.18, 0.01)
        aug_seed = int(st.number_input("Seed", 0, 9999, 42, step=1))

    _ml_key = (n_aug, noise_sc, aug_seed)
    if st.session_state.get("_ml_key") != _ml_key:
        with st.spinner("Training models (cached for all future interactions)…"):
            trained         = train_models(n_aug, noise_sc, aug_seed)
            loo_res, y_orig = loo_evaluate(n_aug, noise_sc, aug_seed)
            df_aug          = augment(n_aug, noise_sc, aug_seed)
            pca_data        = pca_project(n_aug, noise_sc, aug_seed)
        st.session_state._ml_key = _ml_key
    else:
        trained         = train_models(n_aug, noise_sc, aug_seed)
        loo_res, y_orig = loo_evaluate(n_aug, noise_sc, aug_seed)
        df_aug          = augment(n_aug, noise_sc, aug_seed)
        pca_data        = pca_project(n_aug, noise_sc, aug_seed)

    mt1, mt2, mt3, mt4, mt5 = st.tabs([
        "Augmented Data","Model Evaluation",
        "Feature Importance","Predict New Site","Anomaly Detection",
    ])

    with mt1:
        feat_v    = st.selectbox("Feature to inspect", MINERAL_FEATURES + ["Monazite"])
        orig_mask = df_aug["src"] == "original"
        aug_mask  = df_aug["src"] == "augmented"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.set_facecolor("none"); ax.tick_params(colors="#5f6368")
            for sp in ax.spines.values(): sp.set_edgecolor("#dadce0")
        axes[0].hist(df_aug.loc[aug_mask, feat_v], bins=25, color=ACCENT, alpha=0.55, label="Augmented")
        axes[0].hist(df_aug.loc[orig_mask, feat_v], bins=7, color=DANGER_COL, alpha=0.85,
                     label="Original", edgecolor="white")
        axes[0].set_title(f"Distribution — {feat_v}", fontsize=12, color="#202124")
        axes[0].legend(facecolor="white", edgecolor="#dadce0", labelcolor="#202124")
        axes[1].scatter(pca_data["coords"][df_aug["src"]=="augmented",0],
                        pca_data["coords"][df_aug["src"]=="augmented",1],
                        c=ACCENT, alpha=0.20, s=14, label="Augmented")
        axes[1].scatter(pca_data["coords"][df_aug["src"]=="original",0],
                        pca_data["coords"][df_aug["src"]=="original",1],
                        c=DANGER_COL, s=65, zorder=5, label="Original")
        axes[1].set_title(f"PCA Space ({pca_data['var'].sum():.1%} variance)",
                          fontsize=12, color="#202124")
        axes[1].legend(facecolor="white", edgecolor="#dadce0", labelcolor="#202124")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)
        st.caption(f"Dataset: {len(df_aug)} total ({len(irel_df)} original + {n_aug} synthetic)")

    with mt2:
        perf_df = pd.DataFrame([{"Model":n,"LOO R²":s["r2"],
                                  "LOO MAE":s["mae"],"LOO RMSE":s["rmse"]}
                                 for n, s in loo_res.items()])
        html_table(perf_df)
        st.markdown("")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.patch.set_alpha(0)
        for ax, (name, stats), c in zip(axes, loo_res.items(),
                                         [ACCENT, SUCCESS_COL, WARN_COL]):
            p = np.clip(stats["preds"], 0.0, None)
            ax.set_facecolor("none"); ax.tick_params(colors="#5f6368")
            for sp in ax.spines.values(): sp.set_edgecolor("#dadce0")
            ax.scatter(y_orig, p, color=c, s=65, zorder=5)
            for i, sn in enumerate(irel_df["id"]):
                ax.annotate(sn, (y_orig[i], p[i]), fontsize=7, color="#9aa0a6", ha="left")
            lo, hi = min(y_orig.min(),p.min()), max(y_orig.max(),p.max())
            ax.plot([lo,hi],[lo,hi], "--", color="#dadce0", lw=1.2)
            ax.set_title(f"{name}\nLOO R²={stats['r2']:.4f}", fontsize=10, color="#202124")
            ax.set_xlabel("Actual", color="#5f6368", fontsize=9)
            ax.set_ylabel("Predicted", color="#5f6368", fontsize=9)
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)
        st.info(f"LOO-CV: each fold trains on 9 originals + {n_aug} augmented samples, "
                "then tests on the held-out station.")

    with mt3:
        perm = permutation_importance(n_aug, noise_sc, aug_seed)
        fig_imp = px.bar(x=perm["features"], y=perm["importance"],
                         color=perm["importance"],
                         color_continuous_scale=[[0,"#e8f0fe"],[0.5,ACCENT],[1,"#0d47a1"]],
                         labels={"x":"Mineral Feature","y":"MAE increase on permutation"},
                         title="Permutation Importance — KNN (model-agnostic)")
        fig_imp.update_layout(**chart_layout(height=360))
        st.plotly_chart(fig_imp, use_container_width=True)

        imp_df = pd.DataFrame({
            "Feature":          MINERAL_FEATURES,
            "Random Forest":    trained["rf"].feature_importances_.round(4),
            "Gradient Boosting":trained["gb"].feature_importances_.round(4),
        }).sort_values("Random Forest", ascending=False)
        html_table(imp_df)

    with mt4:
        section_label("Predict Monazite % from Mineral Composition")
        model_choice = st.radio("Model", ["Random Forest","Gradient Boosting","KNN"],
                                horizontal=True)
        st.caption("Enter mineral percentages from XRF or optical survey.")
        ic = st.columns(3)
        new_feat = {}
        for i, feat in enumerate(MINERAL_FEATURES):
            with ic[i % 3]:
                new_feat[feat] = st.number_input(
                    feat, 0.0, 100.0, round(float(irel_df[feat].mean()),2),
                    step=0.01, key=f"ml_{feat}")
        run_ci = st.checkbox("Compute 90% bootstrap CI (500 resamples)")

        if st.button("Run Prediction"):
            X_new   = np.array([[new_feat[f] for f in MINERAL_FEATURES]])
            mdl_key = {"Random Forest":"rf","Gradient Boosting":"gb","KNN":"knn"}[model_choice]
            pred    = max(0.0, float(trained[mdl_key].predict(
                trained["scaler"].transform(X_new))[0]))
            rp      = rad_compute(pred, K40_DEFAULT, occ_factor)
            rl_p, rc_p = risk_label(rp["ra_eq"])
            stat_card("Predicted Monazite %", f"{pred:.4f} %")
            st.markdown("")
            if run_ci:
                with st.spinner("Bootstrap CI (500 resamples)…"):
                    ci_mean, ci_lo, ci_hi = bootstrap_ci(
                        tuple(new_feat[f] for f in MINERAL_FEATURES),
                        model_choice, n_aug, noise_sc, aug_seed)
                st.markdown(f"**90% CI:** [ {ci_lo:.4f}%, {ci_hi:.4f}% ] "
                            f"(bootstrap mean: {ci_mean:.4f}%)")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Ra-eq",  f"{rp['ra_eq']:.2f} Bq/kg")
            c2.metric("Gamma",  f"{rp['gamma']:.6f} µSv/h")
            c3.metric("AED",    f"{rp['aed']:.4f} µSv/yr")
            c4.metric("ELCR",   f"{rp['elcr']:.3e}")
            badge(rc_p, rl_p)
            dists = np.linalg.norm(irel_df[MINERAL_FEATURES].values - X_new, axis=1)
            nn    = irel_df.iloc[np.argmin(dists)]
            st.caption(f"Nearest reference station: {nn['station']} — "
                       f"Monazite = {nn['Monazite']:.2f}%, distance = {dists.min():.3f}")

    with mt5:
        section_label("Isolation Forest — Anomaly Detection")
        st.caption("Lower score = more anomalous relative to the training distribution.")
        cont = st.slider("Contamination estimate", 0.05, 0.40, 0.10, 0.01)
        anom = anomaly_scores(n_aug, noise_sc, aug_seed, cont)
        anom_colors = [DANGER_COL if l == "Anomaly" else SUCCESS_COL for l in anom["labels"]]
        fig_a = go.Figure(go.Bar(
            x=anom["stations"], y=anom["scores"], marker_color=anom_colors,
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>"))
        fig_a.add_hline(y=0, line_dash="dot", line_color="#dadce0")
        fig_a.update_layout(**chart_layout(
            title="Isolation Forest Decision Scores (negative = anomalous)", height=340))
        st.plotly_chart(fig_a, use_container_width=True)
        anom_df = pd.DataFrame({
            "Station":    anom["stations"],
            "Score":      np.round(anom["scores"], 4),
            "Label":      anom["labels"],
            "Monazite %": anom["monazite"],
        }).sort_values("Score")
        html_table(anom_df)


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
elif MODE == "Spatial Heatmap":
    section_label("Spatial Ra-eq Interpolation — RBF Thin-Plate Spline")
    st.caption("Grid interpolates the 10 station measurements. "
               "Not suitable for extrapolation beyond the sampled corridor.")
    res_val = st.slider("Grid resolution", 20, 80, 45, 5)
    with st.spinner("Interpolating grid…"):
        sg = spatial_grid(occ_factor, res_val)

    color_scale = [[0,"#e6f4ea"],[0.35,"#fbf8e3"],[0.55,"#fce8e6"],
                   [0.75,"#f5c6c4"],[1.00,"#8b0000"]]
    fig_h = go.Figure(go.Heatmap(
        z=sg["z"], x=sg["lon"], y=sg["lat"],
        colorscale=color_scale, zmin=0, zmax=800,
        colorbar=dict(title="Ra-eq (Bq/kg)", tickformat=".0f"),
        hovertemplate="Lat: %{y:.3f}<br>Lon: %{x:.3f}<br>Ra-eq: %{z:.1f}<extra></extra>"))
    pt_colors = [DANGER_COL if r > HBRA_THRESHOLD else ACCENT for r in fdf["ra_eq"]]
    fig_h.add_trace(go.Scatter(
        x=fdf["lon"], y=fdf["lat"], mode="markers+text", text=fdf["id"],
        textfont=dict(size=9, color="#202124"), textposition="top center",
        marker=dict(size=11, color=pt_colors, line=dict(width=2, color="white")),
        hovertext=[f"{s}: Ra-eq={r:.1f} Bq/kg"
                   for s, r in zip(fdf["station"], fdf["ra_eq"])],
        showlegend=False))
    fig_h.update_layout(**chart_layout(
        title="Radium Equivalent Spatial Distribution (RBF Thin-Plate Spline)",
        xaxis={"title":"Longitude"}, yaxis={"title":"Latitude"}, height=560))
    st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    section_label("Formula and Standards Reference")
    rf1, rf2, rf3, rf4 = st.tabs(["UNSCEAR 1988 Pipeline","ML Algorithms",
                                   "Regulatory Standards","Environmental Resources"])

    with rf1:
        st.markdown("""
| Step | Formula | Units |
|------|---------|-------|
| Th-232 activity | A_Th = (Monazite%/100) × 75000 | Bq/kg |
| U-238 activity  | A_U  = (Monazite%/100) × 4000  | Bq/kg |
| Dose rate       | D = 0.604·A_Th + 0.462·A_U + 0.0417·A_K | nGy/h |
| Gamma           | γ = (D/1000) × 0.7 | µSv/h |
| Ra-eq           | A_U + 1.43·A_Th + 0.077·A_K | Bq/kg |
| AED             | γ × 8760 × occupancy × 0.7 | µSv/yr |
| ELCR            | AED × 0.05 × 70 / 1,000,000 | — |

**South India monazite constants:** Th-232 = 75,000 Bq/kg · U-238 = 4,000 Bq/kg · K-40 = 350 Bq/kg
""")

    with rf2:
        st.markdown("""
| Algorithm | Category | Parameters | Role |
|-----------|----------|------------|------|
| Random Forest | Ensemble bagging | n_estimators=300, max_depth=5 | Primary predictor |
| Gradient Boosting | Ensemble boosting | n_estimators=200, lr=0.04 | Secondary predictor |
| KNN (k=5) | Instance-based | weights=distance | Baseline + permutation target |
| Ridge | Regularised OLS | α=0.1 | Collinearity-robust fit |
| Lasso | Sparse linear | α=0.01 | Feature selection |
| PCA | Dimensionality reduction | n_components=2 | Mineral space visualisation |
| Isolation Forest | Anomaly detection | contamination=0.1 | Outlier identification |
| RBF Interpolation | Spatial | thin-plate spline | Ra-eq heatmap |

**Augmentation:** Gaussian noise (σ = scale × feature std) + 30% Mix-up blending · Heavies = Σ heavy minerals.
""")

    with rf3:
        st.markdown("""
| Parameter | Threshold | Authority |
|-----------|-----------|-----------|
| Ra-eq Safe | ≤ 370 Bq/kg | IAEA |
| Ra-eq HBRA | 370–740 Bq/kg | IAEA |
| Ra-eq Very High | > 740 Bq/kg | IAEA |
| Annual public dose | ≤ 1 mSv/yr | IAEA |
| Annual occupational | ≤ 20 mSv/yr | IAEA |
| ELCR acceptable | < 1 × 10⁻⁵ | WHO |
| Gamma low | < 0.1 µSv/h | UNSCEAR |
| Gamma moderate | 0.1–0.5 µSv/h | UNSCEAR |
| Gamma high | 0.5–2.0 µSv/h | UNSCEAR |
| World average dose | ~59 nGy/h | UNSCEAR 2008 |
""")

    with rf4:
        st.markdown('<div class="section-label">Environmental & Scientific Resources</div>',
                    unsafe_allow_html=True)
        st.caption("Authoritative sources for natural radioactivity, environmental "
                   "monitoring, and river ecosystem research.")
        st.markdown("""
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px">
  <a class="nature-link" href="https://www.unscear.org" target="_blank">
    UNSCEAR — UN Scientific Committee on Atomic Radiation
  </a>
  <a class="nature-link" href="https://www.iaea.org/topics/radiation-protection" target="_blank">
    IAEA — Radiation Protection
  </a>
  <a class="nature-link" href="https://www.who.int/teams/environment-climate-change-and-health/radiation-and-health" target="_blank">
    WHO — Radiation & Health
  </a>
  <a class="nature-link" href="https://www.irel.co.in" target="_blank">
    IREL (India) Limited — Rare Earth Minerals
  </a>
  <a class="nature-link" href="https://www.tnforest.gov.in" target="_blank">
    Tamil Nadu Forest Department
  </a>
  <a class="nature-link" href="https://www.worldwildlife.org/places/western-ghats" target="_blank">
    WWF — Western Ghats Biodiversity
  </a>
  <a class="nature-link" href="https://india.mongabay.com/topic/western-ghats/" target="_blank">
    Mongabay India — Western Ghats
  </a>
  <a class="nature-link" href="https://www.kanyakumari.nic.in" target="_blank">
    Kanyakumari District Administration
  </a>
</div>
""", unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            "The Thamirabarani River is one of South India's few perennial rivers, "
            "originating in the Western Ghats biodiversity hotspot and flowing through "
            "the ecologically sensitive coastal plains of Kanyakumari — a region "
            "recognised for its unique confluence of the Arabian Sea, Bay of Bengal, "
            "and Indian Ocean."
        )


# ── Author credit ────────────────────────────────────────────────────────────
st.markdown('<div class="author-credit">Abinesh R</div>', unsafe_allow_html=True)

# ── Full Results Table — HTML table, always visible ─────────────────────────
with st.expander("Full Results Table", expanded=False):
    disp = fdf[["id","station","monazite","a_th","a_u",
                "dose_ngy","gamma","ra_eq","aed","elcr"]].copy()
    disp = disp.round({"monazite":3,"a_th":2,"a_u":2,"dose_ngy":3,
                       "gamma":6,"ra_eq":2,"aed":4,"elcr":6})
    disp.columns = ["ID","Station","Monazite %","A-Th (Bq/kg)","A-U (Bq/kg)",
                    "Dose (nGy/h)","Gamma (µSv/h)","Ra-eq (Bq/kg)","AED (µSv/yr)","ELCR"]
    disp["Status"] = [risk_label(r)[0] for r in fdf["ra_eq"]]
    html_table(disp)
    st.markdown("")
    st.download_button("Download CSV", disp.to_csv(index=False),
                       "thamirabarani_analysis.csv", "text/csv")