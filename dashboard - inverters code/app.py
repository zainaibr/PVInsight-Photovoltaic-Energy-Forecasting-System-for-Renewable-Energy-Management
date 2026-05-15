"""
☀️ Solar Energy Analytics Dashboard — Streamlit
================================================
Production-level PV performance dashboard.
Deploy to Streamlit Cloud → connect GitHub repo → done.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☀️ Solar PV Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  (dark theme tokens)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"
C_TEAL   = "#00d4aa"
C_CYAN   = "#56d8ff"
C_MAG    = "#ff6eb4"
C_YEL    = "#f0c040"
C_ORA    = "#ff8c42"
C_PUR    = "#c084fc"
C_GRN    = "#3fb950"
C_RED    = "#f85149"
PALETTE  = [C_TEAL, C_CYAN, C_MAG, C_YEL, C_ORA, C_PUR, C_GRN, C_RED]

def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — safe for all Plotly color props."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=SUBTEXT),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=SUBTEXT),
    legend=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font_color=TEXT),
)

def apply_layout(fig, title="", height=420):
    fig.update_layout(**LAYOUT, title=dict(text=title, font=dict(color=TEXT, size=14)),
                      height=height)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── root background ── */
.stApp { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── KPI cards ── */
.kpi-card {
    background: #161b22;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #21262d;
    text-align: center;
}
.kpi-value { font-size: 1.9rem; font-weight: 700; margin: 4px 0; }
.kpi-label { font-size: 0.82rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; }

/* ── section headers ── */
h2 { color: #e6edf3 !important; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #e6edf3 !important; }

/* ── tab bar ── */
[data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
[data-baseweb="tab"]      { color: #8b949e !important; }
[aria-selected="true"]    { color: #00d4aa !important; border-bottom: 2px solid #00d4aa; }

/* ── metric widget ── */
[data-testid="metric-container"] { background:#161b22; border-radius:8px; padding:12px 16px; border:1px solid #21262d; }
[data-testid="stMetricValue"]    { color: #e6edf3; }
[data-testid="stMetricLabel"]    { color: #8b949e; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }

/* make plotly chart backgrounds match */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA  (load CSV or auto-generate synthetic)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading dataset…")
def load_data(uploaded=None):
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        if not raw.empty:
            df = raw.copy()
            df.columns = df.columns.str.strip()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            num_cols = df.select_dtypes("object").columns.difference(["source_file"])
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
            _add_time_features(df)
            return df, False

    df_syn = _generate_synthetic()
    return df_syn, True


def _generate_synthetic() -> tuple:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=8760, freq="h")
    n   = len(idx)
    hour = idx.hour.values
    doy  = idx.dayofyear.values

    lat  = np.radians(31.95)
    decl = np.radians(23.45) * np.sin(np.radians(360 / 365 * (doy - 81)))
    ha   = np.radians(15 * (hour - 12))
    cos_z = np.clip(np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha), 0, 1)
    G_clear = 0.75 * 1361.0 * cos_z

    cloud    = np.repeat(rng.beta(2, 5, n // 24 + 1)[: n // 24], 24)[:n]
    irr_base = G_clear * (0.3 + 0.7 * (1 - cloud))

    sensors = {}
    for i in range(1, 7):
        off  = rng.uniform(-15, 15)
        noise = rng.normal(0, irr_base * 0.02 + 1, n)
        sensors[f"irr_sensor_{i}"] = np.clip(irr_base + off + noise, 0, 1300)

    irr_avg    = np.mean([sensors[k] for k in sensors], axis=0)
    irr_tilted = np.clip(irr_avg * 1.08 * rng.normal(1, 0.01, n), 0, 1350)

    T_amb = (10 + 10 * np.sin(np.radians(360 / 365 * (doy - 15)))
             + 6 * np.sin(np.radians(360 / 24 * (hour - 14)))
             + rng.normal(0, 2, n))
    T_mod = T_amb + 0.03 * irr_avg + rng.normal(0, 1, n)

    eta_stc, temp_coeff = 0.185, -0.0045
    irr_kw   = irr_avg / 1000.0
    eta      = eta_stc * (1 + temp_coeff * (T_mod - 25))
    power_kw = np.clip(100.0 * irr_kw * eta, 0, None)

    yield_arr, daily_cum = np.zeros(n), 0.0
    for i in range(n):
        if i > 0 and idx[i].date() != idx[i - 1].date():
            daily_cum = 0.0
        daily_cum    += power_kw[i]
        yield_arr[i]  = daily_cum

    pred_irr  = irr_avg * rng.normal(1.0, 0.05, n)
    pred_temp = T_mod   + rng.normal(0, 1.5, n)

    df = pd.DataFrame({
        **sensors,
        "irradiation_tilted":         irr_tilted,
        "power_analyzer":             np.clip(power_kw * 1000, 0, None),
        "generated_yield":            yield_arr,
        "avg_module_temp":            T_mod,
        "irradiance_avg":             irr_avg,
        "source_file":                "synthetic_2022",
        "actual_irradiance":          irr_avg,
        "predicted_irradiance":       pred_irr,
        "actual_temperature":         T_mod,
        "predicted_temperature":      pred_temp,
        "power_kw":                   power_kw,
        "irradiance_kw_m2":           irr_kw,
        "efficiency":                 np.clip(eta * 100, 0, 25),
        "irradiance_error":           irr_avg - pred_irr,
        "irradiance_abs_error":       np.abs(irr_avg - pred_irr),
        "temp_error":                 T_mod - pred_temp,
        "temp_abs_error":             np.abs(T_mod - pred_temp),
        "predicted_irradiance_kw_m2": pred_irr / 1000.0,
        "Predicted_Energy_Yield":     np.clip(100 * (pred_irr / 1000) * eta_stc, 0, None),
    }, index=idx)

    _add_time_features(df)
    return df


def _add_time_features(df):
    df["hour"]      = df.index.hour
    df["month"]     = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["doy"]       = df.index.dayofyear
    df["season"]    = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Autumn",  10: "Autumn", 11: "Autumn",
    })


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar PV Dashboard")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload your CSV",
        type=["csv"],
        help="Must match the schema: datetime, irr_sensor_1…6, power_kw, etc.",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Filters")

result = load_data(uploaded_file)
df_raw, is_synthetic = result

with st.sidebar:
    yr_min = df_raw.index.year.min()
    yr_max = df_raw.index.year.max()

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    sel_months = st.multiselect(
        "Months", options=list(range(1, 13)),
        format_func=lambda m: month_names[m - 1],
        default=list(range(1, 13)),
    )
    sel_hours = st.slider("Hours of day", 0, 23, (0, 23))
    irr_min   = st.slider(
        "Min irradiance filter (W/m²)", 0, 200, 0,
        help="Filter out low-irradiance / night records",
    )

    st.markdown("---")
    st.markdown("### 📊 Sample size")
    scatter_n = st.slider("Scatter plot sample", 500, 5000, 2000, step=500)

    st.markdown("---")
    if is_synthetic:
        st.info("ℹ️ Using **synthetic** 2022 dataset (CSV was empty). Upload your own CSV above.")
    else:
        st.success("✅ Using **uploaded** dataset.")

# ─── apply filters ────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_months:
    df = df[df["month"].isin(sel_months)]
df = df[(df["hour"] >= sel_hours[0]) & (df["hour"] <= sel_hours[1])]
if irr_min > 0:
    df = df[df["irradiance_avg"] >= irr_min]

df_day = df[df["irradiance_avg"] > 20].copy()
df_day["sensor_std"]      = df_day[[c for c in df_day.columns if "irr_sensor" in c]].std(axis=1)
df_day["clearness_index"] = df_day["irradiance_avg"] / df_raw["irradiance_avg"].max()

if df.empty:
    st.error("No data after filtering — please adjust the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='padding:18px 0 4px 0'>
      <h1 style='color:#e6edf3;margin:0'>☀️ Solar PV Analytics Dashboard</h1>
      <p style='color:#8b949e;margin:4px 0 0 0'>
        2022 · 100 kW Rooftop System · Amman, Jordan (31.95°N) ·
        <b style='color:#00d4aa'>{len(df):,}</b> records selected
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
total_energy   = df["power_kw"].sum()
peak_power     = df["power_kw"].max()
avg_eff        = df_day["efficiency"].mean() if not df_day.empty else 0
cap_factor     = df["power_kw"].mean() / 100.0 * 100
irr_mae        = df_day["irradiance_abs_error"].mean() if not df_day.empty else 0
temp_mae       = df_day["temp_abs_error"].mean() if not df_day.empty else 0
avg_irr        = df["irradiance_avg"].mean()
avg_temp       = df["avg_module_temp"].mean()

kpi_data = [
    ("Total Energy Yield",    f"{total_energy:,.0f} kWh",  C_TEAL),
    ("Peak Power Output",     f"{peak_power:.1f} kW",      C_CYAN),
    ("Avg Module Efficiency", f"{avg_eff:.2f} %",          C_YEL),
    ("Capacity Factor",       f"{cap_factor:.1f} %",       C_ORA),
    ("Avg Irradiance",        f"{avg_irr:.0f} W/m²",       C_PUR),
    ("Avg Module Temp",       f"{avg_temp:.1f} °C",        C_MAG),
    ("Irradiance MAE",        f"{irr_mae:.1f} W/m²",       C_GRN),
    ("Temperature MAE",       f"{temp_mae:.2f} °C",        C_RED),
]

cols_kpi = st.columns(8)
for col, (label, value, color) in zip(cols_kpi, kpi_data):
    with col:
        st.markdown(
            f"""<div class='kpi-card' style='border-top:3px solid {color}'>
                  <div class='kpi-value' style='color:{color}'>{value}</div>
                  <div class='kpi-label'>{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Time Series",
    "📊 Distributions",
    "🔥 Heatmaps",
    "🔗 Correlations",
    "🎯 Model Analysis",
    "🔬 Advanced",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📈 Time-Series & Trend Analysis")

    daily = df.resample("D").agg(
        power_total=("power_kw", "sum"),
        irradiance_mean=("irradiance_avg", "mean"),
        temp_mean=("avg_module_temp", "mean"),
        efficiency_mean=("efficiency", "mean"),
    ).dropna()

    # ── Daily energy + rolling averages ───────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"],
        fill="tozeroy", mode="lines",
        line=dict(color=C_CYAN, width=0.8),
        fillcolor="rgba(86,216,255,0.12)",
        name="Daily kWh",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"].rolling(7).mean(),
        mode="lines", line=dict(color=C_TEAL, width=2),
        name="7-day MA",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["power_total"].rolling(30).mean(),
        mode="lines", line=dict(color=C_YEL, width=2.5),
        name="30-day MA",
    ))
    # peak annotation
    if len(daily) > 0:
        pk = daily["power_total"].idxmax()
        fig.add_annotation(
            x=pk, y=daily.loc[pk, "power_total"],
            text=f"⚡ Peak<br>{pk.strftime('%b %d')}",
            showarrow=True, arrowhead=2, arrowcolor=C_YEL,
            font=dict(color=C_YEL, size=11),
            bgcolor=PANEL, bordercolor=C_YEL,
        )
    apply_layout(fig, "Daily Energy Output with Rolling Averages (kWh/day)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly bars + efficiency trend ───────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        monthly_pwr = df.resample("ME")["power_kw"].sum()
        monthly_irr = df.groupby("month")["irradiance_avg"].mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[month_names[m - 1] for m in monthly_irr.index],
            y=monthly_irr.values,
            marker_color=PALETTE[:12], opacity=0.85,
            name="Avg Irradiance (W/m²)",
        ))
        fig2.add_trace(go.Scatter(
            x=[month_names[m - 1] for m in monthly_irr.index],
            y=monthly_irr.values,
            mode="lines+markers",
            line=dict(color=C_YEL, width=2),
            marker=dict(size=7, color=C_YEL),
            name="Trend", yaxis="y",
        ))
        apply_layout(fig2, "Monthly Average Irradiance (W/m²)", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        eff_roll30 = daily["efficiency_mean"].rolling(30).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=daily.index, y=daily["efficiency_mean"],
            mode="lines", line=dict(color=C_PUR, width=0.7),
            opacity=0.4, name="Daily Efficiency",
        ))
        fig3.add_trace(go.Scatter(
            x=daily.index, y=eff_roll30,
            mode="lines", line=dict(color=C_MAG, width=2.5),
            name="30-day MA",
            fill="tonexty", fillcolor="rgba(255,110,180,0.08)",
        ))
        apply_layout(fig3, "Module Efficiency Trend (%)", height=340)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Rolling volatility ─────────────────────────────────────────────────
    vol_7d = df["power_kw"].rolling(24 * 7).std()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=vol_7d.index, y=vol_7d.values,
        mode="lines", line=dict(color=C_ORA, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,140,66,0.15)",
        name="7-day rolling σ",
    ))
    apply_layout(fig4, "Power Output — 7-Day Rolling Volatility (σ kW)", height=280)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Feature Distributions")

    dist_cols = [
        "irradiance_avg", "power_kw", "avg_module_temp", "efficiency",
        "generated_yield", "irradiation_tilted", "irradiance_abs_error", "temp_abs_error",
    ]

    c1, c2 = st.columns(2)
    col_toggle = [c1, c2, c1, c2, c1, c2, c1, c2]

    for col_container, col_name in zip(col_toggle, dist_cols):
        with col_container:
            data_src = df_day[col_name].dropna() if col_name in df_day.columns else df[col_name].dropna()
            if len(data_src) < 10:
                continue
            color = PALETTE[dist_cols.index(col_name) % len(PALETTE)]
            sk, ku = data_src.skew(), data_src.kurt()

            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(
                x=data_src, nbinsx=60,
                marker_color=color, opacity=0.35,
                histnorm="probability density", name="Histogram",
                showlegend=False,
            ))
            # KDE
            kde = gaussian_kde(data_src)
            xs  = np.linspace(data_src.min(), data_src.max(), 300)
            fig_d.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines",
                line=dict(color=color, width=2.5), name="KDE",
            ))
            fig_d.add_vline(x=data_src.mean(),   line=dict(color=C_YEL, width=1.5, dash="dash"),
                            annotation_text=f"μ={data_src.mean():.1f}",
                            annotation_font_color=C_YEL)
            fig_d.add_vline(x=data_src.median(), line=dict(color=C_MAG, width=1.5, dash="dot"),
                            annotation_text=f"M={data_src.median():.1f}",
                            annotation_font_color=C_MAG)

            title_d = f"{col_name.replace('_',' ').title()} — skew={sk:.2f} | kurt={ku:.2f}"
            apply_layout(fig_d, title_d, height=280)
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Boxplot panel ──────────────────────────────────────────────────────
    st.markdown("### Outlier Detection — Boxplots")
    fig_box = go.Figure()
    for i, col_name in enumerate(dist_cols):
        data_src = df_day[col_name].dropna() if col_name in df_day.columns else df[col_name].dropna()
        fig_box.add_trace(go.Box(
            y=data_src,
            name=col_name.replace("_", " "),
            marker_color=PALETTE[i % len(PALETTE)],
            line_color=PALETTE[i % len(PALETTE)],
            fillcolor="rgba(0,0,0,0)",
            boxmean=True,
        ))
    apply_layout(fig_box, "IQR Boxplots — All Key Variables", height=400)
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔥 Seasonal Heatmaps — Production Fingerprint")
    st.caption("Hour-of-day × Month: reveals *when* energy is produced across the year.")

    hm_metric = st.radio(
        "Select metric",
        ["irradiance_avg", "power_kw", "efficiency", "avg_module_temp"],
        format_func=lambda c: c.replace("_", " ").title(),
        horizontal=True,
    )

    piv = df.pivot_table(values=hm_metric, index="hour", columns="month", aggfunc="mean")
    piv.columns = [month_names[m - 1] for m in piv.columns]

    color_scales = {
        "irradiance_avg": "plasma",
        "power_kw":       "viridis",
        "efficiency":     "magma",
        "avg_module_temp":"RdYlBu_r",
    }

    fig_hm = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns.tolist(),
        y=piv.index.tolist(),
        colorscale=color_scales.get(hm_metric, "plasma"),
        colorbar=dict(title=hm_metric.replace("_", " "), tickfont=dict(color=TEXT)),
        hoverongaps=False,
        hovertemplate="Month: %{x}<br>Hour: %{y}<br>Value: %{z:.2f}<extra></extra>",
    ))
    apply_layout(fig_hm,
                 f"Hour × Month Heatmap — {hm_metric.replace('_',' ').title()}",
                 height=520)
    fig_hm.update_layout(
        yaxis_title="Hour of Day", xaxis_title="Month",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Weekday × Month ────────────────────────────────────────────────────
    st.markdown("### Weekday × Month — Average Power (kW)")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    piv_wd = df.pivot_table(values="power_kw", index="dayofweek", columns="month", aggfunc="mean")
    piv_wd.index   = [day_names[i] for i in piv_wd.index]
    piv_wd.columns = [month_names[m - 1] for m in piv_wd.columns]

    fig_wd = go.Figure(go.Heatmap(
        z=piv_wd.values, x=piv_wd.columns.tolist(), y=piv_wd.index.tolist(),
        colorscale="teal",
        colorbar=dict(title="kW", tickfont=dict(color=TEXT)),
        hovertemplate="Month: %{x}<br>Day: %{y}<br>Avg Power: %{z:.2f} kW<extra></extra>",
    ))
    apply_layout(fig_wd, "Weekday × Month — Avg Power (kW)", height=380)
    st.plotly_chart(fig_wd, use_container_width=True)

    # ── Sensor bias heatmap ────────────────────────────────────────────────
    st.markdown("### Sensor Bias Matrix (Hour × Sensor)")
    sensor_cols = [c for c in df_day.columns if c.startswith("irr_sensor")]
    if sensor_cols:
        bias_data = {c: df_day[c] - df_day["irradiance_avg"] for c in sensor_cols}
        bias_df   = pd.DataFrame(bias_data)
        bias_df["hour"] = df_day["hour"]
        piv_bias = bias_df.groupby("hour")[sensor_cols].mean()

        fig_sb = go.Figure(go.Heatmap(
            z=piv_bias.values,
            x=[c.replace("irr_","") for c in sensor_cols],
            y=piv_bias.index.tolist(),
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Bias (W/m²)", tickfont=dict(color=TEXT)),
            hovertemplate="Sensor: %{x}<br>Hour: %{y}<br>Bias: %{z:.2f} W/m²<extra></extra>",
        ))
        apply_layout(fig_sb, "Hourly Sensor Bias vs Array Average (W/m²)", height=380)
        st.plotly_chart(fig_sb, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔗 Correlation & Relationship Analysis")

    corr_cols = [
        "irradiance_avg", "irradiation_tilted", "power_kw", "generated_yield",
        "avg_module_temp", "efficiency", "irradiance_kw_m2",
        "actual_irradiance", "predicted_irradiance",
        "irradiance_abs_error", "temp_abs_error", "Predicted_Energy_Yield",
    ]
    available = [c for c in corr_cols if c in df_day.columns]
    corr = df_day[available].corr()

    # ── Heatmap ────────────────────────────────────────────────────────────
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    z_masked = corr.values.copy()
    z_masked[mask] = None

    fig_corr = go.Figure(go.Heatmap(
        z=z_masked,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, C_PUR], [0.5, PANEL], [1, C_TEAL]],
        zmid=0, zmin=-1, zmax=1,
        text=np.where(~mask, np.round(corr.values, 2).astype(str), ""),
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT),
        colorbar=dict(title="Pearson r", tickfont=dict(color=TEXT)),
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    apply_layout(fig_corr, "Pearson Correlation Matrix — Daytime Records", height=560)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Scatter deep-dive ──────────────────────────────────────────────────
    st.markdown("### Interactive Scatter Deep-Dive")
    num_cols_list = [c for c in df_day.select_dtypes(include=np.number).columns
                     if c not in ["hour","month","dayofweek","doy"]]

    sc1, sc2, sc3 = st.columns(3)
    x_col = sc1.selectbox("X axis", num_cols_list, index=num_cols_list.index("irradiance_avg") if "irradiance_avg" in num_cols_list else 0)
    y_col = sc2.selectbox("Y axis", num_cols_list, index=num_cols_list.index("power_kw") if "power_kw" in num_cols_list else 1)
    c_col = sc3.selectbox("Colour by", ["season","month","hour"] + num_cols_list, index=0)

    samp = df_day[[x_col, y_col, c_col]].dropna().sample(min(scatter_n, len(df_day)), random_state=0)

    if c_col in ["season", "month", "hour"]:
        season_cmap = {"Winter": C_CYAN, "Spring": C_GRN, "Summer": C_YEL, "Autumn": C_ORA}
        if c_col == "season":
            colors_sc = samp["season"].map(season_cmap)
        else:
            colors_sc = px.colors.sample_colorscale("plasma",
                            (samp[c_col] - samp[c_col].min()) /
                            (samp[c_col].max() - samp[c_col].min() + 1e-9))
        fig_sc = go.Figure(go.Scatter(
            x=samp[x_col], y=samp[y_col], mode="markers",
            marker=dict(color=colors_sc if isinstance(colors_sc, list) else colors_sc.tolist(),
                        size=5, opacity=0.55),
            text=samp[c_col].astype(str), hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.2f}}<br>{c_col}=%{{text}}<extra></extra>",
        ))
    else:
        fig_sc = go.Figure(go.Scatter(
            x=samp[x_col], y=samp[y_col], mode="markers",
            marker=dict(color=samp[c_col], colorscale="plasma", size=5, opacity=0.55,
                        colorbar=dict(title=c_col, tickfont=dict(color=TEXT))),
            hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.2f}}<extra></extra>",
        ))

    # regression line
    valid_sc = df_day[[x_col, y_col]].dropna()
    if len(valid_sc) > 10:
        m_sc, b_sc, r_sc, *_ = stats.linregress(valid_sc[x_col], valid_sc[y_col])
        xs_sc = np.linspace(valid_sc[x_col].min(), valid_sc[x_col].max(), 100)
        fig_sc.add_trace(go.Scatter(
            x=xs_sc, y=m_sc*xs_sc+b_sc, mode="lines",
            line=dict(color=C_RED, width=2.5, dash="dash"),
            name=f"r = {r_sc:.3f}",
        ))

    apply_layout(fig_sc, f"{y_col} vs {x_col} (n={len(samp):,})", height=440)
    fig_sc.update_xaxes(title=x_col.replace("_", " "))
    fig_sc.update_yaxes(title=y_col.replace("_", " "))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Top correlations bar ───────────────────────────────────────────────
    if "power_kw" in corr.columns:
        st.markdown("### Top Feature Correlations with **power_kw**")
        top_corr = corr["power_kw"].drop("power_kw").abs().sort_values(ascending=True)
        colors_bar = [C_RED if corr.loc[f, "power_kw"] < 0 else C_TEAL for f in top_corr.index]
        fig_bar = go.Figure(go.Bar(
            x=corr.loc[top_corr.index, "power_kw"].values,
            y=top_corr.index.tolist(),
            orientation="h",
            marker_color=colors_bar, opacity=0.85,
            text=[f"{v:.3f}" for v in corr.loc[top_corr.index, "power_kw"].values],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        apply_layout(fig_bar, "Pearson r with power_kw", height=380)
        fig_bar.update_xaxes(title="Pearson r")
        st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🎯 Prediction Model Analysis")

    if "actual_irradiance" not in df_day.columns or df_day.empty:
        st.warning("Prediction columns not available in this dataset.")
    else:
        c1, c2 = st.columns(2)

        # Actual vs Predicted Irradiance
        with c1:
            samp_p = df_day[["actual_irradiance","predicted_irradiance"]].dropna().sample(
                min(scatter_n, len(df_day)), random_state=1)
            r2_irr = np.corrcoef(samp_p["actual_irradiance"], samp_p["predicted_irradiance"])[0,1]**2
            mn_p, mx_p = samp_p.min().min(), samp_p.max().max()

            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=samp_p["actual_irradiance"], y=samp_p["predicted_irradiance"],
                mode="markers", marker=dict(color=C_TEAL, size=4, opacity=0.3),
                name="Samples",
            ))
            fig_avp.add_trace(go.Scatter(
                x=[mn_p, mx_p], y=[mn_p, mx_p],
                mode="lines", line=dict(color=C_YEL, width=2, dash="dash"),
                name="1:1 line",
            ))
            apply_layout(fig_avp, f"Irradiance: Actual vs Predicted  |  R²={r2_irr:.4f}", height=380)
            fig_avp.update_xaxes(title="Actual (W/m²)")
            fig_avp.update_yaxes(title="Predicted (W/m²)")
            st.plotly_chart(fig_avp, use_container_width=True)

        # Residuals Distribution
        with c2:
            resid = df_day["irradiance_error"].dropna()
            xs_n  = np.linspace(resid.min(), resid.max(), 300)
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=resid, nbinsx=80, histnorm="probability density",
                marker_color=C_CYAN, opacity=0.4, name="Residuals",
            ))
            fig_res.add_trace(go.Scatter(
                x=xs_n, y=stats.norm.pdf(xs_n, resid.mean(), resid.std()),
                mode="lines", line=dict(color=C_MAG, width=2.5), name="Normal fit",
            ))
            fig_res.add_vline(x=0, line=dict(color=C_YEL, width=1.5, dash="dash"))
            apply_layout(fig_res,
                         f"Irradiance Residuals  |  μ={resid.mean():.2f}  σ={resid.std():.2f}",
                         height=380)
            fig_res.update_xaxes(title="Error (W/m²)")
            fig_res.update_yaxes(title="Density")
            st.plotly_chart(fig_res, use_container_width=True)

        # Daily MAE over time
        daily_mae = df_day["irradiance_abs_error"].resample("D").mean()
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=daily_mae.index, y=daily_mae.values,
            mode="lines", line=dict(color=C_RED, width=0.7),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)", name="Daily MAE",
        ))
        fig_mae.add_trace(go.Scatter(
            x=daily_mae.index, y=daily_mae.rolling(14).mean(),
            mode="lines", line=dict(color=C_ORA, width=2.5), name="14-day MA",
        ))
        apply_layout(fig_mae, "Daily Mean Absolute Irradiance Error", height=300)
        fig_mae.update_xaxes(title="Date")
        fig_mae.update_yaxes(title="MAE (W/m²)")
        st.plotly_chart(fig_mae, use_container_width=True)

        # Q-Q Plot
        c3, c4 = st.columns(2)
        with c3:
            qq_sample = resid.sample(min(2000, len(resid)), random_state=4)
            (osm, osr), (slope_q, intercept_q, r_q) = stats.probplot(qq_sample)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=list(osm), y=list(osr), mode="markers",
                marker=dict(color=C_CYAN, size=5, opacity=0.4), name="Residuals",
            ))
            fig_qq.add_trace(go.Scatter(
                x=list(osm), y=[slope_q*v + intercept_q for v in osm],
                mode="lines", line=dict(color=C_YEL, width=2.5, dash="dash"),
                name="Normal ref.",
            ))
            apply_layout(fig_qq, f"Q-Q Plot  |  r²={r_q**2:.4f}", height=340)
            st.plotly_chart(fig_qq, use_container_width=True)

        with c4:
            # Temperature error scatter
            samp_te = df_day[["actual_temperature","temp_error"]].dropna().sample(
                min(scatter_n, len(df_day)), random_state=5)
            fig_te = go.Figure(go.Scatter(
                x=samp_te["actual_temperature"], y=samp_te["temp_error"],
                mode="markers",
                marker=dict(color=C_PUR, size=4, opacity=0.25),
                name="Temp residuals",
            ))
            fig_te.add_hline(y=0, line=dict(color=C_YEL, width=1.5, dash="dash"))
            apply_layout(fig_te, "Temperature Residuals vs Actual Temp", height=340)
            fig_te.update_xaxes(title="Actual Temp (°C)")
            fig_te.update_yaxes(title="Error (°C)")
            st.plotly_chart(fig_te, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ADVANCED
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🔬 Advanced Analysis")

    # ── ACF ────────────────────────────────────────────────────────────────
    st.markdown("### Autocorrelation (ACF) — power_kw")
    hourly_pwr = df["power_kw"].dropna()
    max_lag = 72
    acf_vals = [hourly_pwr.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    ci_acf   = 1.96 / np.sqrt(len(hourly_pwr))

    fig_acf = go.Figure()
    colors_acf = [C_TEAL if v >= 0 else C_RED for v in acf_vals]
    for lag, val, color in zip(range(1, max_lag + 1), acf_vals, colors_acf):
        fig_acf.add_shape(type="line",
                          x0=lag, y0=0, x1=lag, y1=val,
                          line=dict(color=color, width=3))
    fig_acf.add_trace(go.Scatter(
        x=list(range(1, max_lag + 1)), y=acf_vals,
        mode="markers", marker=dict(color=colors_acf, size=6),
        showlegend=False,
    ))
    fig_acf.add_hline(y=ci_acf,  line=dict(color=C_YEL, width=1.5, dash="dash"), annotation_text="95% CI")
    fig_acf.add_hline(y=-ci_acf, line=dict(color=C_YEL, width=1.5, dash="dash"))
    fig_acf.add_hline(y=0, line=dict(color=SUBTEXT, width=0.8))
    apply_layout(fig_acf, "Autocorrelation Function (ACF) — power_kw | 72h", height=340)
    fig_acf.update_xaxes(title="Lag (hours)")
    fig_acf.update_yaxes(title="ACF")
    st.plotly_chart(fig_acf, use_container_width=True)

    # ── Lag-24 scatter ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        lag_df = pd.DataFrame({
            "t":    hourly_pwr.values[24:],
            "t_24": hourly_pwr.values[:-24],
        }).dropna()
        m_lg, b_lg, r_lg, *_ = stats.linregress(lag_df["t_24"], lag_df["t"])
        xs_lg = np.linspace(lag_df.min().min(), lag_df.max().max(), 100)
        samp_lag = lag_df.sample(min(3000, len(lag_df)), random_state=0)

        fig_lag = go.Figure()
        fig_lag.add_trace(go.Scatter(
            x=samp_lag["t_24"], y=samp_lag["t"],
            mode="markers", marker=dict(color=C_ORA, size=4, opacity=0.15),
            name="Samples",
        ))
        fig_lag.add_trace(go.Scatter(
            x=xs_lg, y=m_lg * xs_lg + b_lg,
            mode="lines", line=dict(color=C_RED, width=2.5, dash="dash"),
            name=f"r = {r_lg:.3f}",
        ))
        apply_layout(fig_lag, "Lag-24 Plot: power(t) vs power(t-24h)", height=360)
        fig_lag.update_xaxes(title="power at t-24 (kW)")
        fig_lag.update_yaxes(title="power at t (kW)")
        st.plotly_chart(fig_lag, use_container_width=True)

    # ── Sensor violin ──────────────────────────────────────────────────────
    with c2:
        sensor_cols = [c for c in df_day.columns if c.startswith("irr_sensor")]
        if sensor_cols:
            fig_vio = go.Figure()
            for i, sc in enumerate(sensor_cols):
                fig_vio.add_trace(go.Violin(
                    y=df_day[sc].dropna(),
                    name=sc.replace("irr_", ""),
                    box_visible=True, meanline_visible=True,
                    fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.35),
                    line_color=PALETTE[i % len(PALETTE)],
                ))
            apply_layout(fig_vio, "Sensor Distribution Comparison (Violin)", height=360)
            fig_vio.update_yaxes(title="W/m²")
            st.plotly_chart(fig_vio, use_container_width=True)

    # ── Feature importance (lightweight RF) ────────────────────────────────
    st.markdown("### Feature Importance — Random Forest Proxy")

    @st.cache_data(show_spinner="🌲 Training RF…", max_entries=3)
    def compute_importance(data_hash):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        feat_cols = ["irradiance_avg","irradiation_tilted","avg_module_temp",
                     "efficiency","irradiance_kw_m2","irradiance_abs_error",
                     "temp_abs_error","hour","month","doy"]
        feat_cols = [c for c in feat_cols if c in df_day.columns]
        ml_df = df_day[feat_cols + ["power_kw"]].dropna()
        if len(ml_df) < 50:
            return None, feat_cols
        X = ml_df[feat_cols].values
        y = ml_df["power_kw"].values
        Xs = StandardScaler().fit_transform(X)
        rf = RandomForestRegressor(n_estimators=60, max_depth=6, n_jobs=-1, random_state=42)
        rf.fit(Xs, y)
        return rf.feature_importances_, feat_cols

    imps, feat_cols = compute_importance(str(len(df_day)))

    if imps is not None:
        order_fi = np.argsort(imps)
        fig_fi = go.Figure(go.Bar(
            x=[imps[i] for i in order_fi],
            y=[feat_cols[i].replace("_", " ") for i in order_fi],
            orientation="h",
            marker=dict(
                color=[PALETTE[i % len(PALETTE)] for i in order_fi],
                opacity=0.85,
            ),
            text=[f"{imps[i]:.3f}" for i in order_fi],
            textposition="outside",
            textfont=dict(color=TEXT),
        ))
        apply_layout(fig_fi, "RF Feature Importance — power_kw target", height=400)
        fig_fi.update_xaxes(title="Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Raw Data Preview")
    st.dataframe(
        df.select_dtypes(include=np.number).tail(200).style
          .background_gradient(cmap="YlOrRd", axis=0),
        use_container_width=True,
        height=300,
    )

    # ── Download ───────────────────────────────────────────────────────────
    csv_download = df.reset_index().to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=csv_download,
        file_name="solar_filtered.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.8rem'>"
    "☀️ Solar PV Analytics Dashboard · Built with Streamlit · "
    "Dark theme · Plotly interactive charts</p>",
    unsafe_allow_html=True,
)
