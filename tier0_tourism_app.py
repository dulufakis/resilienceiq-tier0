"""
ResilienceIQ Tier 0 — Realtime Tourism Resilience Dashboard
Live data from: Eurostat (arrivals/nights), Google Trends (demand), Open-Meteo (weather).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data_fetcher import (
    MUNICIPALITIES, WEIGHTS_FALLBACK, COMPONENT_NAMES, PILLARS, PILLAR_NAMES,
    fetch_eurostat_tourism, fetch_google_trends, fetch_weather,
    build_resilience_snapshot,
)

# ─── Page Config ─────────────────────────────────
st.set_page_config(
    page_title="ResilienceIQ Tier 0",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E50B4 0%, #1a3d8f 100%);
        color: white; padding: 18px 24px; border-radius: 10px;
        margin-bottom: 18px; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 28px; }
    .main-header p { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }
    .kpi-card {
        background: #ffffff; border-radius: 12px; padding: 16px;
        border-top: 4px solid #1E50B4; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center; margin-bottom: 8px;
    }
    .kpi-card .value { font-size: 32px; font-weight: 700; }
    .kpi-card .label { font-size: 13px; color: #666; margin-top: 4px; }
    .zone-high { color: #27AE60; }
    .zone-mid  { color: #E67E22; }
    .zone-low  { color: #C0392B; }
    .insight-box {
        background: #f0f4ff; border-left: 5px solid #1E50B4;
        padding: 16px; border-radius: 8px; font-size: 15px; margin: 12px 0;
    }
    .data-source {
        background: #f8f9fa; border-radius: 6px; padding: 8px 12px;
        font-size: 12px; color: #666; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────
with st.sidebar:
    st.markdown("### ResilienceIQ")
    st.caption("Tier 0 — Live Tourism Resilience")
    st.divider()

    page = st.radio("Navigation", [
        "Dashboard",
        "Municipality Deep-Dive",
        "Trends & Time Series",
        "Methodology",
    ], label_visibility="collapsed")

    st.divider()
    st.subheader("What-if Simulator")
    shock_demand = st.slider("Demand Shock (%)", -80, 80, 0, help="Simulate change in tourism demand")
    shock_arrivals = st.slider("Arrivals Shock (%)", -80, 80, 0, help="Simulate change in tourist arrivals")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    if auto_refresh:
        st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─── Fetch Real Data (cached) ───────────────────
@st.cache_data(ttl=1800, show_spinner="Fetching Eurostat data...")
def _fetch_eurostat():
    return fetch_eurostat_tourism()

@st.cache_data(ttl=3600, show_spinner="Fetching Google Trends...")
def _fetch_trends():
    return fetch_google_trends()

@st.cache_data(ttl=900, show_spinner="Fetching live weather...")
def _fetch_weather():
    return fetch_weather()

df_euro = _fetch_eurostat()
df_trends = _fetch_trends()
df_weather = _fetch_weather()

# Global computed values
euro_year = df_euro["year"].max() if not df_euro.empty else "N/A"

# Build snapshot (not cached — reacts to shock sliders)
df = build_resilience_snapshot(
    df_euro, df_trends, df_weather,
    shock_demand=shock_demand,
    shock_arrivals=shock_arrivals,
)

# Extract diagnostics from attrs, then clear to avoid pandas serialization issues
weights = df.attrs.get("weights", WEIGHTS_FALLBACK)
pillar_result = df.attrs.get("pillar_result")
weighting_method = df.attrs.get("weighting_method", "Static")
df.attrs.clear()


# =================================================================
# PAGE 1: DASHBOARD
# =================================================================
if page == "Dashboard":
    st.markdown("""
    <div class='main-header'>
        <h1>Resilience Control Center</h1>
        <p>Live tourism resilience monitoring | Eurostat + Google Trends + Open-Meteo</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──
    avg_score = df["resilience_score"].mean()
    top = df.iloc[0]
    bottom = df.iloc[-1]
    high_n = (df["zone"] == "ΥΨΗΛΗ").sum()
    low_n = (df["zone"] == "ΧΑΜΗΛΗ").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Score", f"{avg_score:.1f}")
    c2.metric("Top Municipality", f"{top['dimos']}", f"{top['resilience_score']:.1f}")
    c3.metric("Bottom Municipality", f"{bottom['dimos']}", f"{bottom['resilience_score']:.1f}")
    c4.metric("High Zone", f"{high_n} / {len(df)}")
    c5.metric("Eurostat Year", f"{euro_year}")

    st.divider()

    # ── Map + Ranking ──
    col_map, col_bar = st.columns([3, 2])

    with col_map:
        st.subheader("Geographic Risk Map")
        fig_map = px.scatter_mapbox(
            df, lat="lat", lon="lon",
            size=df["resilience_score"].fillna(0).clip(lower=15),
            color="zone",
            color_discrete_map={"ΥΨΗΛΗ": "#27AE60", "ΜΕΣΑΙΑ": "#E67E22", "ΧΑΜΗΛΗ": "#C0392B"},
            hover_name="dimos",
            hover_data={
                "resilience_score": ":.1f",
                "region": True,
                "arrivals": ":,.0f",
                "lat": False, "lon": False, "zone": False,
            },
            zoom=5.5, center={"lat": 38.0, "lon": 24.0},
            mapbox_style="carto-positron",
            height=480,
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_bar:
        st.subheader("Resilience Ranking")
        fig_bar = go.Figure(go.Bar(
            x=df["resilience_score"], y=df["dimos"],
            orientation="h",
            marker_color=df["color"],
            text=df["resilience_score"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=480, margin=dict(l=0, r=50, t=0, b=0),
            xaxis=dict(range=[0, 105], title="Resilience Score"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Data Table ──
    st.subheader("Full Data Table (Live)")
    display_cols = ["dimos", "region", "arrivals", "nights", "avg_stay",
                    "yoy_arrivals_%", "demand_index", "weather_score",
                    "resilience_score", "zone"]
    available_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[available_cols].style
        .format({
            "arrivals": "{:,.0f}",
            "nights": "{:,.0f}",
            "avg_stay": "{:.2f}",
            "yoy_arrivals_%": "{:+.1f}%",
            "demand_index": "{:.1f}",
            "weather_score": "{:.1f}",
            "resilience_score": "{:.1f}",
        }, na_rep="N/A"),
        use_container_width=True,
        height=380,
    )

    st.markdown("""
    <div class='data-source'>
    Sources: Eurostat tour_occ_arn2 & tour_occ_nin2 (NUTS-2) | Google Trends (GR) | Open-Meteo forecast API
    </div>
    """, unsafe_allow_html=True)

    # ── Measured Variables & Two-Stage Pillar Weights ──
    with st.expander("Measured Variables & Two-Stage Pillar Weights"):
        sources = {
            "arrivals": f"Eurostat `tour_occ_arn2` (NUTS-2, {euro_year})",
            "nights": f"Eurostat `tour_occ_nin2` (NUTS-2, {euro_year})",
            "demand": "Google Trends API (pytrends, weekly)",
            "weather": "Open-Meteo Forecast API (hourly)",
            "avg_stay": f"Derived: nights / arrivals ({euro_year})",
        }

        if pillar_result:
            st.markdown(f"**Weighting Method:** Two-Stage Pillar Entropy")
            st.markdown("""
**Stage 1:** Within each pillar, Shannon Entropy assigns weights based on cross-municipality variation.
**Stage 2:** Each pillar receives **equal weight (1/3)**, preventing correlated supply indicators from dominating.
            """)

            # Pillar summary
            pillar_rows = []
            for pname, pdef in PILLARS.items():
                pillar_rows.append({
                    "Pillar": PILLAR_NAMES[pname],
                    "Indicators": ", ".join(COMPONENT_NAMES.get(v, v) for v in pdef["indicators"]),
                    "Pillar Weight": f"{pdef['pillar_weight']:.2%}",
                })
            st.dataframe(pd.DataFrame(pillar_rows), use_container_width=True, hide_index=True)

            # Detailed indicator weights
            var_rows = []
            for pname, pdet in pillar_result["pillar_details"].items():
                for var, ww in pdet["within_weights"].items():
                    var_rows.append({
                        "Variable": COMPONENT_NAMES.get(var, var),
                        "Pillar": PILLAR_NAMES[pname],
                        "Within-Pillar w": f"{ww:.2%}",
                        "Final Weight": f"{weights[var]:.2%}",
                        "Source": sources.get(var, ""),
                    })
            st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

            max_var = max(weights, key=weights.get)
            min_var = min(weights, key=weights.get)
            st.info(
                f"**Highest weight:** {COMPONENT_NAMES[max_var]} ({weights[max_var]:.1%})\n\n"
                f"**Lowest weight:** {COMPONENT_NAMES[min_var]} ({weights[min_var]:.1%})"
            )
        else:
            st.markdown("**Weighting Method:** Static (fallback)")
            st.markdown("Pillar entropy computation was not available. Using fallback weights.")

    # Export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv,
                       file_name=f"resilience_tier0_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")


# =================================================================
# PAGE 2: MUNICIPALITY DEEP-DIVE
# =================================================================
elif page == "Municipality Deep-Dive":
    st.markdown("""
    <div class='main-header'>
        <h1>Municipality Deep-Dive</h1>
        <p>Component analysis with real data breakdown</p>
    </div>
    """, unsafe_allow_html=True)

    focus = st.selectbox("Select Municipality:", df["dimos"].tolist())
    f = df[df["dimos"] == focus].iloc[0]

    # KPI row
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
    zone_class = "zone-high" if f["zone"] == "ΥΨΗΛΗ" else "zone-mid" if f["zone"] == "ΜΕΣΑΙΑ" else "zone-low"

    fc1.markdown(f"""<div class='kpi-card'>
        <div class='value {zone_class}'>{f['resilience_score']:.1f}</div>
        <div class='label'>Score ({f['zone']})</div>
    </div>""", unsafe_allow_html=True)

    arrivals_str = f"{f['arrivals']:,.0f}" if pd.notna(f.get('arrivals')) else "N/A"
    fc2.metric("Arrivals (Eurostat)", arrivals_str)

    nights_str = f"{f['nights']:,.0f}" if pd.notna(f.get('nights')) else "N/A"
    fc3.metric("Nights (Eurostat)", nights_str)

    fc4.metric("Avg Stay (days)", f"{f['avg_stay']:.2f}" if pd.notna(f.get('avg_stay')) else "N/A")
    fc5.metric("Demand Index", f"{f['demand_index']:.1f}", delta=f"{shock_demand}%" if shock_demand else None)

    temp_str = f"{f['temperature']:.1f}C" if pd.notna(f.get('temperature')) else "N/A"
    weather_str = f.get("weather_desc", "N/A")
    fc6.metric("Weather Now", temp_str, delta=weather_str)

    st.divider()
    col_wf, col_radar = st.columns(2)

    # ── Waterfall (SHAP-style decomposition) ──
    with col_wf:
        components = ["Arrivals", "Nights", "Demand", "Weather", "Avg Stay"]
        values = [f["w_arrivals"], f["w_nights"], f["w_demand"], f["w_weather"], f["w_avg_stay"]]

        fig_wf = go.Figure(go.Waterfall(
            name="Score Breakdown",
            orientation="v",
            measure=["relative"] * 5 + ["total"],
            x=components + ["TOTAL"],
            y=values + [0],
            text=[f"{v:.1f}" for v in values] + [f"{f['resilience_score']:.1f}"],
            textposition="outside",
            connector={"line": {"color": "#ccc"}},
            increasing={"marker": {"color": "#27AE60"}},
            decreasing={"marker": {"color": "#C0392B"}},
            totals={"marker": {"color": "#1E50B4"}},
        ))
        fig_wf.update_layout(
            title=f"Score Decomposition: {focus}",
            height=420, margin=dict(t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── Radar vs national average ──
    with col_radar:
        cats = ["Arrivals", "Nights", "Demand", "Weather", "Avg Stay"]
        municipality_vals = [f["norm_arrivals"], f["norm_nights"], f["norm_demand"],
                             f["norm_weather"], f["norm_avg_stay"]]
        avg_vals = [df["norm_arrivals"].mean(), df["norm_nights"].mean(), df["norm_demand"].mean(),
                    df["norm_weather"].mean(), df["norm_avg_stay"].mean()]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=municipality_vals + [municipality_vals[0]],
            theta=cats + [cats[0]],
            name=focus, fill="toself", opacity=0.4,
            line=dict(color="#1E50B4"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=cats + [cats[0]],
            name="National Average", fill="toself", opacity=0.2,
            line=dict(color="#888", dash="dash"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"Profile: {focus} vs Average",
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Weather detail ──
    st.subheader(f"Live Weather: {focus}")
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wc1.metric("Temperature", f"{f.get('temperature', 'N/A')}C")
    wc2.metric("Humidity", f"{f.get('humidity', 'N/A')}%")
    wc3.metric("Wind", f"{f.get('wind_kmh', 'N/A')} km/h")
    wc4.metric("Sunshine (7d avg)", f"{f.get('avg_sunshine_hrs', 'N/A')} hrs")
    wc5.metric("UV Index (7d avg)", f"{f.get('avg_uv', 'N/A')}")

    # ── Insight ──
    yoy = f.get("yoy_arrivals_%", 0)
    yoy_str = f"{yoy:+.1f}%" if pd.notna(yoy) else "N/A"

    if f["resilience_score"] >= 65:
        insight = (
            f"**{focus}** ({f['region']}) shows **HIGH** resilience (score {f['resilience_score']:.1f}). "
            f"Eurostat {euro_year} arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Strong tourism activity supported by favorable conditions."
        )
    elif f["resilience_score"] >= 45:
        insight = (
            f"**{focus}** ({f['region']}) is in the **MODERATE** zone (score {f['resilience_score']:.1f}). "
            f"Eurostat {euro_year} arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Room for improvement in underperforming components — see waterfall chart."
        )
    else:
        insight = (
            f"**{focus}** ({f['region']}) is in the **CRITICAL** zone (score {f['resilience_score']:.1f}). "
            f"Eurostat {euro_year} arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Immediate attention needed. Weather score: {f['weather_score']:.1f}/100."
        )
    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)


# =================================================================
# PAGE 3: TRENDS & TIME SERIES
# =================================================================
elif page == "Trends & Time Series":
    st.markdown("""
    <div class='main-header'>
        <h1>Tourism Demand Trends</h1>
        <p>Google Trends real-time data + Eurostat historical arrivals</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Google Trends time series ──
    st.subheader("Google Trends: Tourism Keywords (Greece)")

    if not df_trends.empty:
        kw_cols = [c for c in df_trends.columns if c not in ("date", "isPartial", "demand_index")]
        fig_trends = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, kw in enumerate(kw_cols):
            fig_trends.add_trace(go.Scatter(
                x=df_trends["date"], y=df_trends[kw],
                name=kw, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        # Add composite
        fig_trends.add_trace(go.Scatter(
            x=df_trends["date"], y=df_trends["demand_index"],
            name="Demand Index (mean)", mode="lines",
            line=dict(color="#1E50B4", width=3, dash="dash"),
        ))
        fig_trends.update_layout(
            height=400, margin=dict(t=10, b=10),
            yaxis_title="Search Interest (0-100)",
            xaxis_title="Date",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_trends, use_container_width=True)

        st.markdown(f"""
        <div class='data-source'>
        Source: Google Trends API via pytrends | Keywords: {', '.join(kw_cols)} |
        Latest demand index: {df_trends['demand_index'].iloc[-1]:.1f} |
        Period: {df_trends['date'].iloc[0].strftime('%Y-%m-%d')} to {df_trends['date'].iloc[-1].strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Google Trends data unavailable.")

    st.divider()

    # ── Eurostat Historical ──
    st.subheader("Eurostat: Tourism Arrivals by Region (NUTS-2)")

    if not df_euro.empty:
        # Pivot for chart
        nuts_labels = {m["nuts2"]: m["region"] for m in MUNICIPALITIES.values()}
        df_euro_plot = df_euro.copy()
        df_euro_plot["region"] = df_euro_plot["nuts2"].map(nuts_labels).fillna(df_euro_plot["nuts2"])
        df_euro_plot = df_euro_plot.drop_duplicates(subset=["region", "year"])

        fig_euro = px.bar(
            df_euro_plot, x="year", y="arrivals", color="region",
            barmode="group",
            labels={"arrivals": "Arrivals", "year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=420,
        )
        fig_euro.update_layout(
            margin=dict(t=10, b=10),
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_euro, use_container_width=True)

        # Nights spent
        st.subheader("Eurostat: Nights Spent by Region")
        fig_nights = px.bar(
            df_euro_plot, x="year", y="nights", color="region",
            barmode="group",
            labels={"nights": "Nights Spent", "year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            height=420,
        )
        fig_nights.update_layout(margin=dict(t=10, b=10), xaxis=dict(dtick=1),
                                  legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_nights, use_container_width=True)

        # YoY table
        st.subheader("Year-over-Year Changes")
        latest = df_euro[df_euro["year"] == df_euro["year"].max()].copy()
        latest["region"] = latest["nuts2"].map(nuts_labels)
        st.dataframe(
            latest[["region", "nuts2", "year", "arrivals", "nights", "avg_stay", "yoy_arrivals_%"]].style
            .format({
                "arrivals": "{:,.0f}",
                "nights": "{:,.0f}",
                "avg_stay": "{:.2f}",
                "yoy_arrivals_%": "{:+.1f}%",
            }, na_rep="N/A")
            ,
            use_container_width=True,
        )

        st.markdown("""
        <div class='data-source'>
        Source: Eurostat REST API | Datasets: tour_occ_arn2, tour_occ_nin2 |
        Level: NUTS-2 | Accommodation: Hotels & similar (I551-I553)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Eurostat data unavailable.")


# =================================================================
# PAGE 4: METHODOLOGY
# =================================================================
else:
    st.markdown("""
    <div class='main-header'>
        <h1>Methodology & Data Sources</h1>
        <p>Transparent, reproducible, open-data composite index</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Composite Resilience Index (v{datetime.now().strftime('%Y.%m')})")
    st.markdown(f"**Weighting method:** {weighting_method}")

    st.markdown("### Measured Variables")
    st.markdown("""
    The Resilience Score is computed **in real-time** from 3 live data sources,
    producing 5 normalized indicators across 3 pillars. Weights are derived via
    **Two-Stage Pillar Entropy** to prevent correlated supply indicators from
    dominating single-indicator pillars (demand, weather).
    """)

    # Dynamic weights table
    w_rows = []
    sources_info = {
        "arrivals": ("Eurostat `tour_occ_arn2` (NUTS-2)", "Annual"),
        "nights": ("Eurostat `tour_occ_nin2` (NUTS-2)", "Annual"),
        "demand": ("Google Trends (pytrends)", "Weekly"),
        "weather": ("Open-Meteo Forecast API", "Hourly"),
        "avg_stay": ("Derived (nights / arrivals)", "Annual"),
    }
    for var in weights:
        src, freq = sources_info.get(var, ("", ""))
        w_rows.append({
            "Component": COMPONENT_NAMES.get(var, var),
            "Weight": f"{weights[var]:.1%}",
            "Source": src,
            "Update": freq,
        })
    st.table(pd.DataFrame(w_rows))

    st.markdown("### Two-Stage Pillar Entropy Weighting")
    st.markdown(f"""
    **Problem with flat entropy:** The 5 indicators are not independent — arrivals,
    nights, and avg_stay are correlated supply-side metrics. Flat Shannon Entropy
    would over-weight this cluster (3 variables competing) while under-weighting
    demand (national-level, uniform across municipalities → entropy ≈ 1 → weight ≈ 0).

    **Solution — Two-stage approach:**

    | Stage | What | How |
    |-------|------|-----|
    | **Stage 1** | Within-pillar weights | Shannon Entropy among indicators in each pillar |
    | **Stage 2** | Pillar weights | Equal weight: **1/3** per pillar |

    **Pillars:**
    | Pillar | Indicators | Pillar Weight |
    |--------|-----------|---------------|
    | Supply Capacity | Arrivals, Nights, Avg Stay | 33.3% |
    | Demand Pressure | Tourism Demand | 33.3% |
    | Environmental | Weather Attractiveness | 33.3% |

    **Final weight** = pillar_weight × within_pillar_weight

    **Shannon Entropy (within multi-indicator pillars):**
    ```
    Step 1:  p_ij = x_ij / Σ_i(x_ij)                  — proportion
    Step 2:  E_j  = -(1/ln(n)) × Σ_i[p_ij × ln(p_ij)] — entropy (0=max info, 1=no info)
    Step 3:  d_j  = 1 - E_j                             — diversification degree
    Step 4:  w_j  = d_j / Σ_k(d_k)                      — normalized within-pillar weight
    ```

    Where n = {len(MUNICIPALITIES)} municipalities.
    """)

    # Show pillar diagnostics if available
    if pillar_result:
        st.markdown("### Pillar Weight Diagnostics (Current Data)")

        for pname, pdet in pillar_result["pillar_details"].items():
            with st.expander(f"Pillar: {PILLAR_NAMES[pname]} (weight = {pdet['pillar_weight']:.2%})"):
                diag_rows = []
                for var, ww in pdet["within_weights"].items():
                    row = {
                        "Variable": COMPONENT_NAMES.get(var, var),
                        "Within-Pillar w": ww,
                        "Final Weight": pillar_result["weights"][var],
                    }
                    if pdet["entropy"][var] is not None:
                        row["E_j (Entropy)"] = pdet["entropy"][var]
                        row["d_j (Diversity)"] = pdet["diversity"][var]
                    diag_rows.append(row)
                diag_df = pd.DataFrame(diag_rows)
                fmt = {"Within-Pillar w": "{:.2%}", "Final Weight": "{:.2%}"}
                if "E_j (Entropy)" in diag_df.columns:
                    fmt["E_j (Entropy)"] = "{:.4f}"
                    fmt["d_j (Diversity)"] = "{:.4f}"
                st.dataframe(diag_df.style.format(fmt), use_container_width=True, hide_index=True)

    st.markdown("""
    ### Composite Score Formula

    ```
    Score = (1/3) × [w₁·Arrivals + w₂·Nights + w₃·AvgStay]   ← Supply (entropy-weighted)
          + (1/3) × [Demand]                                    ← Demand
          + (1/3) × [Weather]                                   ← Environment
    ```

    Where `w₁, w₂, w₃` are Shannon Entropy weights within the Supply pillar,
    and all indicator values are min-max normalized to 0-100.
    """)

    st.markdown("""
    ### Weather Attractiveness Score

    The weather sub-score is a composite of:
    - **Temperature** (optimal: 25-30C, penalty for deviation)
    - **Sunshine hours** (7-day forecast average)
    - **Wind speed** (penalty above 20 km/h)
    - **Rain** (penalty for active precipitation)
    """)

    st.markdown("""
    ### Zone Classification

    | Zone | Score Range | Interpretation |
    |------|-----------|----------------|
    | **ΥΨΗΛΗ** (High) | >= 65 | Strong resilience, sustainable trajectory |
    | **ΜΕΣΑΙΑ** (Moderate) | 45 - 64 | Functional with vulnerabilities |
    | **ΧΑΜΗΛΗ** (Low) | < 45 | Critical, requires policy intervention |
    """)

    st.markdown("""
    ### NUTS-2 to Municipality Mapping

    Eurostat data is at NUTS-2 level. We apply proportional tourism-share
    coefficients to estimate municipality-level values:

    | Municipality | NUTS-2 | Region | Share |
    |---|---|---|---|
    | Ρόδος | EL42 | Ν. Αιγαίο | 50% |
    | Σαντορίνη | EL42 | Ν. Αιγαίο | 28% |
    | Μύκονος | EL42 | Ν. Αιγαίο | 22% |
    | Ηράκλειο | EL43 | Κρήτη | 65% |
    | Χανιά | EL43 | Κρήτη | 35% |
    | Κέρκυρα | EL62 | Ιόνια Νησιά | 100% |
    | Αθήνα | EL30 | Αττική | 100% |
    | Θεσσαλονίκη | EL52 | Κ. Μακεδονία | 100% |
    | Ναύπλιο | EL65 | Πελοπόννησος | 30% |
    | Ιωάννινα | EL54 | Ήπειρος | 50% |
    | Λάρισα | EL61 | Θεσσαλία | 40% |
    | Πάτρα | EL63 | Δ. Ελλάδα | 45% |
    | Αλεξανδρούπολη | EL51 | Α. Μακεδονία & Θράκη | 35% |
    | Μυτιλήνη | EL41 | Β. Αιγαίο | 40% |
    """)

    st.markdown("""
    ### Tier Upgrade Path

    - **Tier 0** (Free) - Open data, 14 municipalities, Two-Stage Pillar Entropy weights
    - **Tier 1** (Regional Lite) - All NUTS-3, SHAP + Entropy comparison
    - **Tier 2** (Analytics) - Shift-Share, Martin Index, TFT forecasts
    - **Tier 3** (DSS) - Full decision support, scenario engine, API access
    """)

    st.info("For full analytics (Shift-Share, Martin Index, TFT forecasts) upgrade to ResilienceIQ Tier 2.")
