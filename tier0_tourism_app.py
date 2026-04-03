"""
ResilienceIQ Tier 0 — Realtime Tourism Resilience Dashboard (v2: 3x3 Pillars)
Live data: Eurostat (supply), Google Trends + Wikipedia + Seasonality (demand),
           Open-Meteo Weather + Air Quality + Marine (environment).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

from data_fetcher import (
    MUNICIPALITIES, WEIGHTS_FALLBACK, COMPONENT_NAMES, PILLARS, PILLAR_NAMES,
    fetch_eurostat_tourism, fetch_eurostat_seasonality,
    fetch_google_trends, fetch_destination_trends, fetch_wikipedia_pageviews,
    fetch_weather, fetch_air_quality, fetch_marine,
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
    st.caption("Tier 0 — Live Tourism Resilience (3x3 Pillars)")
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

    st.caption(f"Last update: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")


# ─── Fetch Real Data (cached) ───────────────────
@st.cache_data(ttl=1800, show_spinner="Fetching Eurostat annual data...")
def _fetch_eurostat():
    return fetch_eurostat_tourism()

@st.cache_data(ttl=1800, show_spinner="Fetching Eurostat seasonality...")
def _fetch_seasonality():
    return fetch_eurostat_seasonality()

@st.cache_data(ttl=3600, show_spinner="Fetching Google Trends (national)...")
def _fetch_trends():
    return fetch_google_trends()

@st.cache_data(ttl=3600, show_spinner="Fetching destination search interest...")
def _fetch_dest_trends():
    return fetch_destination_trends()

@st.cache_data(ttl=3600, show_spinner="Fetching Wikipedia pageviews...")
def _fetch_wiki():
    return fetch_wikipedia_pageviews()

@st.cache_data(ttl=900, show_spinner="Fetching live weather...")
def _fetch_weather():
    return fetch_weather()

@st.cache_data(ttl=900, show_spinner="Fetching air quality...")
def _fetch_air_quality():
    return fetch_air_quality()

@st.cache_data(ttl=900, show_spinner="Fetching marine data...")
def _fetch_marine():
    return fetch_marine()

df_euro = _fetch_eurostat()
df_season = _fetch_seasonality()
df_trends = _fetch_trends()
df_dest = _fetch_dest_trends()
df_wiki = _fetch_wiki()
df_weather = _fetch_weather()
df_air = _fetch_air_quality()
df_marine = _fetch_marine()
fetch_ts = datetime.now()

euro_year = df_euro["year"].max() if not df_euro.empty else "N/A"

# Build snapshot (not cached — reacts to shock sliders)
df = build_resilience_snapshot(
    df_euro, df_dest, df_weather, df_air, df_marine, df_wiki, df_season,
    shock_demand=shock_demand, shock_arrivals=shock_arrivals,
)

# Extract diagnostics, then clear attrs for serialization
weights = df.attrs.get("weights", WEIGHTS_FALLBACK)
pillar_result = df.attrs.get("pillar_result")
weighting_method = df.attrs.get("weighting_method", "Static")
df.attrs.clear()


# =================================================================
# PAGE 1: DASHBOARD
# =================================================================
if page == "Dashboard":
    st.markdown(f"""
    <div style="text-align:right; font-size:1.15em; color:#444; margin-bottom:-0.5rem;">
        📡 Data fetched: <b>{fetch_ts.strftime('%d %B %Y, %H:%M:%S')}</b> · Eurostat reference year: <b>{euro_year}</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='main-header'>
        <h1>Resilience Control Center</h1>
        <p>Live tourism resilience | 9 indicators × 3 pillars | Eurostat + Google Trends + Wikipedia + Open-Meteo</p>
    </div>
    """, unsafe_allow_html=True)

    avg_score = df["resilience_score"].mean()
    top = df.iloc[0]
    bottom = df.iloc[-1]
    high_n = (df["zone"] == "ΥΨΗΛΗ").sum()

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
                "resilience_score": ":.1f", "region": True,
                "arrivals": ":,.0f", "lat": False, "lon": False, "zone": False,
            },
            zoom=5.5, center={"lat": 38.0, "lon": 24.0},
            mapbox_style="carto-positron", height=480,
        )
        # Greece border overlay (bundled GeoJSON)
        @st.cache_data
        def _load_greece_border():
            try:
                with open("greece_border.geojson", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
        gr_geo = _load_greece_border()
        if gr_geo:
            fig_map.update_layout(
                mapbox_layers=[{
                    "source": gr_geo,
                    "type": "line",
                    "color": "#888888",
                    "line": {"width": 2.5},
                }]
            )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_bar:
        mu = df["resilience_score"].mean()
        sigma = df["resilience_score"].std()
        high_t = mu + 0.5 * sigma
        low_t = mu - 0.5 * sigma
        st.subheader("Resilience Ranking")
        st.markdown(
            f"**μ** = {mu:.1f} &ensp; **σ** = {sigma:.1f} &ensp; | &ensp;"
            f"🟢 High ≥ **{high_t:.1f}** &ensp; "
            f"🟠 Moderate ≥ **{low_t:.1f}** &ensp; "
            f"🔴 Low < **{low_t:.1f}**"
        )
        fig_bar = go.Figure(go.Bar(
            x=df["resilience_score"], y=df["dimos"], orientation="h",
            marker_color=df["color"],
            text=df["resilience_score"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=480, margin=dict(l=0, r=50, t=0, b=0),
            xaxis=dict(range=[0, 105], title="Resilience Score"),
            yaxis=dict(autorange="reversed"),
        )
        fig_bar.add_vline(x=high_t, line_dash="dash", line_color="#27AE60", line_width=1.5,
                          annotation_text=f"μ+0.5σ = {high_t:.1f}", annotation_position="top right")
        fig_bar.add_vline(x=low_t, line_dash="dash", line_color="#C0392B", line_width=1.5,
                          annotation_text=f"μ−0.5σ = {low_t:.1f}", annotation_position="top left")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Pillar Scores Comparison ──
    st.subheader("Pillar Score Comparison")
    pillar_cols = [f"pillar_{p}" for p in PILLARS]
    if all(c in df.columns for c in pillar_cols):
        fig_pillars = go.Figure()
        colors_p = {"supply": "#2196F3", "demand": "#FF9800", "environment": "#4CAF50"}
        for pname in PILLARS:
            fig_pillars.add_trace(go.Bar(
                name=PILLAR_NAMES[pname],
                x=df["dimos"], y=df[f"pillar_{pname}"],
                marker_color=colors_p[pname],
            ))
        fig_pillars.update_layout(
            barmode="group", height=380,
            margin=dict(t=10, b=10),
            yaxis_title="Pillar Score (0-100)",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_pillars, use_container_width=True)

    # ── Data Table ──
    st.subheader("Full Data Table (Live)")
    display_cols = [
        "dimos", "region", "arrivals", "nights", "avg_stay",
        "dest_interest", "wiki_views", "seasonality_score",
        "weather_score", "air_quality_score", "coastal_score",
        "resilience_score", "zone",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[available_cols].style.format({
            "arrivals": "{:,.0f}", "nights": "{:,.0f}", "avg_stay": "{:.2f}",
            "dest_interest": "{:.1f}", "wiki_views": "{:,.0f}",
            "seasonality_score": "{:.1f}", "weather_score": "{:.1f}",
            "air_quality_score": "{:.1f}", "coastal_score": "{:.1f}",
            "resilience_score": "{:.1f}",
        }, na_rep="N/A"),
        use_container_width=True, height=420,
    )

    st.markdown("""
    <div class='data-source'>
    Sources: Eurostat tour_occ_arn2/nin2/nim (NUTS-2) | Google Trends (GR) |
    Wikipedia Pageviews API (en+de+fr) | Open-Meteo Forecast + Air Quality + Marine APIs
    </div>
    """, unsafe_allow_html=True)

    # ── Weights Expander ──
    with st.expander("Two-Stage Pillar Entropy Weights (3x3)"):
        sources = {
            "arrivals": f"Eurostat tour_occ_arn2 ({euro_year})",
            "nights": f"Eurostat tour_occ_nin2 ({euro_year})",
            "avg_stay": f"Derived: nights/arrivals ({euro_year})",
            "dest_interest": "Google Trends per destination (worldwide)",
            "wiki_views": "Wikipedia Pageviews API (en+de+fr)",
            "seasonality": "Eurostat monthly / fallback",
            "weather": "Open-Meteo Forecast API",
            "air_quality": "Open-Meteo Air Quality API",
            "coastal": "Open-Meteo Marine API",
        }

        if pillar_result:
            pw = pillar_result.get("pillar_weights", {})
            st.markdown("**Stage 1:** Within-pillar Shannon Entropy — weights 3 indicators per pillar by cross-municipality variation")
            st.markdown("**Stage 2:** Cross-pillar Shannon Entropy — weights the 3 pillar composite scores by cross-municipality variation")

            # Pillar summary
            pillar_rows = []
            for pname in PILLARS:
                pillar_rows.append({
                    "Pillar": PILLAR_NAMES[pname],
                    "Indicators": ", ".join(COMPONENT_NAMES.get(v, v) for v in PILLARS[pname]["indicators"]),
                    "Pillar Weight (Entropy)": f"{pw.get(pname, 0):.2%}",
                })
            st.dataframe(pd.DataFrame(pillar_rows), use_container_width=True, hide_index=True)

            # Indicator weights
            var_rows = []
            for pname, pdet in pillar_result["pillar_details"].items():
                for var, ww in pdet["within_weights"].items():
                    var_rows.append({
                        "Variable": COMPONENT_NAMES.get(var, var),
                        "Pillar": PILLAR_NAMES[pname],
                        "Within w": f"{ww:.2%}",
                        "Final Weight": f"{weights.get(var, 0):.2%}",
                        "Source": sources.get(var, ""),
                    })
            st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("**Weighting Method:** Static (fallback)")

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
        <p>9-indicator decomposition with pillar analysis</p>
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
    fc2.metric("Arrivals", arrivals_str)
    fc3.metric("Search Interest", f"{f.get('dest_interest', 0):.1f}")
    fc4.metric("Seasonality", f"{f.get('seasonality_score', 50):.0f}/100")

    temp_str = f"{f['temperature']:.1f}C" if pd.notna(f.get('temperature')) else "N/A"
    fc5.metric("Temperature", temp_str)
    fc6.metric("Air Quality", f"{f.get('air_quality_score', 50):.0f}/100")

    st.divider()
    col_wf, col_radar = st.columns(2)

    # ── Waterfall (9 components) ──
    with col_wf:
        comp_keys = list(COMPONENT_NAMES.keys())
        comp_labels = [COMPONENT_NAMES[k] for k in comp_keys]
        values = [f.get(f"w_{k}", 0) for k in comp_keys]

        fig_wf = go.Figure(go.Waterfall(
            name="Score Breakdown", orientation="v",
            measure=["relative"] * 9 + ["total"],
            x=[c[:12] for c in comp_labels] + ["TOTAL"],
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
            height=450, margin=dict(t=40, b=10),
            showlegend=False, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── Radar: Pillar-level comparison ──
    with col_radar:
        pillar_labels = [PILLAR_NAMES[p] for p in PILLARS]
        municipality_pillar_vals = [f.get(f"pillar_{p}", 50) for p in PILLARS]
        avg_pillar_vals = [df[f"pillar_{p}"].mean() for p in PILLARS]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=municipality_pillar_vals + [municipality_pillar_vals[0]],
            theta=pillar_labels + [pillar_labels[0]],
            name=focus, fill="toself", opacity=0.4,
            line=dict(color="#1E50B4"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_pillar_vals + [avg_pillar_vals[0]],
            theta=pillar_labels + [pillar_labels[0]],
            name="National Average", fill="toself", opacity=0.2,
            line=dict(color="#888", dash="dash"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"Pillar Profile: {focus} vs Average",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── 9-indicator radar ──
    st.subheader("Full 9-Indicator Profile")
    cats_9 = [COMPONENT_NAMES[k][:14] for k in COMPONENT_NAMES]
    mun_vals_9 = [f.get(f"norm_{k}", 50) for k in COMPONENT_NAMES]
    avg_vals_9 = [df[f"norm_{k}"].mean() for k in COMPONENT_NAMES]

    fig_r9 = go.Figure()
    fig_r9.add_trace(go.Scatterpolar(
        r=mun_vals_9 + [mun_vals_9[0]],
        theta=cats_9 + [cats_9[0]],
        name=focus, fill="toself", opacity=0.4, line=dict(color="#1E50B4"),
    ))
    fig_r9.add_trace(go.Scatterpolar(
        r=avg_vals_9 + [avg_vals_9[0]],
        theta=cats_9 + [cats_9[0]],
        name="National Average", fill="toself", opacity=0.2,
        line=dict(color="#888", dash="dash"),
    ))
    fig_r9.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
    )
    st.plotly_chart(fig_r9, use_container_width=True)

    # ── Weather + Environment detail ──
    st.subheader(f"Environment Detail: {focus}")
    wc1, wc2, wc3, wc4, wc5, wc6 = st.columns(6)
    wc1.metric("Weather Score", f"{f.get('weather_score', 'N/A')}")
    wc2.metric("Air Quality", f"{f.get('air_quality_score', 'N/A')}")
    wc3.metric("Coastal Score", f"{f.get('coastal_score', 'N/A')}")
    wc4.metric("Wind", f"{f.get('wind_kmh', 'N/A')} km/h")
    wc5.metric("Sunshine (7d)", f"{f.get('avg_sunshine_hrs', 'N/A')} hrs")
    wc6.metric("UV Index", f"{f.get('avg_uv', 'N/A')}")

    # ── Insight ──
    yoy = f.get("yoy_arrivals_%", 0)
    yoy_str = f"{yoy:+.1f}%" if pd.notna(yoy) else "N/A"

    if f["zone"] == "ΥΨΗΛΗ":
        insight = (
            f"**{focus}** ({f['region']}) shows **HIGH** resilience (score {f['resilience_score']:.1f}). "
            f"Eurostat arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Wikipedia awareness: {f.get('wiki_views', 0):,.0f} views."
        )
    elif f["zone"] == "ΜΕΣΑΙΑ":
        insight = (
            f"**{focus}** ({f['region']}) is in the **MODERATE** zone ({f['resilience_score']:.1f}). "
            f"Eurostat arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Check underperforming pillars in the radar chart."
        )
    else:
        insight = (
            f"**{focus}** ({f['region']}) is in the **CRITICAL** zone ({f['resilience_score']:.1f}). "
            f"Eurostat arrivals: {arrivals_str} (YoY: {yoy_str}). "
            f"Seasonality score: {f.get('seasonality_score', 'N/A')}/100."
        )
    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)


# =================================================================
# PAGE 3: TRENDS & TIME SERIES
# =================================================================
elif page == "Trends & Time Series":
    st.markdown("""
    <div class='main-header'>
        <h1>Tourism Demand Trends</h1>
        <p>Google Trends + Wikipedia Pageviews + Eurostat historical data</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Google Trends ──
    st.subheader("Google Trends: Tourism Keywords (Greece)")
    if not df_trends.empty:
        kw_cols = [c for c in df_trends.columns if c not in ("date", "isPartial", "demand_index")]
        fig_trends = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, kw in enumerate(kw_cols):
            fig_trends.add_trace(go.Scatter(
                x=df_trends["date"], y=df_trends[kw], name=kw, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_trends.add_trace(go.Scatter(
            x=df_trends["date"], y=df_trends["demand_index"],
            name="Demand Index (mean)", mode="lines",
            line=dict(color="#1E50B4", width=3, dash="dash"),
        ))
        fig_trends.update_layout(
            height=400, margin=dict(t=10, b=10),
            yaxis_title="Search Interest (0-100)", xaxis_title="Date",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.warning("Google Trends data unavailable.")

    st.divider()

    # ── Wikipedia Pageviews ──
    st.subheader("Wikipedia Pageviews: Destination Awareness")
    if "wiki_views" in df.columns:
        wiki_sorted = df.sort_values("wiki_views", ascending=True)
        fig_wiki = go.Figure(go.Bar(
            x=wiki_sorted["wiki_views"], y=wiki_sorted["dimos"],
            orientation="h", marker_color="#FF9800",
            text=wiki_sorted["wiki_views"].apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
        ))
        fig_wiki.update_layout(
            height=420, margin=dict(l=0, r=80, t=10, b=10),
            xaxis_title="Pageviews (en+de+fr, last 3 months)",
        )
        st.plotly_chart(fig_wiki, use_container_width=True)
        st.markdown("""
        <div class='data-source'>
        Source: Wikimedia Pageviews API | Languages: en + de + fr |
        Period: last 3 complete months | Proxy for international tourism awareness
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Eurostat Historical ──
    st.subheader("Eurostat: Tourism Arrivals by Region (NUTS-2)")
    if not df_euro.empty:
        nuts_labels = {m["nuts2"]: m["region"] for m in MUNICIPALITIES.values()}
        df_euro_plot = df_euro.copy()
        df_euro_plot["region"] = df_euro_plot["nuts2"].map(nuts_labels).fillna(df_euro_plot["nuts2"])
        df_euro_plot = df_euro_plot.drop_duplicates(subset=["region", "year"])

        fig_euro = px.bar(
            df_euro_plot, x="year", y="arrivals", color="region", barmode="group",
            labels={"arrivals": "Arrivals", "year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2, height=420,
        )
        fig_euro.update_layout(margin=dict(t=10, b=10), xaxis=dict(dtick=1),
                               legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_euro, use_container_width=True)

        # YoY table
        st.subheader("Year-over-Year Changes")
        latest = df_euro[df_euro["year"] == df_euro["year"].max()].copy()
        latest["region"] = latest["nuts2"].map(nuts_labels)
        st.dataframe(
            latest[["region", "nuts2", "year", "arrivals", "nights", "avg_stay", "yoy_arrivals_%"]].style
            .format({
                "arrivals": "{:,.0f}", "nights": "{:,.0f}", "avg_stay": "{:.2f}",
                "yoy_arrivals_%": "{:+.1f}%",
            }, na_rep="N/A"),
            use_container_width=True,
        )
    else:
        st.warning("Eurostat data unavailable.")


# =================================================================
# PAGE 4: METHODOLOGY
# =================================================================
else:
    st.markdown("""
    <div class='main-header'>
        <h1>Methodology & Data Sources</h1>
        <p>Transparent, reproducible, open-data composite index (3x3 balanced pillars)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Composite Resilience Index (v{datetime.now().strftime('%Y.%m')})")
    st.markdown(f"**Weighting method:** {weighting_method}")

    st.markdown("### 3x3 Pillar Architecture")
    st.markdown("""
    The Resilience Score is computed from **9 indicators** across **3 balanced pillars**.
    Each pillar has exactly 3 indicators, enabling Shannon Entropy at both levels:
    """)

    # Architecture table
    arch_rows = []
    sources_info = {
        "arrivals": ("Eurostat `tour_occ_arn2`", "Annual"),
        "nights": ("Eurostat `tour_occ_nin2`", "Annual"),
        "avg_stay": ("Derived (nights/arrivals)", "Annual"),
        "demand": ("Google Trends (pytrends)", "Weekly"),
        "wiki_views": ("Wikipedia Pageviews API (en+de+fr)", "Monthly"),
        "seasonality": ("Eurostat monthly `tour_occ_nim` / fallback", "Annual"),
        "weather": ("Open-Meteo Forecast API", "Hourly"),
        "air_quality": ("Open-Meteo Air Quality API", "Hourly"),
        "coastal": ("Open-Meteo Marine API", "Daily"),
    }

    pw = {}
    if pillar_result:
        pw = pillar_result.get("pillar_weights", {})

    for pname, pdef in PILLARS.items():
        for var in pdef["indicators"]:
            src, freq = sources_info.get(var, ("", ""))
            arch_rows.append({
                "Pillar": PILLAR_NAMES[pname],
                "Indicator": COMPONENT_NAMES.get(var, var),
                "Weight": f"{weights.get(var, 0):.1%}",
                "Source": src,
                "Update": freq,
            })
    st.table(pd.DataFrame(arch_rows))

    st.markdown("### Two-Stage Entropy Weighting")
    st.markdown(f"""
    **Why two-stage?** The 9 indicators are grouped by conceptual domain. Flat entropy
    would let the most heterogeneous cluster dominate. Two-stage ensures each pillar
    competes on equal footing.

    **Architecture:**
    ```
                        RESILIENCE SCORE
                              |
              Stage 2: Cross-pillar entropy
              |               |               |
        Supply (Ws)     Demand (Wd)     Environment (We)
              |               |               |
        Stage 1: Within-pillar entropy
        |    |    |     |    |    |     |    |    |
       arr nght stay  dem  wiki seas  weat  aqi coast
    ```

    **Stage 1** — Within each pillar, Shannon Entropy assigns weights to 3 indicators
    based on cross-municipality variation (Sigma = 1 per pillar).

    **Stage 2** — Compute pillar composite scores (weighted sum within pillar),
    then Shannon Entropy on the 3 pillar scores to determine pillar-level weights (Sigma = 1).

    **Final weight** = pillar_weight x within_pillar_weight

    **Shannon Entropy formula (applied at both stages):**
    ```
    p_ij = x_ij / Sigma_i(x_ij)                         (proportion)
    E_j  = -(1/ln(n)) x Sigma_i[p_ij x ln(p_ij)]       (entropy, 0=max info, 1=no info)
    d_j  = 1 - E_j                                       (diversification)
    w_j  = d_j / Sigma_k(d_k)                            (normalized weight)
    ```
    Where n = {len(MUNICIPALITIES)} municipalities.
    """)

    # Pillar diagnostics
    if pillar_result:
        st.markdown("### Pillar Weight Diagnostics")

        # Cross-pillar weights
        col_pw = st.columns(3)
        for idx, pname in enumerate(PILLARS):
            p_weight = pw.get(pname, 0)
            col_pw[idx].metric(PILLAR_NAMES[pname], f"{p_weight:.1%}")

        for pname, pdet in pillar_result["pillar_details"].items():
            with st.expander(f"Pillar: {PILLAR_NAMES[pname]} (weight = {pw.get(pname, 0):.2%})"):
                diag_rows = []
                for var, ww in pdet["within_weights"].items():
                    row = {
                        "Variable": COMPONENT_NAMES.get(var, var),
                        "Within-Pillar w": ww,
                        "Final Weight": pillar_result["weights"].get(var, 0),
                    }
                    if pdet["entropy"].get(var) is not None:
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
    Pillar_s = w1*Arrivals + w2*Nights + w3*AvgStay       (supply, entropy-weighted)
    Pillar_d = w4*Demand + w5*WikiViews + w6*Seasonality   (demand, entropy-weighted)
    Pillar_e = w7*Weather + w8*AirQuality + w9*Coastal     (environment, entropy-weighted)

    Score = Ws*Pillar_s + Wd*Pillar_d + We*Pillar_e
    ```
    Where all indicators are min-max normalized to 0-100, and Ws+Wd+We=1 (entropy-derived).
    """)

    st.markdown("### New Indicators (v2)")
    st.markdown("""
    | Indicator | Pillar | What it measures | Why it matters |
    |-----------|--------|-----------------|----------------|
    | **Destination Search Interest** | Demand | Google Trends per destination name (worldwide) | Direct measure of tourism demand per city; replaces national-level index |
    | **Destination Awareness** | Demand | Wikipedia pageviews (en+de+fr, 3 months) | Proxy for international recognition and pre-trip research |
    | **Seasonality Balance** | Demand | HHI of monthly tourism distribution | Low seasonality = year-round demand = higher resilience |
    | **Air Quality** | Environment | European AQI (PM2.5, PM10, O3) | Health-conscious tourism; differentiates urban vs island destinations |
    | **Coastal Comfort** | Environment | Wave height (7-day avg) | Sea conditions for Greece's dominant coastal tourism model |
    """)

    st.markdown("### Zone Classification (μ ± 0.5σ)")
    st.markdown("""
    | Zone | Threshold | Interpretation |
    |------|-----------|----------------|
    | **ΥΨΗΛΗ** (High) | ≥ μ + 0.5σ | Strong resilience, sustainable trajectory |
    | **ΜΕΣΑΙΑ** (Moderate) | μ − 0.5σ ≤ score < μ + 0.5σ | Functional with vulnerabilities |
    | **ΧΑΜΗΛΗ** (Low) | < μ − 0.5σ | Critical, requires policy intervention |

    *Where μ = sample mean and σ = sample standard deviation of composite resilience scores.*
    """)

    st.markdown(f"""
    ### NUTS-2 to Municipality Mapping ({len(MUNICIPALITIES)} destinations)

    | Municipality | NUTS-2 | Region | Tourism Share |
    |---|---|---|---|
    | Ρόδος | EL42 | Ν. Αιγαίο | 50% |
    | Σαντορίνη | EL42 | Ν. Αιγαίο | 28% |
    | Μύκονος | EL42 | Ν. Αιγαίο | 22% |
    | Ηράκλειο | EL43 | Κρήτη | 100% |
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

    - **Tier 0** (Free) - Open data, 13 municipalities, 3x3 Two-Stage Pillar Entropy
    - **Tier 1** (Regional Lite) - All NUTS-3, SHAP + Entropy comparison
    - **Tier 2** (Analytics) - Shift-Share, Martin Index, TFT forecasts
    - **Tier 3** (DSS) - Full decision support, scenario engine, API access
    """)

    st.info("For full analytics (Shift-Share, Martin Index, TFT forecasts) upgrade to ResilienceIQ Tier 2.")
