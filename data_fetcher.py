"""
ResilienceIQ Tier 0 — Real Data Fetcher
Fetches live data from:
  1. Eurostat REST API  — Tourism arrivals & nights by NUTS-2 region
  2. Google Trends      — Tourism demand keywords (time-series)
  3. Open-Meteo         — Current weather per municipality
All sources are free and require no API key.
"""

import pandas as pd
import numpy as np
import requests
import warnings
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Municipality reference table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUNICIPALITIES = {
    "Χανιά":            {"lat": 35.5138, "lon": 24.0180, "nuts2": "EL43", "region": "Κρήτη"},
    "Ρόδος":            {"lat": 36.4341, "lon": 28.2176, "nuts2": "EL42", "region": "Ν. Αιγαίο"},
    "Μύκονος":          {"lat": 37.4467, "lon": 25.3289, "nuts2": "EL42", "region": "Ν. Αιγαίο"},
    "Σαντορίνη":        {"lat": 36.3932, "lon": 25.4615, "nuts2": "EL42", "region": "Ν. Αιγαίο"},
    "Κέρκυρα":          {"lat": 39.6243, "lon": 19.9217, "nuts2": "EL62", "region": "Ιόνια Νησιά"},
    "Αθήνα":            {"lat": 37.9838, "lon": 23.7275, "nuts2": "EL30", "region": "Αττική"},
    "Θεσσαλονίκη":      {"lat": 40.6401, "lon": 22.9444, "nuts2": "EL52", "region": "Κ. Μακεδονία"},
    "Ηράκλειο":         {"lat": 35.3387, "lon": 25.1442, "nuts2": "EL43", "region": "Κρήτη"},
    "Ναύπλιο":          {"lat": 37.5675, "lon": 22.8017, "nuts2": "EL65", "region": "Πελοπόννησος"},
    "Ιωάννινα":         {"lat": 39.6650, "lon": 20.8537, "nuts2": "EL54", "region": "Ήπειρος"},
    "Λάρισα":           {"lat": 39.6390, "lon": 22.4191, "nuts2": "EL61", "region": "Θεσσαλία"},
    "Πάτρα":            {"lat": 38.2466, "lon": 21.7346, "nuts2": "EL63", "region": "Δ. Ελλάδα"},
    "Αλεξανδρούπολη":   {"lat": 40.8469, "lon": 25.8743, "nuts2": "EL51", "region": "Α. Μακεδονία & Θράκη"},
    "Μυτιλήνη":         {"lat": 39.1043, "lon": 26.5518, "nuts2": "EL41", "region": "Β. Αιγαίο"},
}

# Pillar definitions: groups of correlated indicators
# Equal pillar weights (1/3 each) + within-pillar entropy
PILLARS = {
    "supply": {
        "label": "Supply Capacity",
        "indicators": ["arrivals", "nights", "avg_stay"],
        "pillar_weight": 1 / 3,
    },
    "demand": {
        "label": "Demand Pressure",
        "indicators": ["demand"],
        "pillar_weight": 1 / 3,
    },
    "environment": {
        "label": "Environmental",
        "indicators": ["weather"],
        "pillar_weight": 1 / 3,
    },
}

# Fallback weights (used only if entropy computation fails)
WEIGHTS_FALLBACK = {
    "arrivals":  0.1333,   # (1/3) * 0.40
    "nights":    0.1000,   # (1/3) * 0.30
    "avg_stay":  0.1000,   # (1/3) * 0.30
    "demand":    0.3333,   # (1/3) * 1.0
    "weather":   0.3333,   # (1/3) * 1.0
}

# Component labels for display
COMPONENT_NAMES = {
    "arrivals": "Tourism Arrivals",
    "nights": "Nights Spent",
    "demand": "Tourism Demand",
    "weather": "Weather Attractiveness",
    "avg_stay": "Average Stay Duration",
}

PILLAR_NAMES = {
    "supply": "Supply Capacity",
    "demand": "Demand Pressure",
    "environment": "Environmental",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHANNON ENTROPY WEIGHTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_entropy_weights(norm_matrix: pd.DataFrame) -> dict:
    """
    Compute Shannon Entropy weights from a normalized data matrix.

    Given n observations (municipalities) and m variables (indicators),
    variables with MORE heterogeneity across municipalities receive
    HIGHER weight — they discriminate better.

    Steps:
        1. p_ij = x_ij / Σ_i(x_ij)          — proportion of region i in variable j
        2. E_j = -(1/ln(n)) × Σ_i[p_ij × ln(p_ij)]  — entropy (0=max info, 1=no info)
        3. d_j = 1 - E_j                     — diversification degree
        4. w_j = d_j / Σ_k(d_k)              — normalized weight (Σw = 1)

    Parameters
    ----------
    norm_matrix : DataFrame
        Columns = variable names, rows = municipalities.
        Values must be > 0 (add small epsilon if needed).

    Returns
    -------
    dict with keys:
        "weights"    : {var_name: weight}  — final normalized weights (sum=1)
        "entropy"    : {var_name: E_j}     — entropy per variable
        "diversity"  : {var_name: d_j}     — diversification degree
        "proportions": DataFrame           — p_ij matrix
    """
    df = norm_matrix.copy()
    variables = list(df.columns)
    n = len(df)

    # Ensure all values are positive (entropy needs p > 0)
    epsilon = 1e-10
    df = df.clip(lower=epsilon)

    # Step 1: Proportions — p_ij = x_ij / Σ_i(x_ij)
    col_sums = df.sum(axis=0)
    p = df.div(col_sums, axis=1)

    # Step 2: Entropy — E_j = -(1/ln(n)) × Σ_i[p_ij × ln(p_ij)]
    k = 1.0 / np.log(n)  # normalization constant
    entropy = {}
    for var in variables:
        p_col = p[var]
        # Shannon formula: -k * Σ(p * ln(p))
        e_j = -k * (p_col * np.log(p_col)).sum()
        entropy[var] = round(e_j, 6)

    # Step 3: Diversification degree — d_j = 1 - E_j
    diversity = {var: round(1 - entropy[var], 6) for var in variables}

    # Step 4: Normalized weights — w_j = d_j / Σ(d_k)
    d_sum = sum(diversity.values())
    if d_sum == 0:
        # All variables have equal entropy — use uniform weights
        weights = {var: round(1.0 / len(variables), 4) for var in variables}
    else:
        weights = {var: round(diversity[var] / d_sum, 4) for var in variables}

    log.info(f"Shannon Entropy weights: {weights}")
    log.info(f"Entropy values E_j: {entropy}")
    log.info(f"Diversification d_j: {diversity}")

    return {
        "weights": weights,
        "entropy": entropy,
        "diversity": diversity,
        "proportions": p,
    }


def compute_pillar_weights(norm_matrix: pd.DataFrame) -> dict:
    """
    Two-stage weighting: within-pillar Shannon Entropy + equal pillar weights.

    Stage 1 — For multi-indicator pillars (supply: arrivals, nights, avg_stay),
              compute entropy weights among those indicators (sum = 1 within pillar).
    Stage 2 — Each pillar receives equal weight (1/3).

    Final indicator weight = pillar_weight × within_pillar_weight.

    This avoids the correlation bias where 3 supply-side indicators would
    dominate the single demand and weather indicators under flat entropy.
    """
    key_map = {
        "norm_arrivals": "arrivals", "norm_nights": "nights",
        "norm_demand": "demand", "norm_weather": "weather",
        "norm_avg_stay": "avg_stay",
    }
    df = norm_matrix.rename(columns=key_map)

    flat_weights = {}
    pillar_details = {}

    for pillar_name, pillar_def in PILLARS.items():
        indicators = pillar_def["indicators"]
        pw = pillar_def["pillar_weight"]

        if len(indicators) == 1:
            # Single-indicator pillar: weight = pillar_weight × 1.0
            var = indicators[0]
            flat_weights[var] = round(pw, 4)
            pillar_details[pillar_name] = {
                "pillar_weight": pw,
                "within_weights": {var: 1.0},
                "entropy": {var: None},
                "diversity": {var: None},
            }
        else:
            # Multi-indicator pillar: entropy within pillar
            sub_df = df[indicators]
            try:
                ent = compute_entropy_weights(sub_df)
                within_w = ent["weights"]
                for var in indicators:
                    flat_weights[var] = round(pw * within_w[var], 4)
                pillar_details[pillar_name] = {
                    "pillar_weight": pw,
                    "within_weights": within_w,
                    "entropy": ent["entropy"],
                    "diversity": ent["diversity"],
                }
            except Exception as e:
                log.warning(f"Within-pillar entropy failed for {pillar_name}: {e}")
                uniform = 1.0 / len(indicators)
                for var in indicators:
                    flat_weights[var] = round(pw * uniform, 4)
                pillar_details[pillar_name] = {
                    "pillar_weight": pw,
                    "within_weights": {v: uniform for v in indicators},
                    "entropy": {v: None for v in indicators},
                    "diversity": {v: None for v in indicators},
                }

    log.info(f"Two-stage pillar weights: {flat_weights}")
    return {
        "weights": flat_weights,
        "pillar_details": pillar_details,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EUROSTAT: Tourism Arrivals & Nights (NUTS-2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _eurostat_decode(data: dict, targets: set) -> dict:
    """Decode Eurostat JSON-stat flat format into {geo: {year: value}}."""
    dims = data["dimension"]
    dim_ids = data["id"]
    dim_sizes = data["size"]
    geos = dims["geo"]["category"]["index"]
    times = dims["time"]["category"]["index"]
    vals = data["value"]

    # Compute strides
    strides = []
    for i in range(len(dim_sizes)):
        s = 1
        for j in range(i + 1, len(dim_sizes)):
            s *= dim_sizes[j]
        strides.append(s)

    geo_dim = dim_ids.index("geo")
    time_dim = dim_ids.index("time")

    result = {}
    for code in targets:
        if code not in geos:
            continue
        geo_idx = geos[code]
        result[code] = {}
        for t_name, t_idx in times.items():
            flat = geo_idx * strides[geo_dim] + t_idx * strides[time_dim]
            val = vals.get(str(flat))
            if val is not None:
                result[code][t_name] = val
    return result


def fetch_eurostat_tourism(since: str = "2019") -> pd.DataFrame:
    """
    Fetch tourism arrivals AND nights spent from Eurostat for Greek NUTS-2.
    Returns DataFrame: nuts2, year, arrivals, nights, avg_stay, yoy_arrivals_%
    """
    targets = {m["nuts2"] for m in MUNICIPALITIES.values()}

    dfs = {}
    for dataset, col_name in [("tour_occ_arn2", "arrivals"), ("tour_occ_nin2", "nights")]:
        url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
        params = {
            "format": "JSON",
            "lang": "en",
            "sinceTimePeriod": since,
            "unit": "NR",
            "nace_r2": "I551-I553",
            "c_resid": "TOTAL",
        }
        try:
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            decoded = _eurostat_decode(r.json(), targets)
            rows = []
            for geo, yearly in decoded.items():
                for year, val in yearly.items():
                    rows.append({"nuts2": geo, "year": int(year), col_name: val})
            dfs[col_name] = pd.DataFrame(rows)
            log.info(f"Eurostat {dataset}: {len(rows)} records")
        except Exception as e:
            log.warning(f"Eurostat {dataset} failed: {e}")
            dfs[col_name] = pd.DataFrame(columns=["nuts2", "year", col_name])

    # Merge arrivals + nights
    if dfs["arrivals"].empty and dfs["nights"].empty:
        return pd.DataFrame()

    df = pd.merge(dfs["arrivals"], dfs["nights"], on=["nuts2", "year"], how="outer")
    df = df.sort_values(["nuts2", "year"])

    # Derived metrics
    df["avg_stay"] = (df["nights"] / df["arrivals"]).round(2)

    # YoY change
    df["yoy_arrivals_%"] = df.groupby("nuts2")["arrivals"].pct_change() * 100
    df["yoy_arrivals_%"] = df["yoy_arrivals_%"].round(1)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GOOGLE TRENDS: Tourism Demand Keywords
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_google_trends(
    keywords: Optional[list] = None,
    timeframe: str = "today 12-m",
) -> pd.DataFrame:
    """
    Fetch Google Trends time-series for tourism-related keywords (Greece).
    Returns DataFrame: date, keyword columns, demand_index (mean).
    """
    from pytrends.request import TrendReq

    if keywords is None:
        keywords = ["ξενοδοχεία", "διακοπές", "παραλίες", "Airbnb Greece"]

    try:
        pytrends = TrendReq(hl="el-GR", tz=120)
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="GR")
        df = pytrends.interest_over_time()

        if df.empty:
            log.warning("Google Trends returned empty data")
            return pd.DataFrame()

        df = df.reset_index()
        kw_cols = [c for c in df.columns if c not in ("date", "isPartial")]
        df["demand_index"] = df[kw_cols].mean(axis=1).round(1)
        log.info(f"Google Trends: {len(df)} rows, keywords={kw_cols}")
        return df
    except Exception as e:
        log.warning(f"Google Trends failed: {e}")
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. OPEN-METEO: Current Weather per Municipality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEATHER_CODE_LABELS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


def fetch_weather() -> pd.DataFrame:
    """
    Fetch current weather + 7-day forecast for all municipalities via Open-Meteo.
    Returns DataFrame: dimos, temperature, humidity, weather, wind,
                       avg_sunshine_hrs, avg_uv, weather_score (0-100).
    """
    rows = []
    lats = [m["lat"] for m in MUNICIPALITIES.values()]
    lons = [m["lon"] for m in MUNICIPALITIES.values()]
    names = list(MUNICIPALITIES.keys())

    # Batch request (Open-Meteo supports comma-separated coords)
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
            "current": "temperature_2m,relative_humidity_2m,weathercode,windspeed_10m",
            "daily": "sunshine_duration,uv_index_max,temperature_2m_max",
            "timezone": "Europe/Athens",
            "forecast_days": 7,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()

        # If single location, wrap in list
        if isinstance(data, dict) and "current" in data:
            data = [data]

        for i, name in enumerate(names):
            d = data[i]
            cur = d.get("current", {})
            daily = d.get("daily", {})

            temp = cur.get("temperature_2m", None)
            humidity = cur.get("relative_humidity_2m", None)
            wcode = cur.get("weathercode", 0)
            wind = cur.get("windspeed_10m", None)

            sunshine_secs = daily.get("sunshine_duration", [0])
            avg_sunshine_hrs = np.mean(sunshine_secs) / 3600 if sunshine_secs else 0
            avg_uv = np.mean(daily.get("uv_index_max", [0]))
            avg_temp_max = np.mean(daily.get("temperature_2m_max", [20]))

            # Weather attractiveness score (0-100)
            # Ideal: 25-30C, low wind, high sunshine, no rain
            temp_score = max(0, 100 - abs(avg_temp_max - 27) * 4)
            sun_score = min(avg_sunshine_hrs / 14 * 100, 100)
            wind_penalty = max(0, (wind or 0) - 20) * 2
            rain_penalty = 30 if wcode >= 61 else 0
            weather_score = round(np.clip(
                temp_score * 0.35 + sun_score * 0.40 - wind_penalty - rain_penalty,
                0, 100
            ), 1)

            rows.append({
                "dimos": name,
                "temperature": temp,
                "humidity": humidity,
                "weather_desc": WEATHER_CODE_LABELS.get(wcode, f"Code {wcode}"),
                "wind_kmh": wind,
                "avg_sunshine_hrs": round(avg_sunshine_hrs, 1),
                "avg_uv": round(avg_uv, 1),
                "avg_temp_max_7d": round(avg_temp_max, 1),
                "weather_score": weather_score,
            })

        log.info(f"Weather: {len(rows)} municipalities fetched")
    except Exception as e:
        log.warning(f"Open-Meteo failed: {e}")
        for name in names:
            rows.append({
                "dimos": name, "temperature": None, "humidity": None,
                "weather_desc": "N/A", "wind_kmh": None,
                "avg_sunshine_hrs": 0, "avg_uv": 0,
                "avg_temp_max_7d": 20, "weather_score": 50,
            })

    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPOSITE INDEX: Build Resilience Snapshot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _normalize(series: pd.Series, lower=None, upper=None) -> pd.Series:
    """Min-max normalize a series to 0-100."""
    lo = lower if lower is not None else series.min()
    hi = upper if upper is not None else series.max()
    if hi == lo:
        return pd.Series(50, index=series.index)
    return ((series - lo) / (hi - lo) * 100).clip(0, 100)


def build_resilience_snapshot(
    df_eurostat: pd.DataFrame,
    df_trends: pd.DataFrame,
    df_weather: pd.DataFrame,
    shock_demand: float = 0,
    shock_arrivals: float = 0,
) -> pd.DataFrame:
    """
    Merge all real data sources and compute composite Resilience Score.
    Returns one row per municipality with all indicators and final score.
    """
    # Start with municipality reference
    rows = []
    for name, meta in MUNICIPALITIES.items():
        rows.append({
            "dimos": name,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "nuts2": meta["nuts2"],
            "region": meta["region"],
        })
    df = pd.DataFrame(rows)

    # ── Eurostat: latest year per NUTS-2 ──
    if not df_eurostat.empty:
        latest_year = df_eurostat["year"].max()
        prev_year = latest_year - 1
        euro_latest = df_eurostat[df_eurostat["year"] == latest_year].copy()
        euro_prev = df_eurostat[df_eurostat["year"] == prev_year].copy()

        df = df.merge(
            euro_latest[["nuts2", "arrivals", "nights", "avg_stay", "yoy_arrivals_%"]],
            on="nuts2", how="left",
        )
        # Add previous year for comparison
        if not euro_prev.empty:
            df = df.merge(
                euro_prev[["nuts2", "arrivals"]].rename(columns={"arrivals": "arrivals_prev"}),
                on="nuts2", how="left",
            )
        df["eurostat_year"] = latest_year
    else:
        df["arrivals"] = np.nan
        df["nights"] = np.nan
        df["avg_stay"] = np.nan
        df["yoy_arrivals_%"] = np.nan
        df["eurostat_year"] = None

    # Multiple municipalities share NUTS2 — proportional split heuristic
    # (Eurostat is region-level; we distribute proportionally by known tourism share)
    tourism_shares = {
        "Χανιά": 0.35, "Ηράκλειο": 0.65,                       # EL43 Kriti
        "Μύκονος": 0.22, "Σαντορίνη": 0.28, "Ρόδος": 0.50,    # EL42 Notio Aigaio
        "Κέρκυρα": 1.0,                                         # EL62 Ionia Nisia
        "Αθήνα": 1.0,                                           # EL30 Attiki
        "Θεσσαλονίκη": 1.0,                                     # EL52 K. Makedonia
        "Ναύπλιο": 0.30,                                        # EL65 Peloponnisos
        "Ιωάννινα": 0.50,                                       # EL54 Ipeiros
        "Λάρισα": 0.40,                                         # EL61 Thessalia
        "Πάτρα": 0.45,                                          # EL63 Dytiki Ellada
        "Αλεξανδρούπολη": 0.35,                                 # EL51 An. Makedonia & Thraki
        "Μυτιλήνη": 0.40,                                       # EL41 Voreio Aigaio
    }
    for col in ["arrivals", "nights"]:
        if col in df.columns:
            df[col] = df.apply(
                lambda r: r[col] * tourism_shares.get(r["dimos"], 0.5) if pd.notna(r[col]) else np.nan,
                axis=1,
            )

    # ── Google Trends: national demand index ──
    if not df_trends.empty and "demand_index" in df_trends.columns:
        latest_demand = df_trends["demand_index"].iloc[-1]
        avg_demand_3m = df_trends["demand_index"].tail(12).mean()  # ~3 months
        demand_trend = ((latest_demand / avg_demand_3m) - 1) * 100 if avg_demand_3m > 0 else 0
    else:
        latest_demand = 50
        demand_trend = 0

    df["demand_index"] = round(latest_demand * (1 + shock_demand / 100), 1)
    df["demand_trend_%"] = round(demand_trend, 1)

    # ── Weather ──
    if not df_weather.empty:
        df = df.merge(df_weather, on="dimos", how="left")
    else:
        df["weather_score"] = 50
        df["temperature"] = None

    # Apply shocks
    if shock_arrivals != 0:
        for col in ["arrivals", "nights"]:
            if col in df.columns:
                df[col] = df[col] * (1 + shock_arrivals / 100)

    # ── Normalize components to 0-100 ──
    df["norm_arrivals"] = _normalize(df["arrivals"]) if "arrivals" in df.columns else 50
    df["norm_nights"] = _normalize(df["nights"]) if "nights" in df.columns else 50
    df["norm_demand"] = df["demand_index"].clip(0, 100)
    df["norm_weather"] = df["weather_score"] if "weather_score" in df.columns else 50
    df["norm_avg_stay"] = _normalize(df["avg_stay"], lower=1, upper=8) if "avg_stay" in df.columns else 50

    # ── Two-Stage Pillar Entropy Weights (data-driven) ──
    norm_cols = ["norm_arrivals", "norm_nights", "norm_demand", "norm_weather", "norm_avg_stay"]

    try:
        pillar_result = compute_pillar_weights(df[norm_cols])
        weights = pillar_result["weights"]
        df.attrs["pillar_result"] = pillar_result
        df.attrs["weighting_method"] = "Two-Stage Pillar Entropy"
        log.info("Using two-stage pillar entropy weights")
    except Exception as e:
        log.warning(f"Pillar entropy failed ({e}), using fallback weights")
        weights = WEIGHTS_FALLBACK
        df.attrs["pillar_result"] = None
        df.attrs["weighting_method"] = "Static (fallback)"

    df.attrs["weights"] = weights

    # ── Composite Score ──
    df["resilience_score"] = (
        weights["arrivals"] * df["norm_arrivals"]
        + weights["nights"] * df["norm_nights"]
        + weights["demand"] * df["norm_demand"]
        + weights["weather"] * df["norm_weather"]
        + weights["avg_stay"] * df["norm_avg_stay"]
    ).round(1)

    # Weighted contributions (for waterfall chart)
    df["w_arrivals"] = (weights["arrivals"] * df["norm_arrivals"]).round(1)
    df["w_nights"] = (weights["nights"] * df["norm_nights"]).round(1)
    df["w_demand"] = (weights["demand"] * df["norm_demand"]).round(1)
    df["w_weather"] = (weights["weather"] * df["norm_weather"]).round(1)
    df["w_avg_stay"] = (weights["avg_stay"] * df["norm_avg_stay"]).round(1)

    # Zone classification
    df["zone"] = df["resilience_score"].apply(
        lambda s: "ΥΨΗΛΗ" if s >= 65 else "ΜΕΣΑΙΑ" if s >= 45 else "ΧΑΜΗΛΗ"
    )
    df["color"] = df["zone"].map({"ΥΨΗΛΗ": "#27AE60", "ΜΕΣΑΙΑ": "#E67E22", "ΧΑΜΗΛΗ": "#C0392B"})

    df = df.sort_values("resilience_score", ascending=False).reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience: fetch everything in one call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_all(shock_demand: float = 0, shock_arrivals: float = 0):
    """Fetch all data sources and return (snapshot_df, eurostat_df, trends_df, weather_df)."""
    df_euro = fetch_eurostat_tourism()
    df_trends = fetch_google_trends()
    df_weather = fetch_weather()
    snapshot = build_resilience_snapshot(
        df_euro, df_trends, df_weather,
        shock_demand=shock_demand,
        shock_arrivals=shock_arrivals,
    )
    return snapshot, df_euro, df_trends, df_weather


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    snap, euro, trends, weather = fetch_all()

    print("\n=== TWO-STAGE PILLAR ENTROPY WEIGHTS ===")
    pr = snap.attrs.get("pillar_result")
    if pr:
        print(f"\n{'Pillar':<20s} {'Weight':>8s}")
        print("-" * 30)
        for pname, pdef in PILLARS.items():
            print(f"{PILLAR_NAMES[pname]:<20s} {pdef['pillar_weight']:8.2%}")

        print(f"\n{'Variable':<22s} {'Pillar':<14s} {'Within w':>10s} {'Final w':>10s}")
        print("-" * 60)
        for pname, pdet in pr["pillar_details"].items():
            for var, ww in pdet["within_weights"].items():
                fw = pr["weights"][var]
                print(f"{COMPONENT_NAMES.get(var, var):<22s} "
                      f"{PILLAR_NAMES[pname]:<14s} "
                      f"{ww:10.4f} "
                      f"{fw:10.4f}")
        print(f"{'SUM':<22s} {'':>14s} {'':>10s} {sum(pr['weights'].values()):10.4f}")

    print("\n=== RESILIENCE SNAPSHOT ===")
    print(snap[["dimos", "region", "arrivals", "nights", "demand_index",
                "weather_score", "resilience_score", "zone"]].to_string(index=False))
    print(f"\nWeighting: {snap.attrs.get('weighting_method', 'N/A')}")
    print(f"Eurostat year: {snap['eurostat_year'].iloc[0]}")
    print(f"Timestamp: {datetime.now()}")
