"""
ResilienceIQ Tier 0 — Real Data Fetcher (v2: 3x3 Balanced Pillars)
Fetches live data from:
  1. Eurostat REST API  — Tourism arrivals & nights (annual), seasonality (monthly)
  2. Google Trends      — Tourism demand keywords (time-series)
  3. Open-Meteo         — Weather, Air Quality, Marine (per municipality)
  4. Wikimedia          — Wikipedia pageviews (destination awareness)
All sources are free and require no API key.
"""

import pandas as pd
import numpy as np
import requests
import warnings
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Municipality reference table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUNICIPALITIES = {
    "Χανιά":            {"lat": 35.5138, "lon": 24.0180, "nuts2": "EL43", "region": "Κρήτη",
                         "wiki_en": "Chania", "coastal": True},
    "Ρόδος":            {"lat": 36.4341, "lon": 28.2176, "nuts2": "EL42", "region": "Ν. Αιγαίο",
                         "wiki_en": "Rhodes", "coastal": True},
    "Μύκονος":          {"lat": 37.4467, "lon": 25.3289, "nuts2": "EL42", "region": "Ν. Αιγαίο",
                         "wiki_en": "Mykonos", "coastal": True},
    "Σαντορίνη":        {"lat": 36.3932, "lon": 25.4615, "nuts2": "EL42", "region": "Ν. Αιγαίο",
                         "wiki_en": "Santorini", "coastal": True},
    "Κέρκυρα":          {"lat": 39.6243, "lon": 19.9217, "nuts2": "EL62", "region": "Ιόνια Νησιά",
                         "wiki_en": "Corfu", "coastal": True},
    "Αθήνα":            {"lat": 37.9838, "lon": 23.7275, "nuts2": "EL30", "region": "Αττική",
                         "wiki_en": "Athens", "coastal": True},
    "Θεσσαλονίκη":      {"lat": 40.6401, "lon": 22.9444, "nuts2": "EL52", "region": "Κ. Μακεδονία",
                         "wiki_en": "Thessaloniki", "coastal": True},
    "Ηράκλειο":         {"lat": 35.3387, "lon": 25.1442, "nuts2": "EL43", "region": "Κρήτη",
                         "wiki_en": "Heraklion", "coastal": True},
    "Ναύπλιο":          {"lat": 37.5675, "lon": 22.8017, "nuts2": "EL65", "region": "Πελοπόννησος",
                         "wiki_en": "Nafplio", "coastal": True},
    "Ιωάννινα":         {"lat": 39.6650, "lon": 20.8537, "nuts2": "EL54", "region": "Ήπειρος",
                         "wiki_en": "Ioannina", "coastal": False},
    "Λάρισα":           {"lat": 39.6390, "lon": 22.4191, "nuts2": "EL61", "region": "Θεσσαλία",
                         "wiki_en": "Larissa", "coastal": False},
    "Πάτρα":            {"lat": 38.2466, "lon": 21.7346, "nuts2": "EL63", "region": "Δ. Ελλάδα",
                         "wiki_en": "Patras", "coastal": True},
    "Αλεξανδρούπολη":   {"lat": 40.8469, "lon": 25.8743, "nuts2": "EL51", "region": "Α. Μακεδονία & Θράκη",
                         "wiki_en": "Alexandroupolis", "coastal": True},
    "Μυτιλήνη":         {"lat": 39.1043, "lon": 26.5518, "nuts2": "EL41", "region": "Β. Αιγαίο",
                         "wiki_en": "Mytilene", "coastal": True},
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3x3 Balanced Pillar definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PILLARS = {
    "supply": {
        "label": "Supply Capacity",
        "indicators": ["arrivals", "nights", "avg_stay"],
    },
    "demand": {
        "label": "Demand Pressure",
        "indicators": ["dest_interest", "wiki_views", "seasonality"],
    },
    "environment": {
        "label": "Environmental",
        "indicators": ["weather", "air_quality", "coastal"],
    },
}

COMPONENT_NAMES = {
    "arrivals":       "Tourism Arrivals",
    "nights":         "Nights Spent",
    "avg_stay":       "Average Stay Duration",
    "dest_interest":  "Destination Search Interest",
    "wiki_views":     "Destination Awareness",
    "seasonality":    "Seasonality Balance",
    "weather":        "Weather Attractiveness",
    "air_quality":    "Air Quality",
    "coastal":        "Coastal Comfort",
}

# Google Trends search terms per destination (worldwide queries)
DEST_SEARCH_TERMS = {
    "Χανιά":            "Chania Crete",
    "Ρόδος":            "Rhodes Greece",
    "Μύκονος":          "Mykonos",
    "Σαντορίνη":        "Santorini",
    "Κέρκυρα":          "Corfu Greece",
    "Αθήνα":            "Athens Greece",
    "Θεσσαλονίκη":      "Thessaloniki",
    "Ηράκλειο":         "Heraklion Crete",
    "Ναύπλιο":          "Nafplio Greece",
    "Ιωάννινα":         "Ioannina Greece",
    "Λάρισα":           "Larissa Greece",
    "Πάτρα":            "Patras Greece",
    "Αλεξανδρούπολη":   "Alexandroupolis Greece",
    "Μυτιλήνη":         "Lesbos Greece",
}

PILLAR_NAMES = {
    "supply":      "Supply Capacity",
    "demand":      "Demand Pressure",
    "environment": "Environmental",
}

# Fallback weights: equal pillar (1/3) x equal within (1/3) = 1/9 each
WEIGHTS_FALLBACK = {k: round(1.0 / 9, 4) for k in COMPONENT_NAMES}

# Known seasonality patterns per NUTS-2 (fallback if monthly Eurostat fails)
# Score 0-100: high = low seasonality = year-round tourism = more resilient
SEASONALITY_FALLBACK = {
    "EL30": 82, "EL41": 35, "EL42": 25, "EL43": 32, "EL51": 45,
    "EL52": 72, "EL54": 55, "EL61": 50, "EL62": 28, "EL63": 52, "EL65": 48,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHANNON ENTROPY WEIGHTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_entropy_weights(norm_matrix: pd.DataFrame) -> dict:
    """
    Compute Shannon Entropy weights from a normalized data matrix.
    Steps:
        1. p_ij = x_ij / Sigma_i(x_ij)
        2. E_j  = -(1/ln(n)) x Sigma_i[p_ij x ln(p_ij)]
        3. d_j  = 1 - E_j
        4. w_j  = d_j / Sigma_k(d_k)
    """
    df = norm_matrix.copy()
    variables = list(df.columns)
    n = len(df)

    epsilon = 1e-10
    df = df.clip(lower=epsilon)

    col_sums = df.sum(axis=0)
    p = df.div(col_sums, axis=1)

    k = 1.0 / np.log(n)
    entropy = {}
    for var in variables:
        p_col = p[var]
        e_j = -k * (p_col * np.log(p_col)).sum()
        entropy[var] = round(e_j, 6)

    diversity = {var: round(1 - entropy[var], 6) for var in variables}

    d_sum = sum(diversity.values())
    if d_sum == 0:
        weights = {var: round(1.0 / len(variables), 4) for var in variables}
    else:
        weights = {var: round(diversity[var] / d_sum, 4) for var in variables}

    log.info(f"Shannon Entropy weights: {weights}")
    return {
        "weights": weights,
        "entropy": entropy,
        "diversity": diversity,
        "proportions": p,
    }


def compute_pillar_weights(norm_matrix: pd.DataFrame) -> dict:
    """
    Two-stage weighting with cross-pillar entropy.

    Stage 1 — Within-pillar Shannon Entropy on each pillar's 3 indicators.
    Stage 2 — Cross-pillar Shannon Entropy on the 3 pillar composite scores.

    Final weight = pillar_weight x within_pillar_weight.
    """
    key_map = {
        "norm_arrivals": "arrivals", "norm_nights": "nights", "norm_avg_stay": "avg_stay",
        "norm_dest_interest": "dest_interest", "norm_wiki_views": "wiki_views",
        "norm_seasonality": "seasonality",
        "norm_weather": "weather", "norm_air_quality": "air_quality", "norm_coastal": "coastal",
    }
    df = norm_matrix.rename(columns=key_map)

    pillar_scores = {}
    pillar_details = {}
    within_weights_all = {}

    # Stage 1: Within-pillar entropy
    for pillar_name, pillar_def in PILLARS.items():
        indicators = pillar_def["indicators"]
        sub_df = df[indicators]
        try:
            ent = compute_entropy_weights(sub_df)
            within_w = ent["weights"]
            within_weights_all[pillar_name] = within_w

            # Pillar composite score per municipality
            score = sum(within_w[v] * df[v] for v in indicators)
            pillar_scores[pillar_name] = score

            pillar_details[pillar_name] = {
                "within_weights": within_w,
                "entropy": ent["entropy"],
                "diversity": ent["diversity"],
            }
        except Exception as e:
            log.warning(f"Within-pillar entropy failed for {pillar_name}: {e}")
            uniform = round(1.0 / len(indicators), 4)
            within_w = {v: uniform for v in indicators}
            within_weights_all[pillar_name] = within_w
            score = sum(uniform * df[v] for v in indicators)
            pillar_scores[pillar_name] = score
            pillar_details[pillar_name] = {
                "within_weights": within_w,
                "entropy": {v: None for v in indicators},
                "diversity": {v: None for v in indicators},
            }

    # Stage 2: Cross-pillar entropy
    pillar_df = pd.DataFrame(pillar_scores)
    try:
        cross_ent = compute_entropy_weights(pillar_df)
        pillar_w = cross_ent["weights"]
        log.info(f"Cross-pillar entropy weights: {pillar_w}")
    except Exception as e:
        log.warning(f"Cross-pillar entropy failed ({e}), using equal weights")
        pillar_w = {p: round(1.0 / len(PILLARS), 4) for p in PILLARS}
        cross_ent = None

    # Combine: final_weight = pillar_weight x within_weight
    flat_weights = {}
    for pillar_name, pw in pillar_w.items():
        pillar_details[pillar_name]["pillar_weight"] = pw
        for var, ww in within_weights_all[pillar_name].items():
            flat_weights[var] = round(pw * ww, 4)

    # Ensure sum = 1 (rounding fix)
    total = sum(flat_weights.values())
    if total > 0 and abs(total - 1.0) > 0.001:
        flat_weights = {k: round(v / total, 4) for k, v in flat_weights.items()}

    log.info(f"Two-stage pillar weights: {flat_weights}")
    return {
        "weights": flat_weights,
        "pillar_details": pillar_details,
        "cross_pillar_entropy": cross_ent,
        "pillar_weights": pillar_w,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. EUROSTAT: Tourism Arrivals & Nights (NUTS-2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _eurostat_decode(data: dict, targets: set) -> dict:
    """Decode Eurostat JSON-stat flat format into {geo: {time: value}}."""
    dims = data["dimension"]
    dim_ids = data["id"]
    dim_sizes = data["size"]
    geos = dims["geo"]["category"]["index"]
    times = dims["time"]["category"]["index"]
    vals = data["value"]

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
    """Fetch annual tourism arrivals AND nights from Eurostat for Greek NUTS-2."""
    targets = {m["nuts2"] for m in MUNICIPALITIES.values()}

    dfs = {}
    for dataset, col_name in [("tour_occ_arn2", "arrivals"), ("tour_occ_nin2", "nights")]:
        url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
        params = {
            "format": "JSON", "lang": "en", "sinceTimePeriod": since,
            "unit": "NR", "nace_r2": "I551-I553", "c_resid": "TOTAL",
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

    if dfs["arrivals"].empty and dfs["nights"].empty:
        return pd.DataFrame()

    df = pd.merge(dfs["arrivals"], dfs["nights"], on=["nuts2", "year"], how="outer")
    df = df.sort_values(["nuts2", "year"])
    df["avg_stay"] = (df["nights"] / df["arrivals"]).round(2)
    df["yoy_arrivals_%"] = df.groupby("nuts2")["arrivals"].pct_change() * 100
    df["yoy_arrivals_%"] = df["yoy_arrivals_%"].round(1)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. EUROSTAT: Seasonality Index (Monthly Nights)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_eurostat_seasonality() -> pd.DataFrame:
    """
    Compute seasonality index per NUTS-2 from Eurostat monthly nights.
    Score 0-100: high = low seasonality = more resilient.
    Falls back to known patterns if API fails.
    """
    targets = {m["nuts2"] for m in MUNICIPALITIES.values()}
    latest_year = datetime.now().year - 1

    try:
        url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/tour_occ_nim"
        params = {
            "format": "JSON", "lang": "en",
            "sinceTimePeriod": f"{latest_year}M01",
            "untilTimePeriod": f"{latest_year}M12",
            "unit": "NR", "nace_r2": "I551-I553", "c_resid": "TOTAL",
        }
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        decoded = _eurostat_decode(r.json(), targets)

        rows = []
        for nuts2, months in decoded.items():
            if len(months) < 6:
                rows.append({"nuts2": nuts2, "seasonality_score": SEASONALITY_FALLBACK.get(nuts2, 50)})
                continue
            total = sum(months.values())
            if total == 0:
                rows.append({"nuts2": nuts2, "seasonality_score": 50})
                continue
            shares = [v / total for v in months.values()]
            hhi = sum(s ** 2 for s in shares)
            n_m = len(shares)
            hhi_min = 1.0 / n_m
            score = round((1 - hhi) / (1 - hhi_min) * 100, 1) if hhi_min < 1 else 50
            rows.append({"nuts2": nuts2, "seasonality_score": np.clip(score, 0, 100)})

        if rows:
            log.info(f"Eurostat seasonality: {len(rows)} regions (live)")
            # Fill missing regions from fallback
            found = {r["nuts2"] for r in rows}
            for nuts2 in targets - found:
                rows.append({"nuts2": nuts2, "seasonality_score": SEASONALITY_FALLBACK.get(nuts2, 50)})
            return pd.DataFrame(rows)

    except Exception as e:
        log.warning(f"Eurostat monthly seasonality failed: {e}, using fallback")

    rows = []
    for nuts2 in targets:
        rows.append({"nuts2": nuts2, "seasonality_score": SEASONALITY_FALLBACK.get(nuts2, 50)})
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GOOGLE TRENDS: Tourism Demand Keywords
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_google_trends(
    keywords: Optional[list] = None,
    timeframe: str = "today 12-m",
) -> pd.DataFrame:
    """Fetch Google Trends time-series for tourism keywords (Greece)."""
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
# 3b. GOOGLE TRENDS: Destination Search Interest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_destination_trends(timeframe: str = "today 3-m") -> pd.DataFrame:
    """
    Fetch Google Trends search interest per destination (worldwide).
    Uses batched queries with Athens as normalization anchor.
    Returns DataFrame: dimos, dest_interest (0-100 relative to top destination).
    Falls back to Wikipedia pageviews proportional if Google Trends fails.
    """
    import time as _time
    from pytrends.request import TrendReq

    anchor_dimos = "Αθήνα"
    anchor_term = DEST_SEARCH_TERMS[anchor_dimos]
    others = [(d, t) for d, t in DEST_SEARCH_TERMS.items() if d != anchor_dimos]

    try:
        pytrends = TrendReq(hl="en-US", tz=120)
        results = {}

        # Process in batches of 4 + anchor
        for i in range(0, len(others), 4):
            batch = others[i:i + 4]
            keywords = [anchor_term] + [t for _, t in batch]

            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="")
            df = pytrends.interest_over_time()

            if not df.empty:
                recent = df.tail(4).mean()
                anchor_val = max(recent.get(anchor_term, 1), 1)
                for dimos, term in batch:
                    val = recent.get(term, 0)
                    results[dimos] = round(val / anchor_val * 100, 1)

            if i + 4 < len(others):
                _time.sleep(2)  # Rate limit courtesy

        results[anchor_dimos] = 100.0

        rows = [{"dimos": d, "dest_interest": results.get(d, 0)} for d in DEST_SEARCH_TERMS]
        log.info(f"Destination Trends: {len(rows)} fetched via Google Trends")
        return pd.DataFrame(rows)

    except Exception as e:
        log.warning(f"Destination Trends (Google) failed: {e}")
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. WIKIPEDIA PAGEVIEWS: Destination Awareness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_wikipedia_pageviews() -> pd.DataFrame:
    """
    Fetch monthly Wikipedia pageviews for each destination (en + de + fr).
    Destination-specific proxy for international tourism awareness.
    """
    today = datetime.now()
    end_date = (today.replace(day=1) - timedelta(days=1))
    start_date = (end_date.replace(day=1) - timedelta(days=75)).replace(day=1)
    start_str = start_date.strftime("%Y%m01")
    end_str = end_date.strftime("%Y%m%d")

    rows = []
    for dimos, meta in MUNICIPALITIES.items():
        article = meta["wiki_en"]
        total_views = 0
        for lang in ["en", "de", "fr"]:
            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"{lang}.wikipedia.org/all-access/all-agents/{article}/monthly/"
                f"{start_str}/{end_str}"
            )
            try:
                r = requests.get(url, headers={"User-Agent": "ResilienceIQ/1.0"}, timeout=10)
                if r.status_code == 200:
                    for item in r.json().get("items", []):
                        total_views += item.get("views", 0)
            except Exception:
                pass

        rows.append({"dimos": dimos, "wiki_views": total_views})

    log.info(f"Wikipedia pageviews: {len(rows)} destinations fetched")
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. OPEN-METEO: Weather Attractiveness
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
    """Fetch current weather + 7-day forecast for all municipalities."""
    rows = []
    lats = [m["lat"] for m in MUNICIPALITIES.values()]
    lons = [m["lon"] for m in MUNICIPALITIES.values()]
    names = list(MUNICIPALITIES.keys())

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
        if isinstance(data, dict) and "current" in data:
            data = [data]

        for i, name in enumerate(names):
            d = data[i]
            cur = d.get("current", {})
            daily = d.get("daily", {})
            temp = cur.get("temperature_2m")
            humidity = cur.get("relative_humidity_2m")
            wcode = cur.get("weathercode", 0)
            wind = cur.get("windspeed_10m")
            sunshine_secs = daily.get("sunshine_duration", [0])
            avg_sunshine_hrs = np.mean(sunshine_secs) / 3600 if sunshine_secs else 0
            avg_uv = np.mean(daily.get("uv_index_max", [0]))
            avg_temp_max = np.mean(daily.get("temperature_2m_max", [20]))

            temp_score = max(0, 100 - abs(avg_temp_max - 27) * 4)
            sun_score = min(avg_sunshine_hrs / 14 * 100, 100)
            wind_penalty = max(0, (wind or 0) - 20) * 2
            rain_penalty = 30 if wcode >= 61 else 0
            weather_score = round(np.clip(
                temp_score * 0.35 + sun_score * 0.40 - wind_penalty - rain_penalty, 0, 100), 1)

            rows.append({
                "dimos": name, "temperature": temp, "humidity": humidity,
                "weather_desc": WEATHER_CODE_LABELS.get(wcode, f"Code {wcode}"),
                "wind_kmh": wind, "avg_sunshine_hrs": round(avg_sunshine_hrs, 1),
                "avg_uv": round(avg_uv, 1), "avg_temp_max_7d": round(avg_temp_max, 1),
                "weather_score": weather_score,
            })
        log.info(f"Weather: {len(rows)} municipalities fetched")
    except Exception as e:
        log.warning(f"Open-Meteo forecast failed: {e}")
        for name in names:
            rows.append({
                "dimos": name, "temperature": None, "humidity": None,
                "weather_desc": "N/A", "wind_kmh": None,
                "avg_sunshine_hrs": 0, "avg_uv": 0, "avg_temp_max_7d": 20,
                "weather_score": 50,
            })
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. OPEN-METEO: Air Quality Index
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_air_quality() -> pd.DataFrame:
    """Fetch European AQI for all municipalities. Score: lower AQI = higher score."""
    lats = [m["lat"] for m in MUNICIPALITIES.values()]
    lons = [m["lon"] for m in MUNICIPALITIES.values()]
    names = list(MUNICIPALITIES.keys())

    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
            "current": "european_aqi,pm2_5,pm10",
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "current" in data:
            data = [data]

        rows = []
        for i, name in enumerate(names):
            cur = data[i].get("current", {})
            eaqi = cur.get("european_aqi", 50)
            # EAQI: 0-20 Good, 20-40 Fair, 40-60 Moderate, 60-80 Poor, 80+ Very Poor
            aqi_score = round(max(0, min(100, 100 - eaqi)), 1)
            rows.append({
                "dimos": name, "european_aqi": eaqi,
                "pm2_5": cur.get("pm2_5"), "pm10": cur.get("pm10"),
                "air_quality_score": aqi_score,
            })
        log.info(f"Air Quality: {len(rows)} municipalities fetched")
        return pd.DataFrame(rows)

    except Exception as e:
        log.warning(f"Air Quality API failed: {e}")
        return pd.DataFrame([{
            "dimos": n, "european_aqi": None, "pm2_5": None,
            "pm10": None, "air_quality_score": 50,
        } for n in names])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. OPEN-METEO MARINE: Coastal Comfort Index
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_marine() -> pd.DataFrame:
    """
    Fetch wave data for coastal municipalities from Open-Meteo Marine API.
    Inland cities receive a fixed low score (15).
    """
    names = list(MUNICIPALITIES.keys())
    coastal_names = [n for n in names if MUNICIPALITIES[n].get("coastal", True)]
    coastal_scores = {}

    if coastal_names:
        lats = [MUNICIPALITIES[n]["lat"] for n in coastal_names]
        lons = [MUNICIPALITIES[n]["lon"] for n in coastal_names]
        try:
            r = requests.get("https://marine-api.open-meteo.com/v1/marine", params={
                "latitude": ",".join(str(x) for x in lats),
                "longitude": ",".join(str(x) for x in lons),
                "daily": "wave_height_max",
                "timezone": "Europe/Athens",
                "forecast_days": 7,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "daily" in data:
                data = [data]

            for i, name in enumerate(coastal_names):
                daily = data[i].get("daily", {})
                wave_heights = daily.get("wave_height_max", [1.0])
                # Filter out None values
                wave_heights = [w for w in wave_heights if w is not None]
                avg_wave = np.mean(wave_heights) if wave_heights else 1.0
                # Calm sea = high score: 0m=100, 1m=50, 2m+=0
                coastal_scores[name] = round(max(0, min(100, 100 - avg_wave * 50)), 1)

            log.info(f"Marine: {len(coastal_scores)} coastal municipalities fetched")
        except Exception as e:
            log.warning(f"Marine API failed: {e}")

    rows = []
    for name in names:
        if MUNICIPALITIES[name].get("coastal", True):
            score = coastal_scores.get(name, 50)
        else:
            score = 15  # Inland — minimal coastal tourism benefit
        rows.append({"dimos": name, "coastal_score": score})
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
    df_dest_trends: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_air_quality: pd.DataFrame,
    df_marine: pd.DataFrame,
    df_wiki: pd.DataFrame,
    df_seasonality: pd.DataFrame,
    shock_demand: float = 0,
    shock_arrivals: float = 0,
) -> pd.DataFrame:
    """
    Merge all data sources and compute 3x3 composite Resilience Score.
    9 indicators across 3 pillars, two-stage entropy weighting.
    """
    # Start with municipality reference
    rows = []
    for name, meta in MUNICIPALITIES.items():
        rows.append({
            "dimos": name, "lat": meta["lat"], "lon": meta["lon"],
            "nuts2": meta["nuts2"], "region": meta["region"],
        })
    df = pd.DataFrame(rows)

    # ── SUPPLY PILLAR: Eurostat annual ──
    if not df_eurostat.empty:
        latest_year = df_eurostat["year"].max()
        euro_latest = df_eurostat[df_eurostat["year"] == latest_year].copy()
        euro_prev = df_eurostat[df_eurostat["year"] == latest_year - 1].copy()

        df = df.merge(
            euro_latest[["nuts2", "arrivals", "nights", "avg_stay", "yoy_arrivals_%"]],
            on="nuts2", how="left",
        )
        if not euro_prev.empty:
            df = df.merge(
                euro_prev[["nuts2", "arrivals"]].rename(columns={"arrivals": "arrivals_prev"}),
                on="nuts2", how="left",
            )
        df["eurostat_year"] = latest_year
    else:
        for col in ["arrivals", "nights", "avg_stay", "yoy_arrivals_%"]:
            df[col] = np.nan
        df["eurostat_year"] = None

    # Tourism share split for shared NUTS-2 regions
    tourism_shares = {
        "Χανιά": 0.35, "Ηράκλειο": 0.65,
        "Μύκονος": 0.22, "Σαντορίνη": 0.28, "Ρόδος": 0.50,
        "Κέρκυρα": 1.0, "Αθήνα": 1.0, "Θεσσαλονίκη": 1.0,
        "Ναύπλιο": 0.30, "Ιωάννινα": 0.50, "Λάρισα": 0.40,
        "Πάτρα": 0.45, "Αλεξανδρούπολη": 0.35, "Μυτιλήνη": 0.40,
    }
    for col in ["arrivals", "nights"]:
        if col in df.columns:
            df[col] = df.apply(
                lambda r: r[col] * tourism_shares.get(r["dimos"], 0.5) if pd.notna(r[col]) else np.nan,
                axis=1,
            )

    # ── DEMAND PILLAR: Destination Search Interest (per-city Google Trends) ──
    if not df_dest_trends.empty and "dest_interest" in df_dest_trends.columns:
        df = df.merge(df_dest_trends, on="dimos", how="left")
        df["dest_interest"] = df["dest_interest"].fillna(0)
    else:
        df["dest_interest"] = 0  # Will be overridden by wiki fallback below

    # ── DEMAND PILLAR: Wikipedia Pageviews ──
    if not df_wiki.empty:
        df = df.merge(df_wiki, on="dimos", how="left")
    else:
        df["wiki_views"] = 0

    # ── DEMAND PILLAR: Seasonality ──
    if not df_seasonality.empty:
        df = df.merge(df_seasonality, on="nuts2", how="left")
    else:
        df["seasonality_score"] = df["nuts2"].map(SEASONALITY_FALLBACK).fillna(50)

    # ── ENVIRONMENT PILLAR: Weather ──
    if not df_weather.empty:
        df = df.merge(df_weather, on="dimos", how="left")
    else:
        df["weather_score"] = 50
        df["temperature"] = None

    # ── ENVIRONMENT PILLAR: Air Quality ──
    if not df_air_quality.empty:
        df = df.merge(df_air_quality, on="dimos", how="left")
    else:
        df["air_quality_score"] = 50

    # ── ENVIRONMENT PILLAR: Marine / Coastal ──
    if not df_marine.empty:
        df = df.merge(df_marine, on="dimos", how="left")
    else:
        df["coastal_score"] = df["dimos"].apply(
            lambda d: 50 if MUNICIPALITIES[d].get("coastal", True) else 15)

    # Apply shocks
    if shock_arrivals != 0:
        for col in ["arrivals", "nights"]:
            if col in df.columns:
                df[col] = df[col] * (1 + shock_arrivals / 100)

    # ── Fallback: if dest_interest is all zeros, derive from wiki_views ──
    if df["dest_interest"].sum() == 0 and "wiki_views" in df.columns:
        max_wiki = df["wiki_views"].max()
        if max_wiki > 0:
            df["dest_interest"] = (df["wiki_views"] / max_wiki * 100).round(1)
            log.info("dest_interest: using Wikipedia pageviews as proportional fallback")
        else:
            df["dest_interest"] = 50

    # Apply demand shock to dest_interest
    if shock_demand != 0:
        df["dest_interest"] = (df["dest_interest"] * (1 + shock_demand / 100)).clip(0, 100).round(1)

    # ── Normalize all 9 indicators to 0-100 ──
    # Supply
    df["norm_arrivals"] = _normalize(df["arrivals"]) if "arrivals" in df.columns else 50
    df["norm_nights"] = _normalize(df["nights"]) if "nights" in df.columns else 50
    df["norm_avg_stay"] = _normalize(df["avg_stay"], lower=1, upper=8) if "avg_stay" in df.columns else 50
    # Demand
    df["norm_dest_interest"] = _normalize(df["dest_interest"]) if "dest_interest" in df.columns else 50
    df["norm_wiki_views"] = _normalize(df["wiki_views"]) if "wiki_views" in df.columns else 50
    df["norm_seasonality"] = df["seasonality_score"].clip(0, 100) if "seasonality_score" in df.columns else 50
    # Environment
    df["norm_weather"] = df["weather_score"] if "weather_score" in df.columns else 50
    df["norm_air_quality"] = df["air_quality_score"].clip(0, 100) if "air_quality_score" in df.columns else 50
    df["norm_coastal"] = df["coastal_score"].clip(0, 100) if "coastal_score" in df.columns else 50

    # ── Two-Stage Pillar Entropy Weights ──
    norm_cols = [
        "norm_arrivals", "norm_nights", "norm_avg_stay",
        "norm_dest_interest", "norm_wiki_views", "norm_seasonality",
        "norm_weather", "norm_air_quality", "norm_coastal",
    ]

    try:
        pillar_result = compute_pillar_weights(df[norm_cols])
        weights = pillar_result["weights"]
        df.attrs["pillar_result"] = pillar_result
        df.attrs["weighting_method"] = "Two-Stage Pillar Entropy (3x3)"
        log.info("Using two-stage pillar entropy weights (3x3)")
    except Exception as e:
        log.warning(f"Pillar entropy failed ({e}), using fallback weights")
        weights = WEIGHTS_FALLBACK
        df.attrs["pillar_result"] = None
        df.attrs["weighting_method"] = "Static (fallback)"

    df.attrs["weights"] = weights

    # ── Composite Score ──
    df["resilience_score"] = sum(
        weights.get(key, 0) * df[f"norm_{key}"] for key in COMPONENT_NAMES
    ).round(1)

    # Weighted contributions (for waterfall chart)
    for key in COMPONENT_NAMES:
        df[f"w_{key}"] = (weights.get(key, 0) * df[f"norm_{key}"]).round(1)

    # Pillar composite scores (for diagnostics)
    for pillar_name, pillar_def in PILLARS.items():
        indicators = pillar_def["indicators"]
        df[f"pillar_{pillar_name}"] = sum(
            df[f"norm_{v}"] for v in indicators
        ) / len(indicators)

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
    """Fetch all data sources and return tuple of DataFrames."""
    df_euro = fetch_eurostat_tourism()
    df_trends = fetch_google_trends()          # National trends (for time-series page)
    df_dest = fetch_destination_trends()        # Per-destination search interest
    df_weather = fetch_weather()
    df_air = fetch_air_quality()
    df_marine = fetch_marine()
    df_wiki = fetch_wikipedia_pageviews()
    df_season = fetch_eurostat_seasonality()

    snapshot = build_resilience_snapshot(
        df_euro, df_dest, df_weather, df_air, df_marine, df_wiki, df_season,
        shock_demand=shock_demand, shock_arrivals=shock_arrivals,
    )
    return snapshot, df_euro, df_trends, df_dest, df_weather, df_air, df_marine, df_wiki, df_season


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    snap, euro, trends, dest, weather, air, marine, wiki, season = fetch_all()

    print("\n=== TWO-STAGE PILLAR ENTROPY (3x3) ===")
    pr = snap.attrs.get("pillar_result")
    if pr:
        pw = pr.get("pillar_weights", {})
        print(f"\n{'Pillar':<20s} {'Entropy wt':>10s}")
        print("-" * 32)
        for pname in PILLARS:
            print(f"{PILLAR_NAMES[pname]:<20s} {pw.get(pname, 0):10.2%}")

        print(f"\n{'Variable':<24s} {'Pillar':<18s} {'Within w':>10s} {'Final w':>10s}")
        print("-" * 66)
        for pname, pdet in pr["pillar_details"].items():
            for var, ww in pdet["within_weights"].items():
                fw = pr["weights"].get(var, 0)
                print(f"{COMPONENT_NAMES.get(var, var):<24s} "
                      f"{PILLAR_NAMES[pname]:<18s} {ww:10.4f} {fw:10.4f}")
        print(f"{'SUM':<24s} {'':>18s} {'':>10s} {sum(pr['weights'].values()):10.4f}")

    print("\n=== RESILIENCE SNAPSHOT ===")
    cols = ["dimos", "region", "resilience_score", "zone",
            "pillar_supply", "pillar_demand", "pillar_environment"]
    available = [c for c in cols if c in snap.columns]
    print(snap[available].to_string(index=False))

    print(f"\nWeighting: {snap.attrs.get('weighting_method', 'N/A')}")
    print(f"Eurostat year: {snap['eurostat_year'].iloc[0]}")
    print(f"Destinations: {len(snap)}")
    print(f"Timestamp: {datetime.now()}")
