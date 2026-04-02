# ResilienceIQ Tier 0 — Tourism Resilience Dashboard (Real Data)

Tourism resilience monitoring for Greek municipalities using **live open data**.

## Data Sources (all free, no API keys)

| Source | Indicator | Update | API |
|--------|-----------|--------|-----|
| **Eurostat** | Arrivals & nights by NUTS-2 | Annual | REST `tour_occ_arn2`, `tour_occ_nin2` |
| **Google Trends** | Tourism demand keywords (GR) | Weekly | pytrends |
| **Open-Meteo** | Weather attractiveness | Hourly | Forecast API |

## Files

| File | Description |
|------|-------------|
| `data_fetcher.py` | Real data pipeline (Eurostat + Trends + Weather) |
| `tier0_tourism_app.py` | Streamlit realtime dashboard (4 pages) |
| `create_excel_dashboard.py` | Excel dashboard generator |
| `ResilienceIQ_Dashboard.xlsx` | Pre-built Excel snapshot |

## Quick Start

```bash
pip install -r requirements.txt

# Streamlit dashboard (live)
streamlit run tier0_tourism_app.py

# Regenerate Excel snapshot
python create_excel_dashboard.py
```

## Dashboard Pages

1. **Dashboard** — Map, ranking, KPIs, full data table
2. **Municipality Deep-Dive** — Waterfall decomposition, radar vs average, live weather
3. **Trends & Time Series** — Google Trends chart, Eurostat historical bars, YoY table
4. **Methodology** — Weights, formula, NUTS-2 mapping, zone classification

## Composite Score Formula

```
Score = 0.25 * norm(Arrivals) + 0.20 * norm(Nights) + 0.25 * norm(Demand)
      + 0.15 * Weather_Score + 0.15 * norm(Avg_Stay)
```

## Tier Upgrade Path

Tier 0 (Free) -> Tier 1 (Regional Lite) -> Tier 2 (Analytics) -> Tier 3 (DSS)
