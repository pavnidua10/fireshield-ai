# 🔥 FireShield AI

> Intelligent wildfire risk prediction powered by real-time weather data, geospatial analysis, and machine learning.

<img width="1365" height="596" alt="image" src="https://github.com/user-attachments/assets/751060f6-a5e5-4ae2-ad40-439fa3e967ae" />
<img width="1306" height="559" alt="image" src="https://github.com/user-attachments/assets/b67f1522-de90-4b25-a703-ec61d18f124b" />
<img width="1126" height="591" alt="image" src="https://github.com/user-attachments/assets/b610face-f336-44c4-bc61-6c06f5211d0f" />

---

## What is FireShield AI?

FireShield AI is a full-stack web application that predicts the risk of forest fires for any location on Earth. It combines live weather data, satellite-derived terrain information, and a trained XGBoost machine learning model to classify fire risk as **Low**, **Medium**, or **High** — in real time.

---

## Features

- **Global coverage** — search any city, national park, or drop a pin anywhere on Earth
- **Live data pipeline** — fetches real-time weather, soil moisture, wind, and rainfall via Open-Meteo
- **Geospatial analysis** — pulls forest density, road distance, and village proximity via OpenStreetMap Overpass API
- **11-feature ML model** — XGBoost classifier trained on real UCI Forest Fires data
- **Interactive GIS map** — Leaflet.js map with satellite/street toggle and click-to-scan
- **AI risk report** — animated gauge, confidence score, explanation, and response recommendations
- **Radar analytics chart** — visualises all 5 key risk dimensions simultaneously
- **Emergency alert** — auto-popup on high risk detection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Gunicorn |
| ML Model | XGBoost, scikit-learn |
| Weather API | Open-Meteo (free, no key required) |
| Geospatial | OpenStreetMap Overpass API, Nominatim |
| Frontend | HTML, CSS, Bootstrap 5, Leaflet.js, Chart.js |
| Deployment | Render |

---

## How It Works

```
User enters a location
        ↓
Geocode → lat/lon via OpenStreetMap Nominatim
        ↓
Fetch live data:
  • Open-Meteo  → temperature, humidity, wind, rainfall, soil moisture
  • Open-Meteo  → elevation
  • Overpass    → forest density, road distance, village distance
        ↓
Compute synthetic features:
  • NDVI (vegetation index)
  • Leaf Litter Index (fuel load)
  • Fire History proxy
        ↓
Scale 11 features → XGBoost model → predict risk class
        ↓
Return: risk level + confidence + explanation + recommendations
```

---

## The 11 Features

| Feature | Source | Description |
|---|---|---|
| Temperature | Open-Meteo | Current air temperature (°C) |
| Humidity | Open-Meteo | Relative humidity (%) |
| Wind Speed | Open-Meteo | Wind speed (km/h) |
| Rainfall 7d | Open-Meteo | 7-day cumulative rainfall (mm) |
| Soil Moisture | Open-Meteo | Volumetric soil water content (%) |
| Forest Cover | Overpass/OSM | Forest/woodland density (%) |
| Elevation | Open-Meteo | Terrain elevation (m) |
| Road Distance | Overpass/OSM | Distance to nearest road (m) |
| NDVI | Computed | Vegetation greenness index (0–1) |
| Leaf Litter Index | Computed | Dry fuel load estimate (0–1) |
| Fire History | Computed | Historical fire propensity (0–10) |

---


## Dataset

The model is trained on the **UCI Forest Fires Dataset** from Montesinho Natural Park, Portugal. It contains 517 real fire records with weather indices (FFMC, DMC, DC, ISI), meteorological data, and burned area measurements.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/162/forest+fires)

---

## Model Performance

| Metric | Score |
|---|---|
| Overall Accuracy | 77.2% |
| High Risk Precision | 93% |
| Classes | Low / Medium / High |
| Algorithm | XGBoost (200 estimators, depth 6) |

High risk detection achieves 93% precision — the most critical class for real-world fire safety.

---

## Deployment

The app is deployed on **Render** (free tier). Live URL:

> **https://fireshield-ai.onrender.com**

Note: the free tier sleeps after 15 minutes of inactivity. First load after sleep may take ~30 seconds.

---

## External APIs Used

All APIs used are **free with no API key required.**

- [Open-Meteo](https://open-meteo.com/) — weather and elevation data
- [OpenStreetMap Nominatim](https://nominatim.org/) — geocoding
- [Overpass API](https://overpass-api.de/) — geospatial forest/road/village data

---

