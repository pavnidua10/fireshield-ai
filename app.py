from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import requests
import xgboost as xgb
import math
from datetime import datetime

app = Flask(__name__)

# Load Model & Scaler
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

# ── Helper: convert any numpy type to plain Python ──────────────────
def to_python(val):
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    return val

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

# ── Geospatial helpers ───────────────────────────────────────────────
def geocode_city(city):
    headers = {"User-Agent": "FireShieldAI/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
    response = requests.get(url, headers=headers).json()
    if response:
        return float(response[0]['lat']), float(response[0]['lon']), response[0]['display_name']
    return None, None, None

def get_open_meteo_data(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relative_humidity_2m,soil_moisture_0_to_7cm"
        f"&daily=rain_sum&timezone=auto"
    )
    res = requests.get(url).json()

    current_temp = float(res.get('current_weather', {}).get('temperature', 30))
    current_wind = float(res.get('current_weather', {}).get('windspeed', 15))

    try:
        current_hour = datetime.utcnow().hour
        humidity_arr  = res.get('hourly', {}).get('relative_humidity_2m', [])
        soil_arr      = res.get('hourly', {}).get('soil_moisture_0_to_7cm', [])
        current_humidity  = float(humidity_arr[current_hour]) if humidity_arr else 40.0
        soil_moisture_raw = float(soil_arr[current_hour])     if soil_arr     else 0.2
        soil_moisture_pct = min(soil_moisture_raw * 100, 100.0)
    except Exception:
        current_humidity  = 40.0
        soil_moisture_pct = 20.0

    try:
        rain_arr = res.get('daily', {}).get('rain_sum', [])
        rain_7d  = float(sum(r for r in rain_arr if r is not None))
    except Exception:
        rain_7d = 0.0

    return current_temp, current_humidity, current_wind, rain_7d, soil_moisture_pct

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url).json()
        if 'elevation' in res and len(res['elevation']) > 0:
            return float(res['elevation'][0])
    except Exception:
        pass
    return 1000.0

def get_overpass_infrastructure(lat, lon, radius=5000):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      nwr["landuse"="forest"](around:{radius},{lat},{lon});
      nwr["natural"="wood"](around:{radius},{lat},{lon});
      nwr["boundary"="national_park"](around:{radius},{lat},{lon});
      node["highway"](around:{radius},{lat},{lon});
      node["place"="village"](around:{radius},{lat},{lon});
      node["place"="town"](around:{radius},{lat},{lon});
    );
    out center;
    """
    try:
        res      = requests.post(overpass_url, data={'data': query}).json()
        elements = res.get('elements', [])

        forest_elements  = 0
        road_distance    = float(radius + 1000)
        village_distance = float(radius + 1000)
        is_park          = False

        for el in elements:
            tags    = el.get('tags', {})
            ext_lat = el.get('lat') or el.get('center', {}).get('lat')
            ext_lon = el.get('lon') or el.get('center', {}).get('lon')
            if not ext_lat or not ext_lon:
                continue

            if tags.get('landuse') == 'forest' or tags.get('natural') == 'wood':
                forest_elements += 1
            if tags.get('boundary') == 'national_park':
                is_park = True
            if 'highway' in tags:
                dist = haversine(lat, lon, ext_lat, ext_lon)
                if dist < road_distance:
                    road_distance = dist
            if 'place' in tags:
                dist = haversine(lat, lon, ext_lat, ext_lon)
                if dist < village_distance:
                    village_distance = dist

        forest_density = min(float(forest_elements * 15.0), 100.0)
        if is_park:
            forest_density = max(forest_density, 85.0)

        return forest_density, float(road_distance), float(village_distance)

    except Exception as e:
        print(f"Overpass Error: {e}")
        return 0.0, 5000.0, 5000.0

def haversine(lat1, lon1, lat2, lon2):
    R    = 6371000
    ph1  = math.radians(lat1)
    ph2  = math.radians(lat2)
    dph  = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dph/2)**2 + math.cos(ph1) * math.cos(ph2) * math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Main API route ───────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    try:
        data    = request.json
        city    = data.get('city')
        lat     = data.get('lat')
        lon     = data.get('lon')
        loc_name = "Unknown Location"

        if city:
            lat, lon, loc_name = geocode_city(city)
            if not lat:
                return jsonify({"error": "Location not found via geocoding."}), 404
        else:
            loc_name = f"Geo: {round(lat, 4)}, {round(lon, 4)}"

        lat = float(lat)
        lon = float(lon)

        # Fetch data
        current_temp, current_humidity, current_wind, rain_7d, soil_moisture_pct = get_open_meteo_data(lat, lon)
        elevation = get_elevation(lat, lon)
        forest_density, road_distance, village_distance = get_overpass_infrastructure(lat, lon)

        # Computed features — all plain Python floats
        ndvi         = float(min(((forest_density / 100.0) * 0.7) + ((soil_moisture_pct / 100.0) * 0.3), 1.0))
        leaf_litter  = float(min(((forest_density / 100.0) * 0.6) + (((100 - soil_moisture_pct) / 100.0) * 0.4), 1.0))
        fire_history = float(min((current_temp / 10) + ((100 - current_humidity) / 20), 10.0))

        features = np.array([[
            current_temp, current_humidity, current_wind, rain_7d, soil_moisture_pct,
            forest_density, ndvi, fire_history, elevation, road_distance, leaf_litter
        ]])
        features_scaled = scaler.transform(features)

        # Convert prediction outputs to plain Python types immediately
        prediction       = int(model.predict(features_scaled)[0])
        probabilities    = model.predict_proba(features_scaled)[0]
        confidence_score = float(max(probabilities) * 100)

        if prediction == 0:
            risk_level      = "Low Risk"
            color           = "green"
            explanation     = f"Safe parameters. NDVI is {round(ndvi, 2)} with {round(rain_7d, 2)}mm recent rain preventing immediate ignition threats."
            recommendations = ["Continue standard monitoring protocols.", "Perform routine equipment checks."]
        elif prediction == 1:
            risk_level      = "Medium Risk"
            color           = "yellow"
            explanation     = f"Elevated threat detected. Leaf litter index ({round(leaf_litter, 2)}) paired with temperatures of {current_temp}°C creates a combustible zone."
            recommendations = ["Increase frequency of patrol teams.", "Restrict localized campfires.", "Prepare evacuation vectors for nearest village."]
        else:
            risk_level      = "High Risk"
            color           = "red"
            explanation     = f"Critical Danger. Wind speeds of {current_wind}km/h in {round(forest_density, 1)}% dense forest with severe dryness ({round(soil_moisture_pct, 1)}% soil moisture). Fire spread would be catastrophic."
            recommendations = ["Trigger immediate evacuation protocols.", "Dispatch aerial surveying to coordinates.", "Mobilize ground fire units.", "Initiate water bombing if accessible."]

        # All values explicitly cast to plain Python types — no numpy types reach jsonify
        return jsonify({
            "risk_level":      risk_level,
            "color":           color,
            "confidence":      round(confidence_score, 2),
            "explanation":     explanation,
            "recommendations": recommendations,
            "data": {
                "location":         loc_name,
                "lat":              lat,
                "lon":              lon,
                "temperature":      round(current_temp, 2),
                "humidity":         round(current_humidity, 2),
                "wind_speed":       round(current_wind, 2),
                "rain_7d":          round(rain_7d, 2),
                "soil_moisture":    round(soil_moisture_pct, 2),
                "forest_density":   round(forest_density, 2),
                "elevation":        round(elevation, 2),
                "ndvi":             round(ndvi, 2),
                "leaf_litter":      round(leaf_litter, 2),
                "fire_history":     round(fire_history, 2),
                "road_distance":    round(road_distance, 0),
                "village_distance": round(village_distance, 0)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)