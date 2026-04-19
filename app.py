import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date, time, timedelta

# Page configuration
st.set_page_config(
    page_title="Scotland Bus Delay Predictor",
    page_icon="🚌",
    layout="centered"
)

# Minimal CSS tweaks - hide menu/footer, styled header
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .app-header {
        background: #0b2545;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #8da9c4;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }
    .weather-note {
    background: transparent;
    border-left: 3px solid #4a6fa5;
    padding: 0.4rem 0.8rem;
    color: #2196F3;
    font-size: 0.9rem;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# Load model and lookup data
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

@st.cache_data
def load_lookups():
    route_names = pd.read_parquet("app_route_names.parquet")
    headsigns = pd.read_parquet("app_headsigns.parquet")
    stop_lookup = pd.read_parquet("app_stop_lookup.parquet")
    feature_cols = joblib.load("feature_columns.pkl")
    return route_names, headsigns, stop_lookup, feature_cols

model = load_model()
route_names, headsigns, stop_lookup, feature_cols = load_lookups()

# City coordinates (used for nearby places and directions)
CITY_COORDS = {
    "Edinburgh": {"lat": 55.9533, "lon": -3.1883},
    "Glasgow": {"lat": 55.8642, "lon": -4.2518},
    "Paisley": {"lat": 55.8456, "lon": -4.4239},
}

# OpenWeatherMap API - fetch live weather for selected city
def fetch_weather(city):
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},GB&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            st.error(f"Weather API error: {data.get('message', 'Unknown error')}")
            return None

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        windspeed = data["wind"]["speed"] * 3.6
        cloudcover = data["clouds"]["all"]
        rain_mm = data.get("rain", {}).get("1h", 0)
        snow_mm = data.get("snow", {}).get("1h", 0)
        precip = rain_mm + snow_mm
        weather_main = data["weather"][0]["main"]
        weather_desc = data["weather"][0]["description"]

        if snow_mm > 0:
            preciptype = "Snow"
        elif rain_mm > 0:
            preciptype = "Rain"
        else:
            preciptype = "None"

        conditions = map_conditions(weather_main, weather_desc, precip, preciptype)

        return {
            "temp": temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "cloudcover": cloudcover,
            "precip": precip,
            "preciptype": preciptype,
            "conditions": conditions,
            "description": weather_desc
        }

    except Exception as e:
        st.error(f"Failed to fetch weather: {e}")
        return None


def map_conditions(weather_main, weather_desc, precip, preciptype):
    desc = weather_desc.lower()
    main = weather_main.lower()

    if preciptype == "Snow":
        if precip >= 4:
            return "Snow fall: Heavy"
        elif precip >= 1:
            return "Snow fall: Moderate"
        else:
            return "Snow fall: Slight"
    elif "drizzle" in main or "drizzle" in desc:
        if precip >= 2:
            return "Drizzle: Dense"
        elif precip >= 0.5:
            return "Drizzle: Moderate"
        else:
            return "Drizzle: Light"
    elif preciptype == "Rain":
        if precip >= 2:
            return "Rain: Moderate"
        else:
            return "Rain: Slight"
    elif "cloud" in desc or "overcast" in desc:
        if "few" in desc or "scattered" in desc:
            return "Partly cloudy"
        elif "broken" in desc:
            return "Mainly clear"
        else:
            return "Overcast"
    elif "clear" in desc:
        return "Clear sky"
    elif "mist" in desc or "fog" in desc or "haze" in desc:
        return "Overcast"
    else:
        return "Partly cloudy"


# Overpass API - fetch nearby cafes, restaurants, shops
def fetch_nearby_places(lat, lon, radius=500):
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="cafe"](around:{radius},{lat},{lon});
      node["amenity"="restaurant"](around:{radius},{lat},{lon});
      node["shop"](around:{radius},{lat},{lon});
    );
    out body 10;
    """
    url = "https://overpass-api.de/api/interpreter"

    try:
        response = requests.post(url, data={"data": query}, timeout=15)
        if response.status_code != 200:
            return []

        data = response.json()
        places = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name")
            if not name:
                continue

            place_lat = element.get("lat", lat)
            place_lon = element.get("lon", lon)

            dist = haversine(lat, lon, place_lat, place_lon)
            place_type = tags.get("amenity", tags.get("shop", "shop"))

            places.append({
                "name": name,
                "type": place_type.title(),
                "distance_m": round(dist),
            })

        places.sort(key=lambda x: x["distance_m"])
        return places[:5]

    except Exception:
        return []


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# OpenRouteService API - fetch alternative routes
def fetch_alternative_routes(start_lat, start_lon, end_lat, end_lon):
    ORS_KEY = st.secrets["ORS_API_KEY"]
    alternatives = []

    try:
        url = "https://api.openrouteservice.org/v2/directions/foot-walking"
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "km"}
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            route = data["routes"][0]["summary"]
            walk_mins = round(route["duration"] / 60)
            walk_km = round(route["distance"], 1)
            alternatives.append({
                "mode": "Walking",
                "duration": f"{walk_mins} mins",
                "distance": f"{walk_km} km",
            })
    except Exception:
        pass

    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]], "units": "km"}
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            route = data["routes"][0]["summary"]
            drive_mins = round(route["duration"] / 60)
            drive_km = round(route["distance"], 1)
            alternatives.append({
                "mode": "Taxi / Drive",
                "duration": f"{drive_mins} mins",
                "distance": f"{drive_km} km",
            })
    except Exception:
        pass

    return alternatives


# Build prediction input matching the 26 model features
def build_features(direction_id, stop_sequence, day_of_week, is_weekend,
                    hour, is_rush_hour, city, weather):
    features = {
        "direction_id": direction_id,
        "stop_sequence": stop_sequence,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour": hour,
        "temp": weather["temp"],
        "precip": weather["precip"],
        "windspeed": weather["windspeed"],
        "humidity": weather["humidity"],
        "cloudcover": weather["cloudcover"],
        "is_rush_hour": is_rush_hour,
    }

    features["city_Glasgow"] = 1 if city == "Glasgow" else 0
    features["city_Paisley"] = 1 if city == "Paisley" else 0

    condition_cols = [
        "conditions_Drizzle: Dense", "conditions_Drizzle: Light",
        "conditions_Drizzle: Moderate", "conditions_Mainly clear",
        "conditions_Overcast", "conditions_Partly cloudy",
        "conditions_Rain: Moderate", "conditions_Rain: Slight",
        "conditions_Snow fall: Heavy", "conditions_Snow fall: Moderate",
        "conditions_Snow fall: Slight"
    ]
    for col in condition_cols:
        condition_name = col.replace("conditions_", "")
        features[col] = 1 if weather["conditions"] == condition_name else 0

    features["preciptype_Rain"] = 1 if weather["preciptype"] == "Rain" else 0
    features["preciptype_Snow"] = 1 if weather["preciptype"] == "Snow" else 0

    df = pd.DataFrame([features])
    df = df[feature_cols]
    return df


# ── App interface ──────────────────────────────────────────────────────────────

# Styled header
st.markdown("""
<div class="app-header">
    <h1>Scotland Bus Delay Predictor</h1>
    <p>Real-time delay predictions for Edinburgh, Glasgow and Paisley</p>
</div>
""", unsafe_allow_html=True)

city = st.selectbox("Select city", ["Edinburgh", "Glasgow", "Paisley"])

city_routes = route_names[route_names["city"] == city].sort_values("display_name")
route_options = dict(zip(city_routes["display_name"], city_routes["route_id"]))
selected_route_name = st.selectbox("Select route", list(route_options.keys()))
selected_route_id = route_options[selected_route_name]

route_headsigns = headsigns[headsigns["route_id"] == selected_route_id]
direction_options = {f"Towards {row['headsign']}": row["direction_id"] for _, row in route_headsigns.iterrows()}
selected_direction_name = st.selectbox("Select direction", list(direction_options.keys()))
selected_direction_id = direction_options[selected_direction_name]

route_stops = stop_lookup[
    (stop_lookup["route_id"] == selected_route_id) &
    (stop_lookup["direction_id"] == selected_direction_id)
].sort_values("stop_sequence")

stop_options = {
    f"{row.get('stop_name', row['stop_id'])} (Stop {row['stop_sequence']})":
    [row['stop_id'], row['stop_sequence']]
    for _, row in route_stops.iterrows()
}
selected_stop_name = st.selectbox("Select stop", list(stop_options.keys()))
selected_stop_id, selected_stop_sequence = stop_options[selected_stop_name]

# ── Stop location map ──────────────────────────────────────────────────────────
selected_stop_row = route_stops[route_stops["stop_id"] == selected_stop_id]

if not selected_stop_row.empty:
    stop_lat = selected_stop_row.iloc[0].get("stop_lat")
    stop_lon = selected_stop_row.iloc[0].get("stop_lon")

    
    if pd.notna(stop_lat) and pd.notna(stop_lon):
        st.markdown("**Stop Location**")
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[stop_lat, stop_lon], zoom_start=16)
        folium.Marker(
            location=[stop_lat, stop_lon],
            popup=selected_stop_name,
            tooltip=selected_stop_name,
            icon=folium.Icon(color="red", icon="bus", prefix="fa")
        ).add_to(m)
        st_folium(m, width=700, height=400)

# ── Date and time ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    today = date.today()
    max_date = today + timedelta(days=3)
    selected_date = st.date_input("Date", value=today, min_value=today, max_value=max_date)
with col2:
    selected_time = st.time_input("Select time", value=time(12, 0))

selected_hour = selected_time.hour
selected_dow = selected_date.weekday()
selected_is_weekend = 1 if selected_dow >= 5 else 0
selected_is_rush_hour = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

if st.button("Predict Delay", type="primary"):
    with st.spinner("Fetching live weather data..."):
        weather = fetch_weather(city)

    if weather:
        st.markdown("---")
        st.subheader("Current Weather")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Temperature", f"{weather['temp']:.1f}°C")
        w_col2.metric("Wind", f"{weather['windspeed']:.1f} km/h")
        w_col3.metric("Humidity", f"{weather['humidity']}%")
        st.markdown(f"**Conditions:** {weather['description'].title()} | **Cloud Cover:** {weather['cloudcover']}%")

        # Show weather note if date is not today
        if selected_date != today:
            st.markdown(
                '<div class="weather-note">'
                'Weather data reflects current conditions. For future dates, the prediction assumes similar weather.'
                '</div>',
                unsafe_allow_html=True
            )

        features = build_features(selected_direction_id, selected_stop_sequence, selected_dow,
                                  selected_is_weekend, selected_hour, selected_is_rush_hour, city, weather)

        prediction = model.predict(features)[0]
        prediction = round(max(prediction, -3), 1)

        if prediction < 0:
            status = "Ahead of Schedule"
        elif prediction <= 3:
            status = "On Time"
        elif prediction <= 5:
            status = "Slightly Delayed"
        elif prediction <= 10:
            status = "Moderately Delayed"
        else:
            status = "Heavily Delayed"

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Estimated Delay", f"{prediction:.1f} minutes")
        st.markdown(f"**Status: {status}**")

        factors = []
        if selected_is_rush_hour: factors.append("rush hour traffic")
        if weather["precip"] > 0: factors.append(f"{weather['conditions'].lower()}")
        if weather["temp"] <= 0: factors.append("freezing conditions")
        if weather["windspeed"] > 40: factors.append("high winds")

        if factors:
            st.info(f"Contributing factors: {', '.join(factors)}")

        # Nearby Places - triggers when delay > 0
        if prediction > 0:
            st.markdown("---")
            st.subheader("Nearby Places to Wait")
            st.markdown(f"Your bus is **{prediction:.1f} minutes late**. Here are some places nearby:")
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            with st.spinner("Searching for nearby places..."):
                places = fetch_nearby_places(coords["lat"], coords["lon"])
            if places:
                for place in places:
                    st.markdown(f"**{place['name']}** — {place['type']} · {place['distance_m']}m away")
            else:
                st.caption("No nearby places found within 500m.")

        # Alternative Routes - triggers when delay > 0
        if prediction > 0:
            st.markdown("---")
            st.subheader("Alternative Ways to Get There")
            st.markdown("It might be faster to take a different route:")
            coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
            last_stop = route_stops.iloc[-1] if len(route_stops) > 0 else None
            if last_stop is not None:
                dest_coords = CITY_COORDS.get(city, CITY_COORDS["Edinburgh"])
                dest_lat, dest_lon = dest_coords["lat"] + 0.02, dest_coords["lon"] + 0.02
                with st.spinner("Checking alternative routes..."):
                    alternatives = fetch_alternative_routes(coords["lat"], coords["lon"], dest_lat, dest_lon)
                if alternatives:
                    for alt in alternatives:
                        st.markdown(f"**{alt['mode']}** — {alt['duration']} ({alt['distance']})")
                else:
                    st.caption("Could not find alternative routes at this time.")

st.markdown("---")
st.caption("Powered by XGBoost | Weather: OpenWeatherMap | Places: OpenStreetMap | Routes: OpenRouteService | Bus data: BODS")