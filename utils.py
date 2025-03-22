import os
import requests
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import time

# API endpoints
OPENAQ_API_URL = "https://api.openaq.org/v2"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WAQI_API_KEY = os.getenv("WAQI_API_KEY", "63b843296fe374724c45e80851062013e78e7282")

# Set shorter cache duration for real-time data
@st.cache_data(ttl=300)  # Cache for only 5 minutes
def fetch_air_quality_data(city):
    """
    Fetch current AQI (PM10) from OpenAQ using the 'latest' endpoint.
    Falls back to an estimated value if data retrieval fails.
    
    Args:
        city (str): The city to fetch air quality data for
    
    Returns:
        int: AQI value for the city
        dict: Additional air quality data
    """
    # First try with WAQI API
    if WAQI_API_KEY:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    aqi = data["data"].get("aqi")
                    
                    # Get additional data if available
                    additional_data = {}
                    if "iaqi" in data["data"]:
                        iaqi = data["data"]["iaqi"]
                        for key in iaqi:
                            if "v" in iaqi[key]:
                                additional_data[key] = iaqi[key]["v"]
                    
                    st.success(f"✅ Successfully retrieved real-time air quality data from WAQI")
                    return aqi, additional_data
        except Exception as e:
            st.warning(f"Error fetching from WAQI: {e}")
    
    # Fallback to OpenAQ
    params = {"city": city, "limit": 1}
    try:
        response = requests.get(f"{OPENAQ_API_URL}/latest", params=params, timeout=10)
        data = response.json()
        
        results = data.get("results", [])
        if results:
            measurements = results[0].get("measurements", [])
            additional_data = {}
            
            # Add timestamp to verify recency
            if "date" in results[0] and "utc" in results[0]["date"]:
                data_time = pd.to_datetime(results[0]["date"]["utc"])
                data_age = (datetime.datetime.now(datetime.timezone.utc) - data_time).total_seconds() / 60
                additional_data["data_time"] = data_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                additional_data["data_age_minutes"] = round(data_age, 1)
            
            for measure in measurements:
                param = measure.get("parameter")
                value = measure.get("value")
                if param and value is not None:
                    additional_data[param] = value
                    
                # Return pm10 as the main AQI if available
                if param == "pm10":
                    st.success(f"✅ Successfully retrieved real-time air quality data from OpenAQ")
                    return value, additional_data
                    
            # If we have pm25 but not pm10, use that
            if "pm25" in additional_data:
                st.success(f"✅ Successfully retrieved real-time air quality data (PM2.5) from OpenAQ")
                return additional_data["pm25"], additional_data
    except Exception as e:
        st.warning(f"Error retrieving air quality data: {e}")
    
    # If all else fails, use IQAir API if you have an API key
    if os.getenv("IQAIR_API_KEY"):
        try:
            iqair_api_key = os.getenv("IQAIR_API_KEY")
            # Get coordinates first
            lat, lon, _ = fetch_coordinates(city)
            if lat and lon:
                url = f"http://api.airvisual.com/v2/nearest_city?lat={lat}&lon={lon}&key={iqair_api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        aqi = data["data"]["current"]["pollution"]["aqius"]
                        additional_data = {
                            "pm25": data["data"]["current"]["pollution"].get("pm25"),
                            "data_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "data_age_minutes": 0,
                            "source": "IQAir"
                        }
                        st.success(f"✅ Successfully retrieved real-time air quality data from IQAir")
                        return aqi, additional_data
        except Exception as e:
            st.warning(f"Error retrieving IQAir data: {e}")
    
    st.error("❌ Could not retrieve real-time air quality data. Please check your location or try another data source.")
    return None, {}

@st.cache_data(ttl=300)  # Cache for only 5 minutes
def fetch_weather_data(latitude, longitude):
    """
    Fetch current weather forecast data using the Open-Meteo forecast API.
    Retrieves temperature, relative humidity, wind speed, and precipitation.
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
    
    Returns:
        dict: Weather data containing hourly values
    """
    weather_url = (f"{OPEN_METEO_BASE_URL}?latitude={latitude}&longitude={longitude}"
                   f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
                   f"&current_weather=true"  # Include current weather
                   f"&timeformat=unixtime")  # Use unixtime for accurate time comparison
    try:
        response = requests.get(weather_url, timeout=10)
        if response.status_code != 200:
            st.warning(f"Weather API returned status code {response.status_code}")
            return None
            
        weather_response = response.json()
        
        # Verify we have current weather data
        if "current_weather" in weather_response:
            current_time = datetime.datetime.fromtimestamp(weather_response["current_weather"]["time"])
            st.success(f"✅ Successfully retrieved real-time weather data (as of {current_time.strftime('%Y-%m-%d %H:%M')})")
    except Exception as e:
        st.warning(f"Error retrieving current weather data: {e}")
        return None
    
    if "hourly" in weather_response:
        # Include current weather information in additional metadata
        weather_data = weather_response["hourly"]
        weather_data["current_weather"] = weather_response.get("current_weather", {})
        return weather_data
    else:
        st.error("❌ Could not retrieve real-time weather data. Please try again later.")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours as locations don't change frequently
def fetch_coordinates(location):
    """
    Uses the Photon API to convert a location into latitude and longitude.
    
    Args:
        location (str): The location to fetch coordinates for
    
    Returns:
        tuple: (latitude, longitude, place_name)
    """
    url = "https://photon.komoot.io/api/"
    try:
        response = requests.get(url, params={"q": location, "limit": 1}, timeout=10)
        features = response.json().get("features", [])
        if features:
            coords = features[0].get("geometry", {}).get("coordinates", [None, None])
            place_name = features[0]["properties"].get("name", location)
            st.success(f"✅ Successfully located coordinates for {place_name}")
            return coords[1], coords[0], place_name
    except Exception as e:
        st.warning(f"Error retrieving coordinates: {e}")
    
    st.error(f"❌ Could not find coordinates for {location}. Please check the spelling or try another location.")
    return None, None, location

def generate_health_advisory(aqi):
    """
    Provide health recommendations based on AQI.
    
    Args:
        aqi (int): Air Quality Index value
    
    Returns:
        tuple: (status, health_advisory)
    """
    if aqi is None:
        return "Unknown", "Unable to provide health advisory due to missing AQI data."
    
    if aqi <= 50:
        return "Good", "Air quality is good. No precautions necessary."
    elif aqi <= 100:
        return "Moderate", "Air quality is moderate. Sensitive individuals should limit outdoor activities."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Unhealthy for sensitive groups. Consider wearing a mask outdoors."
    elif aqi <= 200:
        return "Unhealthy", "Unhealthy. Everyone should limit outdoor activities."
    elif aqi <= 300:
        return "Very Unhealthy", "Very unhealthy. Avoid outdoor activities. Use air purifiers indoors."
    else:
        return "Hazardous", "Hazardous. Stay indoors and keep windows closed."

def generate_health_symptoms(aqi):
    """
    Predict possible health symptoms and problems due to the AQI.
    
    Args:
        aqi (int): Air Quality Index value
    
    Returns:
        str: Predicted health symptoms
    """
    if aqi is None:
        return "Unable to predict health symptoms due to missing AQI data."
        
    if aqi <= 50:
        return "No significant symptoms expected."
    elif aqi <= 100:
        return "Mild irritation of eyes, throat, or respiratory tract."
    elif aqi <= 150:
        return "Possible respiratory discomfort, coughing, or fatigue."
    elif aqi <= 200:
        return "Increased risk of respiratory issues and aggravation of asthma."
    elif aqi <= 300:
        return "High risk of serious respiratory problems and heart stress."
    else:
        return "Critical health risk. Emergency medical attention may be required."

def get_aqi_color(aqi):
    """
    Return a color based on the AQI value.
    
    Args:
        aqi (int): Air Quality Index value
    
    Returns:
        str: Hex color code
    """
    if aqi is None:
        return "#cccccc"  # Gray for unknown
        
    if aqi <= 50:
        return "#00e400"  # Green
    elif aqi <= 100:
        return "#ffff00"  # Yellow
    elif aqi <= 150:
        return "#ff7e00"  # Orange
    elif aqi <= 200:
        return "#ff0000"  # Red
    elif aqi <= 300:
        return "#99004c"  # Purple
    else:
        return "#7e0023"  # Maroon
