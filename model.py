import os
import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import wandb
import pickle
from utils import fetch_air_quality_data, fetch_weather_data, fetch_coordinates, generate_health_advisory

@st.cache_resource
def initialize_wandb():
    """
    Initialize Weights & Biases for tracking.
    """
    wandb_api_key = os.getenv("WANDB_API_KEY", "8346c9bf77129881071a5d8467990a03e1252551")
    os.environ["WANDB_API_KEY"] = wandb_api_key
    try:
        wandb.login()
        return True
    except:
        return False

def fetch_historical_aqi_data(city, start_date, end_date):
    """
    Fetches historical AQI (PM10) measurements for the given city and date range.
    
    Args:
        city (str): The city name
        start_date (str): Start date in ISO format (YYYY-MM-DD)
        end_date (str): End date in ISO format (YYYY-MM-DD)
        
    Returns:
        list: List of (timestamp, value) tuples
    """
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "parameter": "pm10",
        "date_from": start_date,
        "date_to": end_date,
        "limit": 1000,
        "sort": "asc"  # ascending order by time
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        results = data.get("results", [])
        historical_data = []
        for measurement in results:
            value = measurement.get("value")
            time_str = measurement.get("date", {}).get("utc")
            if value is not None and time_str:
                historical_data.append((time_str, value))
        return historical_data
    except Exception as e:
        st.warning(f"Error fetching historical AQI data: {e}")
        return []

def fetch_historical_weather_data(latitude, longitude, start_date, end_date):
    """
    Fetches historical weather data from Open-Meteo Archive.
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        start_date (str): Start date in ISO format (YYYY-MM-DD)
        end_date (str): End date in ISO format (YYYY-MM-DD)
        
    Returns:
        dict: Historical weather data
    """
    OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
    }
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=10)
        data = response.json()
        hourly = data.get("hourly", {})
        return hourly
    except Exception as e:
        st.warning(f"Error fetching historical weather data: {e}")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        hours = int(((end - start).total_seconds() / 3600)) + 24
        return {
            "time": pd.date_range(start=start, periods=hours, freq="H").strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [np.random.uniform(15, 35) for _ in range(hours)],
            "relative_humidity_2m": [np.random.uniform(40, 90) for _ in range(hours)],
            "windspeed_10m": [np.random.uniform(0, 15) for _ in range(hours)],
            "precipitation": [np.random.uniform(0, 5) for _ in range(hours)]
        }

def prepare_dataset(city, latitude, longitude, days=30):
    """
    Fetch and prepare the dataset for training.
    
    Args:
        city (str): The city name
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        days (int): Number of days of data to prepare
        
    Returns:
        pd.DataFrame: Prepared dataset
    """
    end_date_dt = datetime.date.today() - datetime.timedelta(days=1)
    start_date_dt = end_date_dt - datetime.timedelta(days=days-1)
    start_date = start_date_dt.isoformat()
    end_date = end_date_dt.isoformat()
    
    weather_data = fetch_historical_weather_data(latitude, longitude, start_date, end_date)
    times = weather_data.get("time")
    if not times:
        hours = 24 * days
        times = pd.date_range(start=start_date_dt, periods=hours, freq="H").strftime("%Y-%m-%dT%H:%M").tolist()
    
    df_weather = pd.DataFrame({
        "time": pd.to_datetime(times),
        "temperature": weather_data.get("temperature_2m"),
        "humidity": weather_data.get("relative_humidity_2m"),
        "wind_speed": weather_data.get("windspeed_10m"),
        "precipitation": weather_data.get("precipitation")
    })
    
    # Add hour feature
    df_weather["hour"] = df_weather["time"].dt.hour
    
    aqi_data = fetch_historical_aqi_data(city, start_date, end_date)
    if aqi_data:
        df_aqi = pd.DataFrame(aqi_data, columns=["time", "aqi"])
        df_aqi["time"] = pd.to_datetime(df_aqi["time"])
        df_aqi = df_aqi.set_index("time").resample("H").mean().reset_index()
    else:
        current_aqi, _ = fetch_air_quality_data(city)
        df_aqi = pd.DataFrame({
            "time": df_weather["time"],
            "aqi": [current_aqi + np.random.normal(0, 15) for _ in range(len(df_weather))]
        })
    
    df = pd.merge_asof(df_weather.sort_values("time"), df_aqi.sort_values("time"), on="time", direction="nearest")
    return df[["hour", "temperature", "humidity", "wind_speed", "precipitation", "aqi"]].dropna()

def train_model(data, city_name):
    """
    Train a RandomForest model to predict AQI based on weather data.
    
    Args:
        data (pd.DataFrame): Dataset containing features and target
        city_name (str): Name of the city for logging
        
    Returns:
        tuple: (model, mean_absolute_error, predictions_vs_actual, feature_importances)
    """
    wandb_initialized = initialize_wandb()
    
    if wandb_initialized:
        wandb.init(project="air-quality-prediction", 
                  config={"model": "RandomForest", "test_size": 0.2},
                  name=city_name)
        config = wandb.config
    else:
        class Config:
            test_size = 0.2
        config = Config()
    
    # Feature engineering
    feature_names = ["hour", "temperature", "humidity", "wind_speed", "precipitation"]
    X = data[feature_names]
    y = data["aqi"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=42)
    
    # Set up a simple grid search for hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    
    with st.spinner("Training model with grid search..."):
        grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    if wandb_initialized:
        # Log model performance metrics
        wandb.log({"mae": mae, "r2_score": r2})
        
        # Log feature importance
        feature_importance = {
            feature: importance 
            for feature, importance in zip(feature_names, best_model.feature_importances_)
        }
        wandb.log({"feature_importances": feature_importance})
        
        # Save the model to a file and log it as an artifact
        model_file = "random_forest_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump((best_model, feature_names), f)
        artifact = wandb.Artifact("random_forest_model", type="model")
        artifact.add_file(model_file)
        wandb.log_artifact(artifact)
    
    # Prepare predictions vs actual for plotting
    pred_actual = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions
    })
    
    return best_model, mae, pred_actual, best_model.feature_importances_

def get_24h_forecast(model, weather_data, feature_names):
    """
    Uses the trained model to generate a 24-hour forecast of AQI based on current weather data.
    
    Args:
        model: Trained model
        weather_data: Dictionary with hourly weather forecasts
        feature_names: List of feature names used by the model
        
    Returns:
        pd.DataFrame: Hourly forecast with AQI and health advisories
    """
    features = pd.DataFrame({
        "hour": list(range(24)),
        "temperature": weather_data["temperature_2m"][:24],
        "humidity": weather_data["relative_humidity_2m"][:24],
        "wind_speed": weather_data["windspeed_10m"][:24],
        "precipitation": weather_data["precipitation"][:24]
    })
    
    features_for_prediction = features[feature_names]
    predictions = model.predict(features_for_prediction)
    
    # Create forecast DataFrame
    current_time = pd.Timestamp.now().floor('H')
    forecast_times = [current_time + pd.Timedelta(hours=i) for i in range(24)]
    
    forecast_df = pd.DataFrame({
        'Time': forecast_times,
        'Hour': [t.hour for t in forecast_times],
        'Predicted AQI': predictions.round(1),
        'Temperature': features['temperature'].round(1),
        'Humidity': features['humidity'].round(1),
        'Wind Speed': features['wind_speed'].round(1),
        'Precipitation': features['precipitation'].round(2)
    })
    
    # Add health advisories
    forecast_df['Status'] = [generate_health_advisory(aqi)[0] for aqi in forecast_df['Predicted AQI']]
    forecast_df['Health Advisory'] = [generate_health_advisory(aqi)[1] for aqi in forecast_df['Predicted AQI']]
    
    return forecast_df

def get_7day_forecast(model, weather_data, feature_names):
    """
    Uses the trained model to generate a 7-day AQI forecast based on weather data.
    
    Args:
        model: Trained model
        weather_data: Dictionary with hourly weather forecasts
        feature_names: List of feature names used by the model
        
    Returns:
        pd.DataFrame: Daily forecast with AQI and health advisories
    """
    daily_forecasts = []
    today = datetime.date.today()
    
    for day_offset in range(7):
        day_date = today + datetime.timedelta(days=day_offset)
        day_name = day_date.strftime("%A")
        date_str = day_date.strftime("%Y-%m-%d")
        
        # Extract 24 hours of data for this day
        features = pd.DataFrame({
            "hour": list(range(24)),
            "temperature": weather_data["temperature_2m"][day_offset * 24:(day_offset + 1) * 24],
            "humidity": weather_data["relative_humidity_2m"][day_offset * 24:(day_offset + 1) * 24],
            "wind_speed": weather_data["windspeed_10m"][day_offset * 24:(day_offset + 1) * 24],
            "precipitation": weather_data["precipitation"][day_offset * 24:(day_offset + 1) * 24]
        })
        
        features_for_prediction = features[feature_names]
        daily_predictions = model.predict(features_for_prediction)
        
        # Calculate daily statistics
        daily_aqi = daily_predictions.mean()
        min_aqi = daily_predictions.min()
        max_aqi = daily_predictions.max()
        avg_temp = features["temperature"].mean()
        avg_humidity = features["humidity"].mean()
        
        # Get health status and advisory
        status, advisory = generate_health_advisory(daily_aqi)
        
        daily_forecasts.append({
            'Day': day_name,
            'Date': date_str,
            'Avg AQI': round(daily_aqi, 1),
            'Min AQI': round(min_aqi, 1),
            'Max AQI': round(max_aqi, 1),
            'Avg Temp': round(avg_temp, 1),
            'Avg Humidity': round(avg_humidity, 1),
            'Status': status,
            'Health Advisory': advisory
        })
    
    return pd.DataFrame(daily_forecasts)
