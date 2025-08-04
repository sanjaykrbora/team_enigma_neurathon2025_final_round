import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pickle
import os

# Import custom modules
from utils import (
    fetch_air_quality_data, fetch_weather_data, fetch_coordinates,
    generate_health_advisory, generate_health_symptoms, get_aqi_color
)
from model import prepare_dataset, train_model, get_24h_forecast, get_7day_forecast

# Set page configuration
st.set_page_config(
    page_title="Air Quality Monitor & Predictor",
    page_icon="🌬️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'location' not in st.session_state:
    st.session_state.location = ""
if 'location_input' not in st.session_state:
    st.session_state.location_input = ""
if 'days' not in st.session_state:
    st.session_state.days = 30
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def trigger_analysis(location_value, days_value):
    st.session_state.analyze_clicked = True
    st.session_state.location = location_value
    st.session_state.days = days_value
    st.session_state.last_refresh = datetime.now()

# Page Header
st.title("🌬️ Air Quality Monitor & Predictor")
st.markdown("""
This application monitors air quality in real-time and provides health advisories based on AI predictions.
Enter a location to get started.
""")

# Add auto-refresh checkbox to sidebar
with st.sidebar:
    st.header("Location Settings")
    # Bound input
    st.text_input("Enter city, state, or country:", key="location_input")
    
    # Add some popular cities for quick selection
    st.markdown("### Popular Locations")
    popular_cities = ["New York", "London", "Tokyo", "Beijing", "Delhi", "Sydney", "Paris", "Cape Town", "Rio de Janeiro", "Moscow"]
    
    # Create 2 columns for city buttons
    col1, col2 = st.columns(2)
    
    for i, city in enumerate(popular_cities):
        if i % 2 == 0:
            with col1:
                if st.button(city, key=f"btn_{city}"):
                    st.session_state.location = city
                    st.session_state.location_input = city  # keep input synced
                    st.session_state.analyze_clicked = True
                    st.session_state.button_clicked = True
        else:
            with col2:
                if st.button(city, key=f"btn_{city}"):
                    st.session_state.location = city
                    st.session_state.location_input = city
                    st.session_state.analyze_clicked = True
                    st.session_state.button_clicked = True
    
    st.divider()
    
    st.header("Model Settings")
    days_for_model = st.slider("Days of data for model training:", 
                              min_value=7, max_value=60, value=30, step=1)
    
    st.header("Real-time Settings")
    auto_refresh = st.checkbox("Auto-refresh data", value=False, help="Automatically refresh data every few minutes")
    
    refresh_interval = 15  # Default value
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (minutes)", 
                                   min_value=5, max_value=60, value=15, step=5)
        st.info(f"Data will automatically refresh every {refresh_interval} minutes")
        
        # Calculate time until next refresh
        if "last_refresh" in st.session_state:
            elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds() / 60
            next_refresh = max(0, refresh_interval - elapsed)
            st.write(f"Next refresh in: {next_refresh:.1f} minutes")
    
    if st.button("Analyze Air Quality", key="main_analyze_btn"):
        loc = st.session_state.location_input.strip() or st.session_state.location
        trigger_analysis(loc, days_for_model)

# Force rerun if a button was clicked
if st.session_state.button_clicked:
    st.session_state.button_clicked = False
    st.rerun()

# Check if auto-refresh is needed
if 'auto_refresh' in locals() and auto_refresh and st.session_state.analyze_clicked:
    elapsed_time = (datetime.now() - st.session_state.last_refresh).total_seconds() / 60
    if elapsed_time >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Display real-time indicator
if st.session_state.analyze_clicked:
    st.write(f"📊 Last data update: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Only run analysis if button is clicked
if st.session_state.analyze_clicked:
    location = st.session_state.location
    days = st.session_state.days
    
    with st.spinner(f"Analyzing air quality for {location} in real-time..."):
        try:
            # Fetch location coordinates
            lat, lon, place_name = fetch_coordinates(location)
            
            if lat is None or lon is None:
                st.error(f"Could not find coordinates for {location}. Please check the spelling or try another location.")
                st.stop()
            
            # Fetch current air quality - with real-time validation
            current_aqi, additional_pollutants = fetch_air_quality_data(location)
            
            if current_aqi is None:
                st.error("Could not retrieve real-time air quality data. The analysis will continue with limited information.")
                current_aqi = 0  # Placeholder for UI display
            
            # Check data freshness
            if "data_age_minutes" in additional_pollutants:
                if additional_pollutants["data_age_minutes"] > 60:  # Data older than 1 hour
                    st.warning(f"⚠️ Air quality data is {additional_pollutants['data_age_minutes']:.1f} minutes old. This may not represent current conditions.")
                else:
                    st.success(f"✅ Air quality data is current (last updated {additional_pollutants['data_age_minutes']:.1f} minutes ago)")
            
            # Fetch current weather
            weather_data = fetch_weather_data(lat, lon)
            
            if weather_data is None:
                st.error("Could not retrieve real-time weather data. The analysis will continue with limited information.")
                st.stop()
            
            current_temp = weather_data['temperature_2m'][0] if weather_data['temperature_2m'] else None
            current_humidity = weather_data['relative_humidity_2m'][0] if weather_data['relative_humidity_2m'] else None
            
            # Show current time from weather data
            if "current_weather" in weather_data and "time" in weather_data["current_weather"]:
                current_weather_time = datetime.fromtimestamp(weather_data["current_weather"]["time"])
                st.write(f"🌡️ Current weather as of: {current_weather_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Prepare dataset and train model
            dataset = prepare_dataset(location, lat, lon, days=days)
            model, mae, pred_vs_actual, feature_importances = train_model(dataset, location)
            
            # Generate health advisories
            status, health_advisory = generate_health_advisory(current_aqi)
            health_symptoms = generate_health_symptoms(current_aqi)
            
            # Save model and feature names
            feature_names = ["hour", "temperature", "humidity", "wind_speed", "precipitation"]
            
            # Display results
            st.success(f"Real-time analysis completed for {place_name}!")
            
            # Create main dashboard
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current AQI", current_aqi)
                st.markdown(f"<div style='background-color:{get_aqi_color(current_aqi)};padding:10px;border-radius:5px;text-align:center;'><strong>Status: {status}</strong></div>", unsafe_allow_html=True)
            
            with col2:
                if current_temp is not None:
                    st.metric("Temperature", f"{current_temp} °C")
                else:
                    st.metric("Temperature", "N/A")
            
            with col3:
                if current_humidity is not None:
                    st.metric("Humidity", f"{current_humidity}%")
                else:
                    st.metric("Humidity", "N/A")
            
            # Add real-time data source information
            st.info(f"Data Sources: Air Quality data from {'WAQI' if 'source' not in additional_pollutants else additional_pollutants.get('source', 'OpenAQ')}, Weather data from Open-Meteo")
                    
            # Add download button for current data
            current_data = {
                "Location": [place_name],
                "Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                "AQI": [current_aqi],
                "Temperature": [current_temp if current_temp is not None else "N/A"],
                "Humidity": [current_humidity if current_humidity is not None else "N/A"],
                "Status": [status],
                "Health Advisory": [health_advisory],
                "Health Symptoms": [health_symptoms]
            }
            
            # Add pollutants if available
            if additional_pollutants:
                for pollutant, value in additional_pollutants.items():
                    # Skip metadata fields
                    if pollutant not in ['data_time', 'data_age_minutes', 'source']:
                        current_data[pollutant.upper()] = [value]
                    
            current_df = pd.DataFrame(current_data)
            current_csv = current_df.to_csv(index=False)
            st.download_button(
                label="Download Current Air Quality Data (CSV)",
                data=current_csv,
                file_name=f"{location}_current_aqi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
            
            # Health Information Section
            st.header("Real-time Health Information")
            
            health_col1, health_col2 = st.columns(2)
            
            with health_col1:
                st.subheader("Health Advisory")
                st.info(health_advisory)
            
            with health_col2:
                st.subheader("Potential Health Symptoms")
                st.warning(health_symptoms)
            
            # Air Quality Details
            if additional_pollutants:
                st.header("Additional Air Quality Details")
                pollutant_cols = st.columns(min(4, len(additional_pollutants)))
                
                pollutant_counter = 0
                for pollutant, value in additional_pollutants.items():
                    # Skip metadata fields
                    if pollutant not in ['data_time', 'data_age_minutes', 'source']:
                        with pollutant_cols[pollutant_counter % len(pollutant_cols)]:
                            st.metric(pollutant.upper(), f"{value:.1f}" if isinstance(value, (int, float)) else value)
                            pollutant_counter += 1
            
            # Prediction Performance
            st.header("Model Insights")
            
            model_col1, model_col2 = st.columns(2)
            
            with model_col1:
                st.subheader("Prediction Accuracy")
                st.metric("Mean Absolute Error", f"{mae:.2f}")
                
                # Predictions vs Actual Plot
                fig_pred = px.scatter(pred_vs_actual, x='Actual', y='Predicted', 
                                      title='Model Predictions vs Actual Values',
                                      labels={'Actual': 'Actual AQI', 'Predicted': 'Predicted AQI'})
                fig_pred.add_trace(go.Scatter(x=[pred_vs_actual['Actual'].min(), pred_vs_actual['Actual'].max()], 
                                             y=[pred_vs_actual['Actual'].min(), pred_vs_actual['Actual'].max()],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(color='red', dash='dash')))
                st.plotly_chart(fig_pred)
            
            with model_col2:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Hour of Day'],
                    'Importance': feature_importances
                })
                fig_imp = px.bar(importance_df, x='Feature', y='Importance',
                                title='Feature Importance in AQI Prediction',
                                color='Feature')
                st.plotly_chart(fig_imp)
            
            # AQI Forecast Section
            st.header("Real-time AQI Forecast")
            
            # Generate 24-hour forecast
            forecast_24h = get_24h_forecast(model, weather_data, feature_names)
            
            # Display forecast tabs
            forecast_tab1, forecast_tab2 = st.tabs(["24-Hour Forecast", "7-Day Forecast"])
            
            with forecast_tab1:
                st.subheader("Hourly AQI Forecast")
                
                # Create hourly forecast chart
                fig_hourly = px.line(forecast_24h, x='Time', y='Predicted AQI',
                                  title='24-Hour AQI Forecast',
                                  labels={'Predicted AQI': 'AQI', 'Time': 'Time'})
                
                # Add colored background zones for AQI levels
                fig_hourly.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
                
                st.plotly_chart(fig_hourly)
                
                # Display hourly forecast details
                st.subheader("Detailed Forecast Data")
                formatted_forecast = forecast_24h.copy()
                formatted_forecast['Time'] = formatted_forecast['Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(formatted_forecast, use_container_width=True)
                
                # Download hourly forecast data
                hourly_csv = forecast_24h.to_csv(index=False)
                st.download_button(
                    label="Download 24-Hour Forecast (CSV)",
                    data=hourly_csv,
                    file_name=f"{location}_24h_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            
            with forecast_tab2:
                st.subheader("7-Day AQI Forecast")
                
                # Generate 7-day forecast
                forecast_7day = get_7day_forecast(model, weather_data, feature_names)
                
                # Create 7-day forecast chart
                fig_7day = px.bar(forecast_7day, x='Day', y='Avg AQI',
                               title='7-Day AQI Forecast',
                               labels={'Avg AQI': 'Average AQI', 'Day': 'Day'},
                               color='Avg AQI',
                               color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple', 'maroon'])
                
                # Add error bars for min/max
                fig_7day.add_traces(
                    go.Scatter(
                        x=forecast_7day['Day'],
                        y=forecast_7day['Min AQI'],
                        mode='markers',
                        name='Min AQI',
                        marker=dict(color='blue', size=8)
                    )
                )
                
                fig_7day.add_traces(
                    go.Scatter(
                        x=forecast_7day['Day'],
                        y=forecast_7day['Max AQI'],
                        mode='markers',
                        name='Max AQI',
                        marker=dict(color='red', size=8)
                    )
                )
                
                st.plotly_chart(fig_7day)
                
                # Display 7-day forecast details
                st.subheader("Weekly Forecast Details")
                st.dataframe(forecast_7day, use_container_width=True)
                
                # Download 7-day forecast data
                weekly_csv = forecast_7day.to_csv(index=False)
                st.download_button(
                    label="Download 7-Day Forecast (CSV)",
                    data=weekly_csv,
                    file_name=f"{location}_7day_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please try another location or check your internet connection.")
