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

# Page Header
st.title("🌬️ Air Quality Monitor & Predictor")
st.markdown("""
This application monitors air quality in real-time and provides health advisories based on AI predictions.
Enter a location to get started.
""")

# Add auto-refresh checkbox to sidebar
with st.sidebar:
    st.header("Location Settings")
    location_input = st.text_input("Enter city, state, or country:", "")
    
   # Add some popular cities for quick selection
st.markdown("### Popular Locations")
popular_cities = ["New York", "London", "Tokyo", "Beijing", "Delhi", "Sydney", "Paris", "Cape Town", "Rio de Janeiro", "Moscow"]

# Create 2 columns for city buttons
col1, col2 = st.columns(2)

# Initialize session state for button clicks if not already
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

for i, city in enumerate(popular_cities):
    if i % 2 == 0:
        with col1:
            if st.button(city, key=f"btn_{city}"):
                st.session_state.location = city
                st.session_state.analyze_clicked = True
                st.session_state.button_clicked = True
    else:
        with col2:
            if st.button(city, key=f"btn_{city}"):
                st.session_state.location = city
                st.session_state.analyze_clicked = True
                st.session_state.button_clicked = True

# Force rerun if a button was clicked
if st.session_state.button_clicked:
    st.session_state.button_clicked = False
    st.experimental_rerun()

# Check if auto-refresh is needed
if auto_refresh and st.session_state.analyze_clicked:
    elapsed_time = (datetime.now() - st.session_state.last_refresh).total_seconds() / 60
    if elapsed_time >= refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()  # Force a rerun of the app to refresh data

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
            
            # Tabs for Hourly and Weekly forecasts
            hourly_tab, weekly_tab = st.tabs(["Hourly Forecast", "Weekly Forecast"])
            
            with hourly_tab:
                # Generate hourly forecasts based on real-time weather data
                hourly_forecast = get_24h_forecast(model, weather_data, feature_names)
                
                # Plot the hourly forecast
                fig_hourly = px.line(hourly_forecast, x='Time', y='Predicted AQI',
                                    title='Real-time 24-Hour AQI Forecast',
                                    labels={'Time': 'Time', 'Predicted AQI': 'Predicted AQI'})
                
                # Add color bands for AQI categories
                fig_hourly.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
                fig_hourly.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
                
                st.plotly_chart(fig_hourly)
                
                # Display hourly forecast data with health advisories
                st.subheader("Hourly Forecast Details")
                hourly_display = hourly_forecast[['Time', 'Predicted AQI', 'Status', 'Health Advisory', 'Temperature', 'Humidity']].copy()
                hourly_display['Time'] = hourly_display['Time'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(hourly_display)
                
                # Download hourly forecast
                hourly_csv = hourly_display.to_csv(index=False)
                st.download_button(
                    label="Download Real-time Hourly Forecast (CSV)",
                    data=hourly_csv,
                    file_name=f"{location}_hourly_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
                
            with weekly_tab:
                # Generate 7-day forecast based on real-time weather data
                weekly_forecast = get_7day_forecast(model, weather_data, feature_names)
                
                # Plot the weekly forecast
                fig_weekly = px.bar(weekly_forecast, x='Day', y='Avg AQI',
                                   title='Real-time 7-Day AQI Forecast',
                                   color='Status',
                                   color_discrete_map={
                                       'Good': 'green',
                                       'Moderate': 'yellow',
                                       'Unhealthy for Sensitive Groups': 'orange',
                                       'Unhealthy': 'red',
                                       'Very Unhealthy': 'purple',
                                       'Hazardous': 'maroon'
                                   })
                
                # Add error bars for min/max
                fig_weekly.add_trace(go.Scatter(
                    name='AQI Range',
                    x=weekly_forecast['Day'],
                    y=weekly_forecast['Min AQI'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_weekly.add_trace(go.Scatter(
                    name='AQI Range',
                    x=weekly_forecast['Day'],
                    y=weekly_forecast['Max AQI'],
                    mode='lines',
                    fill='tonexty',
                    line=dict(width=0),
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    showlegend=True
                ))
                
                st.plotly_chart(fig_weekly)
                
                # Display weekly forecast data with health advisories
                st.subheader("Weekly Forecast Details")
                st.dataframe(weekly_forecast)
                
                # Download weekly forecast
                weekly_csv = weekly_forecast.to_csv(index=False)
                st.download_button(
                    label="Download Real-time Weekly Forecast (CSV)",
                    data=weekly_csv,
                    file_name=f"{location}_weekly_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
                
                # Weekly health advisory visualization
                st.subheader("Weekly Health Advisories")
                
                for index, row in weekly_forecast.iterrows():
                    with st.expander(f"{row['Day']} ({row['Date']}) - {row['Status']}"):
                        st.markdown(f"**Average AQI**: {row['Avg AQI']}")
                        st.markdown(f"**AQI Range**: {row['Min AQI']} - {row['Max AQI']}")
                        st.markdown(f"**Weather**: {row['Avg Temp']}°C, {row['Avg Humidity']}% humidity")
                        st.markdown("---")
                        st.markdown(f"**Health Advisory**: {row['Health Advisory']}")
                        
                        # Get detailed symptoms for this AQI
                        symptoms = generate_health_symptoms(row['Avg AQI'])
                        st.markdown(f"**Potential Health Symptoms**: {symptoms}")
                        
                        # Add recommendations based on AQI level
                        if row['Avg AQI'] > 150:
                            st.warning("⚠️ Consider limiting outdoor activities on this day")
                        elif row['Avg AQI'] > 100:
                            st.info("ℹ️ Sensitive groups should take precautions")
                        else:
                            st.success("✅ Air quality is acceptable for most individuals")
                
        except Exception as e:
            st.error(f"An error occurred during real-time analysis: {str(e)}")
            st.info("Please check the location name and try again.")
else:
    st.info("👈 Enter a location in the sidebar and click 'Analyze Air Quality' to begin")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This application uses real-time air quality data from WAQI and OpenAQ
and weather data from Open-Meteo to predict air quality and provide health advisories. The machine learning 
model is trained using scikit-learn's RandomForest algorithm and tracked with Weights & Biases.

**Last updated:** {datetime_now}
""".format(datetime_now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
