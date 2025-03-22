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
if 'days' not in st.session_state:
    st.session_state.days = 30
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

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
        st.session_state.analyze_clicked = True
        st.session_state.location = location_input
        st.session_state.days = days_for_model
        st.session_state.last_refresh = datetime.now()

# Force rerun if a button was clicked
if st.session_state.button_clicked:
    st.session_state.button_clicked = False
    st.experimental_rerun()

# Check if auto-refresh is needed
if 'auto_refresh' in locals() and auto_refresh and st.session_state.analyze_clicked:
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
                st.dataframe(hourly_display, hide_index=True)
            
            with weekly_tab:
                # Generate 7-day forecast
                weekly_forecast = get_7day_forecast(model, weather_data, feature_names)
                
                # Create a better visualization for weekly forecast
                fig_weekly = go.Figure()
                
                # Add bar chart for average AQI
                fig_weekly.add_trace(go.Bar(
                    x=weekly_forecast['Day'],
                    y=weekly_forecast['Avg AQI'],
                    name='Average AQI',
                    marker_color=[get_aqi_color(aqi) for aqi in weekly_forecast['Avg AQI']],
                    hoverinfo='text',
                    hovertext=[f"Date: {date}<br>Avg AQI: {avg}<br>Min AQI: {min_aqi}<br>Max AQI: {max_aqi}<br>Status: {status}" 
                              for date, avg, min_aqi, max_aqi, status in zip(
                                  weekly_forecast['Date'], 
                                  weekly_forecast['Avg AQI'], 
                                  weekly_forecast['Min AQI'],
                                  weekly_forecast['Max AQI'],
                                  weekly_forecast['Status'])]
                ))
                
                # Add error bars for min/max
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_forecast['Day'],
                    y=weekly_forecast['Min AQI'],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Min AQI'
                ))
                
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_forecast['Day'],
                    y=weekly_forecast['Max AQI'],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Max AQI'
                ))
                
                # Add reference lines
                fig_weekly.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
                fig_weekly.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
                fig_weekly.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive Groups")
                
                # Update layout
                fig_weekly.update_layout(
                    title='7-Day AQI Forecast',
                    xaxis_title='Day',
                    yaxis_title='Air Quality Index (AQI)',
                    barmode='group',
                    hovermode="closest"
                )
                
                st.plotly_chart(fig_weekly)
                
                # Show detailed forecast table
                st.subheader("7-Day Forecast Details")
                weekly_display = weekly_forecast[['Day', 'Date', 'Avg AQI', 'Min AQI', 'Max AQI', 'Status', 'Avg Temp', 'Avg Humidity']]
                st.dataframe(weekly_display, hide_index=True)
                
                # Show health advisories for each day
                st.subheader("Weekly Health Advisories")
                for i, row in weekly_forecast.iterrows():
                    with st.expander(f"{row['Day']} ({row['Date']}) - {row['Status']}"):
                        st.markdown(f"**Average AQI:** {row['Avg AQI']}")
                        st.markdown(f"**Health Advisory:** {row['Health Advisory']}")
            
            # Add footer with instructions
            st.markdown("---")
            st.markdown("**Instructions:** Enter a location above or select a popular city to analyze air quality data and get health recommendations.")
            st.markdown("**Note:** The machine learning model is trained on historical data to provide forecasts. Actual conditions may vary.")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
