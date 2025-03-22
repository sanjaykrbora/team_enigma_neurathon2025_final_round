# Air Quality Monitor & Predictor

A real-time air quality monitoring and prediction application that provides:

- Current air quality information for any location
- Health advisories based on AQI levels
- 24-hour and 7-day air quality forecasts
- Machine learning-based predictions

## Features

- Real-time air quality data from multiple sources (WAQI, OpenAQ)
- Weather data integration from Open-Meteo
- AI-powered forecasting using RandomForest models
- Health recommendations based on current and predicted AQI
- Interactive visualizations with Plotly
- Auto-refreshing capabilities for live monitoring

## Installation

1. Ensure you have Python 3.8 or newer installed
2. Clone or download this repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`
5. Alternatively, double-click the `air_guard_open.bat` file (Windows)

## Usage

1. Enter a city, state, or country name
2. Click "Analyze Air Quality" or select one of the popular locations
3. View current air quality information and forecasts
4. Enable auto-refresh for continuous monitoring

## Data Sources

- Air quality data: WAQI and OpenAQ APIs
- Weather data: Open-Meteo API
- Geocoding: Photon API

## Technologies Used

- Python with Streamlit for user interface
- scikit-learn for machine learning models
- Pandas for data manipulation
- Plotly for interactive visualizations
- Weights & Biases for model tracking
