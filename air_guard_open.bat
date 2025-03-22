@echo off
echo Starting Air Quality Monitor...
cd /d "%~dp0"
start http://localhost:5000
streamlit run app.py --server.address localhost --server.port 5000