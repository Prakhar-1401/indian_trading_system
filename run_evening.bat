@echo off
REM Evening Portfolio Check — Runs at 3:45 PM IST (Mon-Fri)
cd /d C:\Users\Z00587UT\indian-trading-system
call venv\Scripts\activate.bat
python daily_morning.py evening
