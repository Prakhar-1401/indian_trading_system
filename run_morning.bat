@echo off
REM Morning Trading Routine — Runs at 8:45 AM IST (Mon-Fri)
cd /d C:\Users\Z00587UT\indian-trading-system
call venv\Scripts\activate.bat
python daily_morning.py morning
