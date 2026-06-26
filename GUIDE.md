# Indian Stock Market Trading System — Complete Guide

> **DISCLAIMER**: This is an educational/research tool. NOT financial advice. 
> Past performance does not guarantee future results. Always do your own research.

---

## STEP 1: Install Python (You Need To Do This)

1. Download Python 3.11+ from https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Restart your terminal after installation
4. Verify: `python --version` should show 3.11+

---

## STEP 2: Set Up the Project
```powershell
cd C:\Users\Z00587UT\indian-trading-system
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `requirements.txt` gives errors, install core packages:
```powershell
pip install yfinance pandas numpy requests beautifulsoup4 vaderSentiment feedparser loguru python-dotenv pyyaml plotly streamlit schedule lxml html5lib
```

---

## STEP 3: Get ICICI Direct API Keys (You Need To Do This)

### 3a. Register for Breeze API
1. Go to https://api.icicidirect.com/apiuser/home
2. Log in with your ICICI Direct trading account
3. Click **"Register Now"** (or **"View"** if already registered)
4. You'll get:
   - **API Key** (permanent, doesn't change)
   - **API Secret** (permanent)

### 3b. Generate Session Token (DAILY)
The session token expires every day. You need to regenerate it:
1. Go to https://api.icicidirect.com/apiuser/login
2. Log in → you'll be redirected to a URL containing `apisession=XXXXX`
3. That `XXXXX` is your session token

### 3c. Put Keys in .env File
Edit `C:\Users\Z00587UT\indian-trading-system\.env`:
```
ICICI_API_KEY=your_actual_api_key
ICICI_API_SECRET=your_actual_secret
ICICI_SESSION_TOKEN=your_daily_token
```

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")> **Note**: The system works WITHOUT ICICI API keys too! It falls back to 
> Yahoo Finance (yfinance) for data. ICICI is only needed for live trading.

---

## STEP 4: Run the System

### Quick Stock Ranking (Start Here!)
```powershell
cd C:\Users\Z00587UT\indian-trading-system
.\venv\Scripts\Activate.ps1
python main.py rank
```
This scores 25 stocks on momentum + quality and shows the ranking.

### Generate Trading Signals
```powershell
python main.py signals
```
Shows BUY/SELL/HOLD signals for the top stocks.

### Run Backtest
```powershell
python main.py backtest
```
Tests the strategy on 2022-2025 data with ₹10 Lakh starting capital.

### Check Market Sentiment
```powershell
python main.py sentiment
```
Scrapes financial news and shows overall market mood.

### Launch Visual Dashboard
```powershell
python main.py dashboard
```
Opens a Streamlit web app at http://localhost:8501 with charts, rankings, and signals.

---

## STEP 5: Understanding the Strategy

### The Multi-Factor Approach
We score each stock on 4 factors (each 0-10):

| Factor | Weight | What It Measures |
|--------|--------|-----------------|
| Momentum (40%) | Price trend | RSI, MACD, moving averages, returns |
| Quality (25%) | Company strength | ROE, debt, profit growth |
| Sentiment (20%) | Market mood | News analysis via NLP |
| Smart Money (15%) | Big players | FII/DII flows, insider trades |

**Composite Score** = Σ(weight × factor_score)

### When Factors Conflict
Example: Great momentum BUT poor quality:
- Momentum: 8/10 × 0.40 = 3.2
- Quality: 2/10 × 0.25 = 0.5
- **Total: 3.7** → Low rank, won't be picked

Balanced stock wins:
- Momentum: 6/10 × 0.40 = 2.4
- Quality: 7/10 × 0.25 = 1.75  
- **Total: 4.15** → Higher rank!

### Key Parameters to Tune (in config/strategy.yaml)
- **factor_weights**: Increase momentum in bull markets, quality in bear markets
- **stop_loss_pct**: 8% default. Tighter = fewer losses but more whipsaws
- **max_stocks**: 15 default. More stocks = more diversified but diluted returns
- **rebalance_frequency**: Weekly default. Monthly = lower fees, less responsive

---

## STEP 6: Backtesting Deep Dive

### What to Look For in Results
| Metric | Good | Great | Bad |
|--------|------|-------|-----|
| CAGR | >12% | >20% | <8% |
| Sharpe Ratio | >1.0 | >2.0 | <0.5 |
| Max Drawdown | >-15% | >-10% | <-25% |
| Win Rate | >50% | >60% | <40% |
| Profit Factor | >1.5 | >2.0 | <1.0 |

### How to Improve Results
1. **Change weights**: Edit `config/strategy.yaml` → `factor_weights`
2. **Tighten stops**: Reduce `stop_loss_pct` from 8% to 5%
3. **Change universe**: Try sector-specific (only IT, only Banking)
4. **Change rebalance**: Try monthly instead of weekly
5. **Run backtest again** and compare metrics

### Other Backtesting Platforms (If You Want to Compare)
1. **Zerodha Streak** (https://streak.zerodha.com) — Web-based, visual, simpler
2. **TradingView** (https://tradingview.com) — Pine Script for technical strategies
3. **Backtrader** (Python) — Already in our requirements, very flexible
4. **QuantConnect** (https://quantconnect.com) — Cloud-based, professional grade

---

## STEP 7: Cloud Deployment

### Option A: Run on a VPS (Cheapest, Recommended)
1. Get a VPS from DigitalOcean/AWS Lightsail (~$5/month)
2. SSH into the server
3. Install Docker:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```
4. Clone/upload your project
5. Copy your `.env` file to the server
6. Run:
```bash
docker-compose up -d
```
7. Dashboard available at `http://your-server-ip:8501`

### Option B: AWS Free Tier
1. Create EC2 t2.micro instance (free for 12 months)
2. Install Python 3.11, clone repo, set up venv
3. Run scheduler as a systemd service:
```bash
# /etc/systemd/system/trading-scheduler.service
[Unit]
Description=Trading Strategy Scheduler
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/indian-trading-system
ExecStart=/home/ubuntu/indian-trading-system/venv/bin/python scheduler.py
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable trading-scheduler
sudo systemctl start trading-scheduler
```

### Option C: Railway/Render (Easiest)
1. Push code to GitHub (WITHOUT .env!)
2. Go to https://railway.app or https://render.com
3. Connect your GitHub repo
4. Add environment variables in the web UI
5. Deploy — it auto-builds from Dockerfile

---

## STEP 8: Daily Workflow

### Morning (Before Market Opens — 9:00 AM)
1. Update ICICI session token in `.env` (if using live trading)
2. Run `python main.py signals` to see today's signals
3. Review the signals, cross-check with your own analysis
4. Place trades manually on ICICI Direct app

### Evening (After Market Closes — 4:00 PM)
1. Run `python main.py sentiment` to see next-day mood
2. Check the dashboard for updated rankings

### Weekly (Monday Morning)
1. The system auto-rebalances signals
2. Review which stocks entered/exited the portfolio
3. Place rebalance trades

---

## File Structure Reference

```
indian-trading-system/
├── main.py                  # Main entry point (rank/signals/backtest/sentiment/dashboard)
├── scheduler.py             # Automated daily job scheduler
├── config/
│   └── strategy.yaml        # ALL strategy parameters (weights, stops, etc.)
├── src/
│   ├── data/
│   │   └── fetcher.py       # ICICI Direct API + yfinance data fetching
│   ├── indicators/
│   │   └── technical.py     # RSI, MACD, moving averages, momentum scoring
│   ├── sentiment/
│   │   └── news_analyzer.py # News scraping + VADER sentiment analysis
│   ├── smart_money/
│   │   └── tracker.py       # FII/DII, insider trades, politician trades
│   ├── ranking/
│   │   └── ranker.py        # Multi-factor scoring + stock ranking engine
│   ├── backtest/
│   │   └── engine.py        # Custom backtesting engine with performance metrics
│   ├── strategy/
│   │   └── executor.py      # Signal generation (BUY/SELL/HOLD)
│   ├── dashboard/
│   │   └── app.py           # Streamlit web dashboard
│   └── utils/
│       └── helpers.py        # Config loading, logging, environment vars
├── .env                     # Your API keys (NEVER commit this!)
├── .env.example             # Template showing what keys are needed
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container for deployment
└── docker-compose.yml       # Multi-container setup (dashboard + scheduler)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No data for symbol" | Check symbol spelling. Indian stocks need exact NSE symbol |
| Breeze API error | Session token expired. Regenerate daily |
| Slow ranking | Normal for 500 stocks. Use `rank_quick()` for faster results |
| Sentiment returns 0 | Stock might not have recent news. Try larger stocks |
| Import errors | Make sure venv is activated: `.\venv\Scripts\Activate.ps1` |
