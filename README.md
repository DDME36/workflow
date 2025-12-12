<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Alert-Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" />
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />
</p>

<h1 align="center">📊 Fear & Greed Alert System</h1>

<p align="center">
  <strong>AI-Powered S&P 500 Trading Signal Generator</strong><br>
  ระบบแจ้งเตือนสัญญาณซื้อ-ขาย S&P 500 อัตโนมัติ โดยใช้ Machine Learning วิเคราะห์ Fear & Greed Index
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-backtest-results">Results</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

## 🎯 Overview

ระบบนี้ใช้หลักการ **"Buy when others are fearful, sell when others are greedy"** โดยวิเคราะห์ตลาดผ่าน:

- 🧠 **Synthetic Fear & Greed Index** - คำนวณจาก 7 ปัจจัยเหมือน CNN
- 🤖 **Voting Ensemble ML Model** - รวม XGBoost, LightGBM, Random Forest
- 🛡️ **Circuit Breaker** - ป้องกันการเทรดในช่วงตลาดผันผวน
- 📈 **Dynamic Exit Strategy** - Trailing Stop, Take Profit, Stagnation Kill

---

## ✨ Features

### 🔮 Synthetic Fear & Greed (7 Factors)
คำนวณ Fear & Greed Index เองจาก 7 ปัจจัยเหมือน CNN:

| Factor | Weight | Description |
|--------|--------|-------------|
| Market Momentum | 25% | S&P 500 vs 125-day MA |
| Stock Price Strength | 25% | 52-week Highs vs Lows |
| Market Volatility | 15% | VIX Index |
| Safe Haven Demand | 10% | Stocks vs Bonds |
| Junk Bond Demand | 10% | JNK vs LQD Spread |
| Put/Call Ratio | 10% | Options Sentiment |
| Market Breadth | 5% | Advance/Decline Ratio |

### 🛡️ Risk Management

```
┌─────────────────────────────────────────────────────────┐
│  CIRCUIT BREAKER (Auto-Block Trading)                   │
├─────────────────────────────────────────────────────────┤
│  🚫 Price < SMA200 AND VIX > 30  → Bear Market Panic    │
│  🚫 RSI < 25 AND MACD Dipping    → Falling Knife        │
│  🚫 VIX > 40                     → Market Crash         │
│  🚫 Days to FOMC ≤ 2             → Fed Meeting Soon     │
└─────────────────────────────────────────────────────────┘
```

### 📊 Market Regime Detection

| Regime | Condition | BUY Threshold | Strategy |
|--------|-----------|---------------|----------|
| 🐂 Bull | Price > SMA200 | Prob > 50% | Buy the Dip |
| 🐻 Bear | Price < SMA200 | Prob > 70% + Fear < 20 | Wait for Panic |

### 📤 Dynamic Exit Strategy

```
EXIT RULES:
├── 🛑 Stop Loss: -4% → ขายทันที
├── ⏰ Stagnation: 3 วันไม่กำไร > 1% → ขายทิ้ง
├── 🎯 Take Profit: RSI > 70 หรือ Fear > 70 → ขาย
├── 📈 Trailing Stop: กำไร > 3% → Stop ที่ทุน
│                     กำไร > 5% → Trail 2%
└── ⏳ Max Hold: 10 วัน
```

---

## 📈 Backtest Results

### Walk-Forward Validation (2019-2024)

```
╔═══════════════════════════════════════════════════════════╗
║                    PERFORMANCE SUMMARY                     ║
╠═══════════════════════════════════════════════════════════╣
║  Total Trades:        52                                   ║
║  Win Rate:            57.69%                               ║
║  Total Return:        +42.40%                              ║
║  Max Drawdown:        -19.41% ✅ (Target: < -25%)          ║
║  Profit Factor:       1.80                                 ║
║  Avg Trade:           +0.82%                               ║
╚═══════════════════════════════════════════════════════════╝
```

### Model Precision

| Model | Target | Precision | Recall |
|-------|--------|-----------|--------|
| BUY | Conservative_Buy | **100%** | 15% |
| SELL | Confirmed_Sell | **67%** | 40% |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/DDME36/workflow.git
cd workflow
pip install -r requirements.txt
```

### 2. Configure Discord Webhook

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your Discord webhook
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/xxx
```

### 3. Train Model (Optional)

```bash
python train_v2.py
```

### 4. Run Daily Check

```bash
python daily_check_v2.py
```

---

## 📁 Project Structure

```
fear_greed_alert/
├── 📊 Core System
│   ├── daily_check_v2.py      # Main daily signal checker
│   ├── train_v2.py            # Model training script
│   └── backtest.py            # Walk-forward backtesting
│
├── 🧠 ML Components
│   ├── models_v2.py           # Voting Ensemble model
│   ├── feature_engineering_v2.py  # 32 pruned features
│   └── fear_predictor.py      # 5-day fear forecast
│
├── 📡 Data & Alerts
│   ├── data_loader.py         # Synthetic Fear & Greed + yfinance
│   ├── discord_alert.py       # Rich Discord notifications
│   └── config.py              # Configuration
│
├── 🤖 Automation
│   ├── .github/workflows/     # GitHub Actions (daily 05:00 ICT)
│   ├── run_daily.pyw          # Windows silent runner
│   └── setup_task.bat         # Windows Task Scheduler
│
└── 📦 Models
    └── models/                # Trained model files (.pkl)
```

---

## ☁️ Deployment

### GitHub Actions (Recommended)

ระบบจะรันอัตโนมัติทุกวัน **05:00 เวลาไทย** (จันทร์-ศุกร์)

#### Setup:

1. **Add Secret:**
   - Go to: `Settings` → `Secrets and variables` → `Actions`
   - Add: `DISCORD_WEBHOOK_URL` = your webhook URL

2. **Enable Actions:**
   - Go to: `Actions` tab
   - Click: "I understand my workflows, go ahead and enable them"

3. **Manual Run:**
   - Go to: `Actions` → `Daily Fear & Greed Check`
   - Click: `Run workflow`

### Windows Task Scheduler

```batch
# Run setup script
setup_task.bat
```

---

## 📱 Discord Alert Preview

```
┌────────────────────────────────────────────┐
│  🚨 EXTREME FEAR - BUY ALERT               │
│  ตลาดกลัวสุดขีด! นี่คือโอกาสที่หายากมาก      │
├────────────────────────────────────────────┤
│  📊 Signal      📈 Regime     🎯 Confidence │
│  🟢 BUY         🐂 BULL       75%          │
├────────────────────────────────────────────┤
│  ━━━━━ FEAR & GREED ━━━━━                  │
│  🔴🔴🔴🔴🔴 EXTREME FEAR                    │
│          15 / 100                          │
├────────────────────────────────────────────┤
│  💹 S&P 500    😱 VIX    📊 RSI            │
│  $4,200        28.5      25               │
├────────────────────────────────────────────┤
│  🔮 5-DAY FORECAST:                        │
│  Current: 15 → Predicted: 12              │
│  Direction: 📉 ลดลง                        │
├────────────────────────────────────────────┤
│  📤 EXIT RULES:                            │
│  🛑 Stop: -4% → ขายทันที                    │
│  ⏰ 3 วันไม่กำไร > 1% → ขายทิ้ง              │
│  🎯 RSI > 70 หรือ Fear > 70 → ขาย          │
└────────────────────────────────────────────┘
```

---

## ⚠️ Disclaimer

```
⚠️ IMPORTANT: This is for educational purposes only.
   Past performance does not guarantee future results.
   Always do your own research before trading.
   
   ระบบนี้สร้างขึ้นเพื่อการศึกษาเท่านั้น
   ผลการทดสอบในอดีตไม่ได้รับประกันผลลัพธ์ในอนาคต
   กรุณาศึกษาและตัดสินใจด้วยตัวเองก่อนลงทุน
```

---

## 📄 License

MIT License - feel free to use and modify.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/DDME36">DDME36</a>
</p>
