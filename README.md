# Fear & Greed Alert System V2 🚨

ระบบแจ้งเตือนจังหวะซื้อหุ้นตาม Fear & Greed Index + ML Model

## Performance

| Metric | Value |
|--------|-------|
| Win Rate | 64.9% |
| Avg Return/Trade | +1.30% |
| Total Return (backtest) | +96.3% |
| Profit Factor | 2.78 |

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Setup Discord
แก้ไขไฟล์ `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

### 3. Train Model
```bash
python train_v2.py
```

### 4. Daily Check
```bash
python daily_check_v2.py
```

### 5. Auto Run ตอนเปิดคอม
รัน `setup_startup.bat` (Double-click)

## Files

```
fear_greed_alert/
├── config.py              # Settings
├── data_loader.py         # Load data
├── feature_engineering_v2.py  # Features
├── models_v2.py           # ML Models
├── fear_predictor.py      # Fear forecast
├── backtest.py            # Backtest
├── discord_alert.py       # Discord
├── train_v2.py            # Train
├── daily_check_v2.py      # Daily check
├── run_daily.pyw          # Silent run
└── setup_startup.bat      # Setup auto-start
```

## Alert Levels

| Fear | Level | Action |
|------|-------|--------|
| < 10 | 🚨 EXTREME | โอกาสหายาก! |
| < 25 | 👀 WATCH | จับตามอง |
| 25+ | 📊 NORMAL | ไม่มี alert |

## Best Entry Zone

จากการทดสอบ **Fear 15-20** ให้ผลดีที่สุด:
- Win Rate: 80.6%
- Avg Return: +2.38%

## Disclaimer

⚠️ ไม่ใช่คำแนะนำการลงทุน ใช้เป็น reference เท่านั้น
