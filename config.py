"""
Configuration สำหรับ Fear & Greed Alert System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Discord Webhook URL (ใส่ใน .env file)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Fear & Greed Thresholds (BUY - ซื้อตอน Fear)
FEAR_WATCH_THRESHOLD = 25      # ต่ำกว่านี้ = น่าจับตามอง
FEAR_EXTREME_THRESHOLD = 10    # ต่ำกว่านี้ = น่าเข้าสุดๆ

# Greed Thresholds (SELL - ขายตอน Greed สูง)
GREED_WATCH_THRESHOLD = 75     # สูงกว่านี้ = น่าจับตามอง (อาจถึงจุดสูงสุด)
GREED_EXTREME_THRESHOLD = 85   # สูงกว่านี้ = น่าขายสุดๆ (Extreme Greed)

# Model Settings
MODEL_PROBABILITY_THRESHOLD = 0.6  # Prob ขั้นต่ำสำหรับ alert
LOOKBACK_DAYS = 5                  # วันที่ดู rebound
REBOUND_TARGET = 0.03              # Target return 3% (ลดจาก 5% เพื่อให้มี signal มากขึ้น)

# Data Settings
START_DATE = "2011-01-01"
TICKER = "^GSPC"  # S&P 500
VIX_TICKER = "^VIX"

# Fear & Greed Data Source
FEAR_GREED_CSV_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed-2011-2023.csv"
CNN_FEAR_GREED_API = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# Model paths
MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
