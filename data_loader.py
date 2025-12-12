"""
Data Loader V2 - ดึงข้อมูล Fear & Greed และ Market Data
รวม Synthetic Fear & Greed คำนวณจาก 7 ตัวแปรเหมือน CNN
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import config


def get_fear_greed_historical():
    """ดึง Fear & Greed Historical Data"""
    try:
        df = pd.read_csv(config.FEAR_GREED_CSV_URL)
        # Handle different column names
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date', 'Fear Greed': 'score'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df.asfreq('D').ffill()
        return df
    except Exception as e:
        print(f"Error loading Fear & Greed data: {e}")
        return None


# ============================================
# SYNTHETIC FEAR & GREED (NEW!)
# คำนวณเหมือน CNN จาก 7 ตัวแปร
# ============================================

def calculate_market_momentum(spx_data):
    """
    1. Market Momentum: S&P 500 vs 125-day moving average
    Score 0-100: ถ้าราคาสูงกว่า MA125 มาก = Greed, ต่ำกว่า = Fear
    """
    ma_125 = spx_data['Close'].rolling(125).mean()
    pct_diff = (spx_data['Close'] - ma_125) / ma_125 * 100
    
    # Normalize to 0-100 scale
    # -8% or below = 0 (Extreme Fear), +8% or above = 100 (Extreme Greed)
    score = 50 + (pct_diff / 8) * 50
    score = score.clip(0, 100)
    return score


def calculate_stock_price_strength(spx_data):
    """
    2. Stock Price Strength: New Highs vs New Lows
    ใช้ proxy: % of days making new 52-week high vs low
    """
    high_52w = spx_data['High'].rolling(252).max()
    low_52w = spx_data['Low'].rolling(252).min()
    
    # Distance from 52w high/low
    dist_from_high = (spx_data['Close'] - high_52w) / high_52w
    dist_from_low = (spx_data['Close'] - low_52w) / low_52w
    
    # Score: near high = greed, near low = fear
    # Normalize: at 52w high = 100, at 52w low = 0
    score = (dist_from_low / (dist_from_low - dist_from_high + 0.001)) * 100
    score = score.clip(0, 100)
    return score


def calculate_market_volatility(vix_data):
    """
    3. Market Volatility: VIX level
    VIX < 12 = Extreme Greed (100), VIX > 35 = Extreme Fear (0)
    """
    # Inverse relationship: high VIX = fear
    score = 100 - ((vix_data - 12) / (35 - 12)) * 100
    score = score.clip(0, 100)
    return score


def calculate_safe_haven_demand(spx_data, bond_data):
    """
    4. Safe Haven Demand: Stock vs Bond performance
    ถ้าหุ้นดีกว่า Bond = Greed, Bond ดีกว่า = Fear
    """
    if bond_data is None or bond_data.empty:
        return pd.Series(50, index=spx_data.index)
    
    # 20-day relative performance
    spx_return = spx_data['Close'].pct_change(20)
    bond_return = bond_data['Close'].pct_change(20)
    
    # Align indices
    common_idx = spx_return.index.intersection(bond_return.index)
    spx_return = spx_return.loc[common_idx]
    bond_return = bond_return.loc[common_idx]
    
    diff = (spx_return - bond_return) * 100
    
    # Normalize: -5% diff = 0, +5% diff = 100
    score = 50 + (diff / 5) * 50
    score = score.clip(0, 100)
    
    # Reindex to original
    return score.reindex(spx_data.index).ffill().fillna(50)


def calculate_junk_bond_demand(hyg_data, lqd_data):
    """
    5. Junk Bond Demand: HYG (High Yield) vs LQD (Investment Grade)
    ถ้า Junk bonds ดีกว่า = Risk-on = Greed
    """
    if hyg_data is None or lqd_data is None:
        return pd.Series(50, index=hyg_data.index if hyg_data is not None else pd.DatetimeIndex([]))
    
    # 20-day relative performance
    hyg_return = hyg_data['Close'].pct_change(20)
    lqd_return = lqd_data['Close'].pct_change(20)
    
    common_idx = hyg_return.index.intersection(lqd_return.index)
    hyg_return = hyg_return.loc[common_idx]
    lqd_return = lqd_return.loc[common_idx]
    
    diff = (hyg_return - lqd_return) * 100
    
    # Normalize
    score = 50 + (diff / 3) * 50
    score = score.clip(0, 100)
    
    return score.reindex(hyg_data.index).ffill().fillna(50)


def calculate_put_call_ratio(vix_data):
    """
    6. Put/Call Options: ใช้ VIX term structure เป็น proxy
    VIX สูง = มีคนซื้อ Put เยอะ = Fear
    """
    # Use VIX change as proxy for put/call sentiment
    vix_ma = vix_data.rolling(20).mean()
    ratio = vix_data / vix_ma
    
    # ratio > 1.2 = fear, ratio < 0.8 = greed
    score = 100 - ((ratio - 0.8) / (1.2 - 0.8)) * 100
    score = score.clip(0, 100)
    return score


def calculate_market_breadth(spx_data):
    """
    7. Market Breadth: ใช้ price momentum เป็น proxy
    ถ้าราคาขึ้นต่อเนื่อง = breadth ดี = greed
    """
    # Rolling positive days ratio
    returns = spx_data['Close'].pct_change()
    positive_days = (returns > 0).rolling(20).sum() / 20 * 100
    
    # Normalize: 30% positive = 0, 70% positive = 100
    score = (positive_days - 30) / (70 - 30) * 100
    score = score.clip(0, 100)
    return score


def calculate_synthetic_fear_greed(start_date=None, end_date=None):
    """
    คำนวณ Synthetic Fear & Greed Index จาก 7 ตัวแปร
    เหมือน CNN Fear & Greed Index
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Calculating Synthetic Fear & Greed Index...")
    
    # Download required data
    spx = yf.download(config.TICKER, start=start_date, end=end_date, progress=False)
    vix = yf.download(config.VIX_TICKER, start=start_date, end=end_date, progress=False)
    
    # Additional data for components
    tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)  # Treasury bonds
    hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)  # High yield
    lqd = yf.download('LQD', start=start_date, end=end_date, progress=False)  # Investment grade
    
    # Handle multi-level columns
    for df in [spx, vix, tlt, hyg, lqd]:
        if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    
    if spx.empty or vix.empty:
        print("Error: Could not download required data")
        return None
    
    # Calculate each component
    components = pd.DataFrame(index=spx.index)
    
    # 1. Market Momentum (weight: 25%)
    components['momentum'] = calculate_market_momentum(spx)
    
    # 2. Stock Price Strength (weight: 25%)
    components['strength'] = calculate_stock_price_strength(spx)
    
    # 3. Market Volatility - VIX (weight: 15%)
    vix_aligned = vix['Close'].reindex(spx.index).ffill()
    components['volatility'] = calculate_market_volatility(vix_aligned)
    
    # 4. Safe Haven Demand (weight: 10%)
    components['safe_haven'] = calculate_safe_haven_demand(spx, tlt)
    
    # 5. Junk Bond Demand (weight: 10%)
    components['junk_bond'] = calculate_junk_bond_demand(hyg, lqd)
    
    # 6. Put/Call Ratio proxy (weight: 10%)
    components['put_call'] = calculate_put_call_ratio(vix_aligned)
    
    # 7. Market Breadth (weight: 5%)
    components['breadth'] = calculate_market_breadth(spx)
    
    # Weighted average (เหมือน CNN)
    weights = {
        'momentum': 0.25,
        'strength': 0.25,
        'volatility': 0.15,
        'safe_haven': 0.10,
        'junk_bond': 0.10,
        'put_call': 0.10,
        'breadth': 0.05
    }
    
    components['synthetic_score'] = sum(
        components[col] * weight 
        for col, weight in weights.items()
    )
    
    # Smooth with 3-day MA to reduce noise
    components['score'] = components['synthetic_score'].rolling(3).mean()
    
    print(f"Synthetic Fear & Greed calculated: {len(components)} days")
    print(f"Current Synthetic Score: {components['score'].iloc[-1]:.1f}")
    
    return components[['score']].dropna()


def get_current_fear_greed():
    """
    ดึง Fear & Greed ล่าสุด
    Priority: 1. Synthetic (แม่นที่สุด) 2. VIX estimate
    """
    try:
        # Method 1: Calculate Synthetic Fear & Greed (BEST!)
        # ใช้ข้อมูล 1 ปีล่าสุดเพื่อคำนวณ
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        
        synthetic = calculate_synthetic_fear_greed(start_date, end_date)
        if synthetic is not None and not synthetic.empty:
            score = synthetic['score'].iloc[-1]
            if pd.notna(score):
                score = float(score)
                rating = get_fear_rating(score)
                return {
                    'score': score, 
                    'rating': rating, 
                    'date': datetime.now(), 
                    'source': 'synthetic_7_factors'
                }
        
        # Method 2: Fallback to VIX estimate
        vix = yf.download(config.VIX_TICKER, period='5d', progress=False)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_val = vix['Close'].iloc[-1]
            if hasattr(vix_val, 'item'):
                vix_val = vix_val.item()
            
            # VIX to Fear estimate: VIX 12=85, VIX 35=15
            estimated_fear = 100 - ((vix_val - 12) / (35 - 12)) * 85
            estimated_fear = max(0, min(100, estimated_fear))
            rating = get_fear_rating(estimated_fear)
            return {
                'score': estimated_fear, 
                'rating': rating, 
                'date': datetime.now(), 
                'source': 'vix_estimate'
            }
        
        return None
    except Exception as e:
        print(f"Error fetching current Fear & Greed: {e}")
        return None


def get_fear_rating(score):
    """แปลง score เป็น rating"""
    if score < 25:
        return "Extreme Fear"
    elif score < 45:
        return "Fear"
    elif score < 55:
        return "Neutral"
    elif score < 75:
        return "Greed"
    else:
        return "Extreme Greed"


def get_market_data(start_date=None, end_date=None):
    """ดึงข้อมูล S&P 500 และ VIX"""
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download S&P 500
    spx = yf.download(config.TICKER, start=start_date, end=end_date, progress=False)
    vix = yf.download(config.VIX_TICKER, start=start_date, end=end_date, progress=False)
    
    if spx.empty:
        return None
    
    # Handle multi-level columns from yfinance
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # Calculate returns
    spx['Return'] = spx['Close'].pct_change()
    
    # Technical Indicators
    spx['RSI'] = RSIIndicator(spx['Close'], window=14).rsi()
    
    macd = MACD(spx['Close'])
    spx['MACD'] = macd.macd()
    spx['MACD_Signal'] = macd.macd_signal()
    spx['MACD_Diff'] = macd.macd_diff()
    
    bb = BollingerBands(spx['Close'])
    spx['BB_High'] = bb.bollinger_hband()
    spx['BB_Low'] = bb.bollinger_lband()
    spx['BB_Pct'] = bb.bollinger_pband()
    
    # Moving Averages
    spx['SMA_20'] = spx['Close'].rolling(20).mean()
    spx['SMA_50'] = spx['Close'].rolling(50).mean()
    spx['SMA_200'] = spx['Close'].rolling(200).mean()
    
    # Price vs MAs
    spx['Price_vs_SMA20'] = (spx['Close'] - spx['SMA_20']) / spx['SMA_20']
    spx['Price_vs_SMA50'] = (spx['Close'] - spx['SMA_50']) / spx['SMA_50']
    
    # Add VIX
    spx['VIX'] = vix['Close']
    
    # Volume features
    spx['Volume_SMA'] = spx['Volume'].rolling(20).mean()
    spx['Volume_Ratio'] = spx['Volume'] / spx['Volume_SMA']
    
    return spx.dropna()


# ============================================
# ECONOMIC CALENDAR (NEW!)
# ดึงวัน FOMC และ CPI
# ============================================

def get_fomc_dates():
    """
    วัน FOMC Meeting 2024-2025
    Source: Federal Reserve Calendar
    """
    fomc_dates = [
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        # 2026
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ]
    return [pd.Timestamp(d) for d in fomc_dates]


def get_cpi_dates():
    """
    วัน CPI Release 2024-2025 (ประมาณกลางเดือน)
    """
    cpi_dates = [
        # 2024
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
        "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
        "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
        # 2025
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
        "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
        "2025-09-10", "2025-10-10", "2025-11-13", "2025-12-10",
    ]
    return [pd.Timestamp(d) for d in cpi_dates]


def get_days_to_event(date, event_dates):
    """คำนวณจำนวนวันถึง event ถัดไป"""
    date = pd.Timestamp(date)
    future_events = [d for d in event_dates if d > date]
    if future_events:
        return (future_events[0] - date).days
    return 999  # No upcoming event


def add_economic_calendar_features(data):
    """
    เพิ่ม Features จากปฏิทินเศรษฐกิจ
    """
    fomc_dates = get_fomc_dates()
    cpi_dates = get_cpi_dates()
    
    data = data.copy()
    
    # Days to FOMC
    data['Days_To_FOMC'] = data.index.map(lambda x: get_days_to_event(x, fomc_dates))
    data['FOMC_Week'] = (data['Days_To_FOMC'] <= 7).astype(int)
    data['FOMC_Tomorrow'] = (data['Days_To_FOMC'] <= 1).astype(int)
    
    # Days to CPI
    data['Days_To_CPI'] = data.index.map(lambda x: get_days_to_event(x, cpi_dates))
    data['CPI_Week'] = (data['Days_To_CPI'] <= 7).astype(int)
    data['CPI_Tomorrow'] = (data['Days_To_CPI'] <= 1).astype(int)
    
    # High Event Risk: FOMC or CPI within 2 days
    data['High_Event_Risk'] = ((data['Days_To_FOMC'] <= 2) | (data['Days_To_CPI'] <= 2)).astype(int)
    
    # Avoid Signal: ไม่ควรออก signal ก่อน event สำคัญ
    data['Avoid_Signal'] = ((data['Days_To_FOMC'] <= 1) | (data['Days_To_CPI'] <= 1)).astype(int)
    
    return data


# ============================================
# COMBINED DATA LOADER
# ============================================

def get_full_market_data(start_date=None, end_date=None, use_synthetic_fg=True):
    """
    โหลดข้อมูลตลาดพร้อม Synthetic Fear & Greed และ Economic Calendar
    """
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get market data
    market_data = get_market_data(start_date, end_date)
    if market_data is None:
        return None
    
    # Get Fear & Greed data
    if use_synthetic_fg:
        # Use synthetic (more accurate)
        fg_data = calculate_synthetic_fear_greed(start_date, end_date)
        if fg_data is None:
            # Fallback to historical
            fg_data = get_fear_greed_historical()
    else:
        fg_data = get_fear_greed_historical()
    
    if fg_data is None:
        return None
    
    # Merge
    data = market_data.join(fg_data['score'], how='inner')
    data = data.rename(columns={'score': 'FearIndex'})
    
    # Add economic calendar features
    data = add_economic_calendar_features(data)
    
    return data
