"""
Feature Engineering V2 - เพิ่ม Features ใหม่เพื่อความแม่นยำ
รวม Economic Calendar Features (FOMC, CPI)
"""
import pandas as pd
import numpy as np
import yfinance as yf
import config
from data_loader import add_economic_calendar_features


def get_additional_market_data(start_date=None, end_date=None):
    """ดึงข้อมูลเพิ่มเติม"""
    if start_date is None:
        start_date = config.START_DATE
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    tickers = {
        'TLT': 'Bond_20Y',      # Treasury bonds (flight to safety)
        'GLD': 'Gold',          # Gold (safe haven)
        'HYG': 'HighYield',     # High yield bonds (risk appetite)
        'XLF': 'Financials',    # Financial sector
        'XLK': 'Tech',          # Tech sector
        'IWM': 'SmallCap',      # Small cap (risk appetite)
        'EEM': 'Emerging',      # Emerging markets
    }
    
    data = pd.DataFrame()
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[f'{name}_Return'] = df['Close'].pct_change()
                data[f'{name}_SMA20'] = df['Close'].rolling(20).mean()
                data[f'{name}_vs_SMA'] = (df['Close'] - data[f'{name}_SMA20']) / data[f'{name}_SMA20']
        except Exception as e:
            print(f"Could not load {ticker}: {e}")
    
    return data


def create_features_v2(market_data, fear_data):
    """สร้าง Features V2 - เพิ่มความแม่นยำ"""
    
    # Start with basic features
    data = market_data.join(fear_data['score'], how='inner')
    data = data.rename(columns={'score': 'FearIndex'})
    
    # ============================================
    # 1. FEAR-BASED FEATURES (Enhanced)
    # ============================================
    data['Fear_Low'] = (data['FearIndex'] < config.FEAR_WATCH_THRESHOLD).astype(int)
    data['Fear_Extreme'] = (data['FearIndex'] < config.FEAR_EXTREME_THRESHOLD).astype(int)
    data['Fear_Change'] = data['FearIndex'].diff()
    data['Fear_Change_3d'] = data['FearIndex'].diff(3)
    data['Fear_Change_5d'] = data['FearIndex'].diff(5)
    
    # Fear momentum - กำลังลงหรือกำลังขึ้น
    data['Fear_SMA_5'] = data['FearIndex'].rolling(5).mean()
    data['Fear_SMA_10'] = data['FearIndex'].rolling(10).mean()
    data['Fear_Momentum'] = data['Fear_SMA_5'] - data['Fear_SMA_10']
    data['Fear_Recovering'] = (data['Fear_Change_3d'] > 0).astype(int)  # Fear กำลังฟื้น
    
    # Fear persistence - อยู่ในโซน fear นานแค่ไหน
    data['Days_In_Fear'] = (data['FearIndex'] < 25).rolling(10).sum()
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        data[f'Fear_Lag_{lag}'] = data['FearIndex'].shift(lag)
    
    # ============================================
    # 2. VOLATILITY FEATURES (Enhanced)
    # ============================================
    # VIX levels
    data['VIX_High'] = (data['VIX'] > 25).astype(int)
    data['VIX_Extreme'] = (data['VIX'] > 35).astype(int)
    data['VIX_Crash'] = (data['VIX'] > 45).astype(int)  # Crash territory - ระวัง!
    
    # VIX momentum
    data['VIX_Change'] = data['VIX'].diff()
    data['VIX_Change_5d'] = data['VIX'].diff(5)
    data['VIX_SMA_10'] = data['VIX'].rolling(10).mean()
    data['VIX_vs_SMA'] = data['VIX'] / data['VIX_SMA_10']
    data['VIX_Declining'] = (data['VIX_Change_5d'] < 0).astype(int)  # VIX กำลังลง = ดี
    
    # Realized volatility
    data['RealizedVol_10d'] = data['Return'].rolling(10).std() * np.sqrt(252)
    data['RealizedVol_20d'] = data['Return'].rolling(20).std() * np.sqrt(252)
    data['Vol_Ratio'] = data['RealizedVol_10d'] / data['RealizedVol_20d']
    
    # ============================================
    # 3. PRICE ACTION FEATURES (Enhanced)
    # ============================================
    # Returns
    for period in [1, 2, 3, 5, 10, 20]:
        data[f'Return_{period}d'] = data['Close'].pct_change(period)
    
    # Drawdown from recent high
    data['High_20d'] = data['High'].rolling(20).max()
    data['High_50d'] = data['High'].rolling(50).max()
    data['Drawdown_20d'] = (data['Close'] - data['High_20d']) / data['High_20d']
    data['Drawdown_50d'] = (data['Close'] - data['High_50d']) / data['High_50d']
    
    # Price vs Moving Averages
    data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    data['Price_vs_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
    data['Price_vs_SMA200'] = (data['Close'] - data['SMA_200']) / data['SMA_200']
    
    # Trend
    data['Uptrend'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    data['SMA20_Above_SMA50'] = (data['SMA_20'] > data['SMA_50']).astype(int)
    
    # ============================================
    # 4. TECHNICAL INDICATORS (Enhanced)
    # ============================================
    # RSI levels
    data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
    data['RSI_Extreme_Oversold'] = (data['RSI'] < 25).astype(int)
    data['RSI_Recovering'] = ((data['RSI'] > data['RSI'].shift(1)) & (data['RSI'] < 40)).astype(int)
    
    # MACD
    data['MACD_Negative'] = (data['MACD'] < 0).astype(int)
    data['MACD_Crossover'] = ((data['MACD'] > data['MACD_Signal']) & 
                              (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))).astype(int)
    
    # Bollinger Bands
    data['Below_BB_Low'] = (data['Close'] < data['BB_Low']).astype(int)
    data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['Close']
    
    # ============================================
    # 5. VOLUME FEATURES
    # ============================================
    data['Volume_Spike'] = (data['Volume_Ratio'] > 1.5).astype(int)
    data['Volume_Extreme'] = (data['Volume_Ratio'] > 2.0).astype(int)
    
    # ============================================
    # 6. COMBINED SIGNALS
    # ============================================
    # Fear + Technical alignment
    data['Fear_RSI_Align'] = ((data['FearIndex'] < 25) & (data['RSI'] < 35)).astype(int)
    data['Fear_VIX_Align'] = ((data['FearIndex'] < 25) & (data['VIX'] > 25)).astype(int)
    
    # Recovery signals
    data['Recovery_Signal'] = (
        (data['Fear_Recovering'] == 1) & 
        (data['VIX_Declining'] == 1) & 
        (data['RSI_Recovering'] == 1)
    ).astype(int)
    
    # Danger signals - ไม่ควรเข้า
    data['Crash_Warning'] = (
        (data['VIX_Crash'] == 1) | 
        (data['Drawdown_20d'] < -0.15) |
        (data['Return_5d'] < -0.10)
    ).astype(int)
    
    # ============================================
    # 6.5 MARKET REGIME DETECTION (NEW!)
    # ============================================
    # Bull Market: Price above 200 SMA, uptrend
    data['Bull_Market'] = (
        (data['Close'] > data['SMA_200']) &
        (data['SMA_50'] > data['SMA_200'])
    ).astype(int)
    
    # Bear Market: Price below 200 SMA
    data['Bear_Market'] = (
        (data['Close'] < data['SMA_200']) &
        (data['SMA_50'] < data['SMA_200'])
    ).astype(int)
    
    # Volatility Regime
    data['High_Vol_Regime'] = (data['RealizedVol_20d'] > data['RealizedVol_20d'].rolling(60).quantile(0.75)).astype(int)
    data['Low_Vol_Regime'] = (data['RealizedVol_20d'] < data['RealizedVol_20d'].rolling(60).quantile(0.25)).astype(int)
    
    # Fear Regime (rolling percentile)
    data['Fear_Regime_Low'] = (data['FearIndex'] < data['FearIndex'].rolling(60).quantile(0.2)).astype(int)
    data['Fear_Regime_High'] = (data['FearIndex'] > data['FearIndex'].rolling(60).quantile(0.8)).astype(int)
    
    # ============================================
    # 6.6 CONFIRMATION SIGNALS (NEW!)
    # ============================================
    # Strong Buy: Multiple confirmations
    data['Strong_Buy_Setup'] = (
        (data['FearIndex'] < 20) &
        (data['RSI'] < 35) &
        (data['VIX'] > 20) &
        (data['Fear_Recovering'] == 1) &
        (data['Crash_Warning'] == 0)
    ).astype(int)
    
    # Moderate Buy: Some confirmations
    data['Moderate_Buy_Setup'] = (
        (data['FearIndex'] < 25) &
        (data['RSI'] < 40) &
        (data['VIX_Declining'] == 1)
    ).astype(int)
    
    # Buy Signal Count (how many signals align)
    data['Buy_Signal_Count'] = (
        (data['FearIndex'] < 25).astype(int) +
        (data['RSI'] < 35).astype(int) +
        (data['VIX'] > 20).astype(int) +
        (data['Fear_Recovering'] == 1).astype(int) +
        (data['VIX_Declining'] == 1).astype(int) +
        (data['Below_BB_Low'] == 1).astype(int) +
        (data['MACD_Crossover'] == 1).astype(int)
    )
    
    # Strong Sell Setup
    data['Strong_Sell_Setup'] = (
        (data['FearIndex'] > 75) &
        (data['RSI'] > 70) &
        (data['VIX'] < 15) &
        (data['Price_vs_SMA20'] > 0.03)
    ).astype(int)
    
    # Sell Signal Count
    data['Sell_Signal_Count'] = (
        (data['FearIndex'] > 70).astype(int) +
        (data['RSI'] > 65).astype(int) +
        (data['VIX'] < 15).astype(int) +
        (data['Price_vs_SMA20'] > 0.03).astype(int) +
        (data['Price_vs_SMA50'] > 0.05).astype(int) +
        (data['Return_20d'] > 0.08).astype(int)
    )
    
    # ============================================
    # 6.7 MEAN REVERSION SIGNALS (NEW!)
    # ============================================
    # Z-score of price vs SMA
    data['Price_Zscore'] = (data['Close'] - data['SMA_50']) / data['Close'].rolling(50).std()
    data['Oversold_Zscore'] = (data['Price_Zscore'] < -2).astype(int)
    data['Overbought_Zscore'] = (data['Price_Zscore'] > 2).astype(int)
    
    # Fear Z-score
    data['Fear_Zscore'] = (data['FearIndex'] - data['FearIndex'].rolling(60).mean()) / data['FearIndex'].rolling(60).std()
    data['Extreme_Fear_Zscore'] = (data['Fear_Zscore'] < -1.5).astype(int)
    data['Extreme_Greed_Zscore'] = (data['Fear_Zscore'] > 1.5).astype(int)
    
    # ============================================
    # 7. TARGETS (Multiple) - BUY SIGNALS
    # ============================================
    # Future returns
    data['Future_Return_5d'] = data['Close'].shift(-5) / data['Close'] - 1
    data['Future_Return_10d'] = data['Close'].shift(-10) / data['Close'] - 1
    data['Future_Return_20d'] = data['Close'].shift(-20) / data['Close'] - 1
    
    # Max drawdown in next 10 days
    future_lows = data['Low'].rolling(10).min().shift(-10)
    data['Future_MaxDrawdown'] = (future_lows - data['Close']) / data['Close']
    
    # Risk-adjusted target: Good return without big drawdown
    data['Good_Entry'] = (
        (data['Future_Return_10d'] > 0.02) &  # At least 2% gain in 10 days
        (data['Future_MaxDrawdown'] > -0.05)   # Max drawdown < 5%
    ).astype(int)
    
    # Conservative target
    data['Safe_Entry'] = (
        (data['Future_Return_5d'] > 0) &
        (data['Future_MaxDrawdown'] > -0.03)
    ).astype(int)
    
    # Original target (for comparison)
    data['Buy_Signal'] = (
        (data['Future_Return_5d'] > config.REBOUND_TARGET) & 
        (data['FearIndex'] < config.FEAR_WATCH_THRESHOLD)
    ).astype(int)
    
    # Best target: Fear zone + good risk/reward
    data['Best_Entry'] = (
        (data['FearIndex'] < 25) &
        (data['Future_Return_10d'] > 0.03) &
        (data['Future_MaxDrawdown'] > -0.05) &
        (data['Crash_Warning'] == 0)
    ).astype(int)
    
    # ============================================
    # PREMIUM TARGETS (NEW!) - Higher precision
    # ============================================
    # Premium Buy: Very strict conditions for high precision
    data['Premium_Buy'] = (
        (data['FearIndex'] < 20) &  # Very low fear
        (data['RSI'] < 35) &  # Oversold
        (data['Future_Return_10d'] > 0.04) &  # Strong rebound
        (data['Future_MaxDrawdown'] > -0.04) &  # Limited downside
        (data['Crash_Warning'] == 0) &
        (data['Bull_Market'] == 1)  # Only in bull market
    ).astype(int)
    
    # Confirmed Buy: Multiple signals align
    data['Confirmed_Buy'] = (
        (data['Buy_Signal_Count'] >= 4) &  # At least 4 signals
        (data['Future_Return_10d'] > 0.02) &
        (data['Future_MaxDrawdown'] > -0.05)
    ).astype(int)
    
    # ============================================
    # STABLE TARGETS - ใช้ได้ทุก regime
    # ============================================
    
    # Stable Buy: ใช้ได้ทั้ง Bull และ Bear market
    # - Fear ต่ำพอสมควร (ไม่ต้องต่ำมาก)
    # - มี positive return ใน 10 วัน
    # - ไม่มี drawdown หนักเกินไป
    data['Stable_Buy'] = (
        (data['FearIndex'] < 30) &  # Fear zone (ไม่ต้องต่ำมาก)
        (data['RSI'] < 45) &  # Not overbought
        (data['Future_Return_10d'] > 0.01) &  # At least 1% gain
        (data['Future_MaxDrawdown'] > -0.06) &  # Max 6% drawdown
        (data['VIX'] < 40)  # Not in crash
    ).astype(int)
    
    # Regime-Aware Buy: ปรับตาม Bull/Bear
    data['Regime_Buy'] = (
        # Bull Market: ซื้อง่ายกว่า
        ((data['Bull_Market'] == 1) & 
         (data['FearIndex'] < 35) & 
         (data['Future_Return_5d'] > 0.005)) |
        # Bear Market: ต้องเข้มงวดกว่า
        ((data['Bear_Market'] == 1) & 
         (data['FearIndex'] < 20) & 
         (data['RSI'] < 35) &
         (data['Future_Return_10d'] > 0.02))
    ).astype(int)
    
    # Conservative Buy: เน้น precision สูงสุด
    data['Conservative_Buy'] = (
        (data['FearIndex'] < 25) &
        (data['RSI'] < 40) &
        (data['VIX'] > 18) &  # มี volatility พอสมควร
        (data['VIX'] < 35) &  # แต่ไม่ crash
        (data['Future_Return_10d'] > 0.015) &
        (data['Future_MaxDrawdown'] > -0.04)
    ).astype(int)
    
    # High Conviction Buy: Very strict, high win rate
    data['High_Conviction_Buy'] = (
        (data['FearIndex'] < 18) &
        (data['RSI'] < 32) &
        (data['VIX'] > 22) &
        (data['VIX'] < 45) &
        (data['Fear_Recovering'] == 1) &
        (data['Future_Return_10d'] > 0.03) &
        (data['Future_MaxDrawdown'] > -0.04)
    ).astype(int)
    
    # Recovery Buy: After fear spike, when recovering
    data['Recovery_Buy'] = (
        (data['FearIndex'] < 25) &
        (data['Fear_Change_3d'] > 3) &
        (data['VIX_Declining'] == 1) &
        (data['Future_Return_5d'] > 0.015) &
        (data['Future_MaxDrawdown'] > -0.03)
    ).astype(int)
    
    # Extreme Oversold Buy: Multiple oversold signals
    data['Extreme_Oversold_Buy'] = (
        (data['FearIndex'] < 20) &
        (data['RSI'] < 30) &
        (data['Below_BB_Low'] == 1) &
        (data['Drawdown_20d'] < -0.05) &
        (data['Future_Return_10d'] > 0.025)
    ).astype(int)
    
    # ============================================
    # 8. GREED/SELL FEATURES (NEW!)
    # ============================================
    # Greed levels
    data['Greed_High'] = (data['FearIndex'] > config.GREED_WATCH_THRESHOLD).astype(int)
    data['Greed_Extreme'] = (data['FearIndex'] > config.GREED_EXTREME_THRESHOLD).astype(int)
    
    # Greed momentum - กำลังขึ้นหรือกำลังลง
    data['Greed_Momentum'] = data['Fear_SMA_5'] - data['Fear_SMA_10']  # บวก = กำลังขึ้น
    data['Greed_Peaking'] = (
        (data['FearIndex'] > 70) & 
        (data['Fear_Change_3d'] < 0)  # Greed สูงแต่เริ่มลง
    ).astype(int)
    
    # Days in Greed zone
    data['Days_In_Greed'] = (data['FearIndex'] > 70).rolling(10).sum()
    
    # Overbought signals
    data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
    data['RSI_Extreme_Overbought'] = (data['RSI'] > 80).astype(int)
    
    # Price extended above moving averages
    data['Price_Extended'] = (data['Price_vs_SMA20'] > 0.05).astype(int)  # >5% above SMA20
    data['Price_Very_Extended'] = (data['Price_vs_SMA50'] > 0.10).astype(int)  # >10% above SMA50
    
    # VIX complacency (very low VIX = danger)
    data['VIX_Complacent'] = (data['VIX'] < 12).astype(int)
    data['VIX_Low'] = (data['VIX'] < 15).astype(int)
    
    # Rally exhaustion signals
    data['Rally_Extended'] = (
        (data['Return_20d'] > 0.10) &  # >10% gain in 20 days
        (data['RSI'] > 65)
    ).astype(int)
    
    # Combined Greed signals
    data['Greed_RSI_Align'] = ((data['FearIndex'] > 75) & (data['RSI'] > 65)).astype(int)
    data['Greed_VIX_Align'] = ((data['FearIndex'] > 75) & (data['VIX'] < 15)).astype(int)
    
    # Top Warning - multiple overbought signals
    data['Top_Warning'] = (
        (data['Greed_High'] == 1) &
        (data['RSI_Overbought'] == 1) &
        (data['Price_Extended'] == 1)
    ).astype(int)
    
    # ============================================
    # 9. SELL TARGETS (IMPROVED!)
    # ============================================
    # Future max gain (to detect if we're at top)
    future_highs = data['High'].rolling(10).max().shift(-10)
    data['Future_MaxGain'] = (future_highs - data['Close']) / data['Close']
    
    # Fear percentile (relative to recent history)
    data['Fear_Percentile'] = data['FearIndex'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Near recent high (price near 20d high)
    data['Near_High'] = (data['Drawdown_20d'] > -0.02).astype(int)  # within 2% of high
    
    # ============================================
    # SELL TARGET 1: Market Top Detection (ลด threshold)
    # ============================================
    # ราคาจะลงใน 5-10 วัน (ลด threshold จาก -2% เป็น -1.5%)
    data['Sell_Signal'] = (
        (data['Future_Return_5d'] < -0.015) &  # ลง >1.5% ใน 5 วัน
        (data['FearIndex'] > 65)  # ลดจาก 75 เป็น 65
    ).astype(int)
    
    # ============================================
    # SELL TARGET 2: Good Exit (ผ่อนคลายลง)
    # ============================================
    data['Good_Exit'] = (
        (data['Future_Return_10d'] < -0.02) &  # ลง >2% ใน 10 วัน (ลดจาก 3%)
        (data['Future_MaxGain'] < 0.03)  # ไม่ขึ้นไปอีก >3%
    ).astype(int)
    
    # ============================================
    # SELL TARGET 3: Momentum Reversal (NEW!)
    # ============================================
    # เมื่อ Greed สูงและเริ่มกลับตัว
    data['Momentum_Sell'] = (
        (data['FearIndex'] > 60) &  # Greed zone
        (data['Fear_Change_3d'] < -5) &  # Greed ลดลงเร็ว
        (data['Future_Return_5d'] < 0)  # ราคาจะลง
    ).astype(int)
    
    # ============================================
    # SELL TARGET 4: Overbought Reversal (NEW!)
    # ============================================
    # RSI overbought + price extended + จะลง
    data['Overbought_Sell'] = (
        (data['RSI'] > 65) &
        (data['Price_vs_SMA20'] > 0.03) &  # >3% above SMA20
        (data['Future_Return_5d'] < -0.01)  # ราคาจะลง >1%
    ).astype(int)
    
    # ============================================
    # SELL TARGET 5: Rally Exhaustion (NEW!)
    # ============================================
    # หลังขึ้นมาเยอะแล้ว + จะลง
    data['Rally_Sell'] = (
        (data['Return_20d'] > 0.08) &  # ขึ้นมา >8% ใน 20 วัน
        (data['Future_Return_10d'] < -0.02)  # จะลง >2%
    ).astype(int)
    
    # ============================================
    # SELL TARGET 6: Best Exit (Combined signals)
    # ============================================
    data['Best_Exit'] = (
        (data['FearIndex'] > 60) &
        (data['RSI'] > 60) &
        (data['Near_High'] == 1) &
        (data['Future_Return_10d'] < -0.02) &
        (data['Future_MaxGain'] < 0.02)
    ).astype(int)
    
    # ============================================
    # PREMIUM SELL TARGETS (NEW!)
    # ============================================
    # Premium Sell: Very strict for high precision
    data['Premium_Sell'] = (
        (data['FearIndex'] > 75) &
        (data['RSI'] > 70) &
        (data['VIX'] < 14) &
        (data['Future_Return_10d'] < -0.03) &
        (data['Future_MaxGain'] < 0.02)
    ).astype(int)
    
    # Confirmed Sell: Multiple signals
    data['Confirmed_Sell'] = (
        (data['Sell_Signal_Count'] >= 4) &
        (data['Future_Return_10d'] < -0.02)
    ).astype(int)
    
    # ============================================
    # SELL TARGET 7: Safe Exit (ผ่อนคลายลง)
    # ============================================
    data['Safe_Exit'] = (
        (data['FearIndex'] > 70) &  # ลดจาก 80
        (data['Future_Return_5d'] < 0) &
        (data['RSI'] > 60)  # ลดจาก 70
    ).astype(int)
    
    # ============================================
    # SELL TARGET 8: Any Decline from High (NEW!)
    # ============================================
    # ใกล้ high + จะลง (general sell signal)
    data['Decline_Signal'] = (
        (data['Near_High'] == 1) &
        (data['Future_Return_5d'] < -0.015)
    ).astype(int)
    
    # ============================================
    # 10. ECONOMIC CALENDAR FEATURES (NEW!)
    # ============================================
    try:
        data = add_economic_calendar_features(data)
    except Exception as e:
        print(f"Warning: Could not add economic calendar features: {e}")
        # Add default values
        data['Days_To_FOMC'] = 999
        data['FOMC_Week'] = 0
        data['FOMC_Tomorrow'] = 0
        data['Days_To_CPI'] = 999
        data['CPI_Week'] = 0
        data['CPI_Tomorrow'] = 0
        data['High_Event_Risk'] = 0
        data['Avoid_Signal'] = 0
    
    return data.dropna()


def get_feature_columns_v2(signal_type='buy', pruned=True):
    """Features V2 - Pruned version (ตัด noise ออก)
    
    Args:
        signal_type: 'buy' for buy signals, 'sell' for sell signals, 'all' for both
        pruned: True = ใช้เฉพาะ Top features, False = ใช้ทั้งหมด
    """
    
    if pruned:
        # ============================================
        # PRUNED FEATURES (Top 25 - ตัด noise ออก)
        # Based on Feature Importance Analysis
        # ============================================
        base_features = [
            # Core Fear/Greed (พระเอก!)
            'FearIndex', 'Fear_Change_5d', 'Fear_Momentum', 'Fear_Recovering',
            
            # VIX (สำคัญมาก)
            'VIX', 'VIX_Change_5d', 'VIX_vs_SMA', 'VIX_Declining',
            
            # Price Action (จำเป็น)
            'Return_5d', 'Return_10d', 'Drawdown_20d',
            'Price_vs_SMA20', 'Price_vs_SMA50', 'Price_vs_SMA200',
            
            # Technical (เลือกเฉพาะตัวสำคัญ)
            'RSI', 'RSI_Oversold', 'BB_Pct',
            
            # Market Regime (สำคัญมาก!)
            'Bull_Market', 'Bear_Market',
            
            # Combined Signals
            'Recovery_Signal', 'Crash_Warning', 'Buy_Signal_Count',
            
            # Economic Calendar
            'High_Event_Risk', 'Avoid_Signal'
        ]
        
        greed_features = [
            'Greed_High', 'Greed_Extreme', 'Greed_Momentum',
            'RSI_Overbought', 'Price_Extended', 'Top_Warning',
            'Sell_Signal_Count'
        ]
    else:
        # ============================================
        # FULL FEATURES (ใช้ทั้งหมด - อาจ overfit)
        # ============================================
        base_features = [
            # Fear/Greed features
            'FearIndex', 'Fear_Low', 'Fear_Extreme', 
            'Fear_Change', 'Fear_Change_3d', 'Fear_Change_5d',
            'Fear_Momentum', 'Fear_Recovering', 'Days_In_Fear',
            'Fear_Lag_1', 'Fear_Lag_2', 'Fear_Lag_3', 'Fear_Lag_5',
            
            # VIX features
            'VIX', 'VIX_High', 'VIX_Extreme', 'VIX_Crash',
            'VIX_Change', 'VIX_Change_5d', 'VIX_vs_SMA', 'VIX_Declining',
            
            # Volatility
            'RealizedVol_10d', 'RealizedVol_20d', 'Vol_Ratio',
            
            # Price action
            'Return', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d',
            'Drawdown_20d', 'Drawdown_50d',
            'Price_vs_SMA20', 'Price_vs_SMA50', 'Price_vs_SMA200',
            'Uptrend', 'SMA20_Above_SMA50',
            
            # Technical
            'RSI', 'RSI_Oversold', 'RSI_Extreme_Oversold', 'RSI_Recovering',
            'MACD', 'MACD_Diff', 'MACD_Negative', 'MACD_Crossover',
            'BB_Pct', 'Below_BB_Low', 'BB_Width',
            
            # Volume
            'Volume_Ratio', 'Volume_Spike',
            
            # Combined signals
            'Fear_RSI_Align', 'Fear_VIX_Align', 'Recovery_Signal', 'Crash_Warning',
            
            # Market Regime
            'Bull_Market', 'Bear_Market', 'High_Vol_Regime', 'Low_Vol_Regime',
            'Fear_Regime_Low', 'Fear_Regime_High',
            
            # Confirmation Signals
            'Strong_Buy_Setup', 'Moderate_Buy_Setup', 'Buy_Signal_Count',
            
            # Mean Reversion
            'Price_Zscore', 'Oversold_Zscore', 'Fear_Zscore', 'Extreme_Fear_Zscore',
            
            # Economic Calendar
            'Days_To_FOMC', 'FOMC_Week', 'FOMC_Tomorrow',
            'Days_To_CPI', 'CPI_Week', 'CPI_Tomorrow',
            'High_Event_Risk', 'Avoid_Signal'
        ]
        
        greed_features = [
            'Greed_High', 'Greed_Extreme', 'Greed_Momentum', 'Greed_Peaking',
            'Days_In_Greed', 'RSI_Overbought', 'RSI_Extreme_Overbought',
            'Price_Extended', 'Price_Very_Extended',
            'VIX_Complacent', 'VIX_Low', 'Rally_Extended',
            'Greed_RSI_Align', 'Greed_VIX_Align', 'Top_Warning',
            'Strong_Sell_Setup', 'Sell_Signal_Count', 'Overbought_Zscore', 'Extreme_Greed_Zscore'
        ]
    
    if signal_type == 'buy':
        return base_features
    elif signal_type == 'sell':
        return base_features + greed_features
    else:  # 'all'
        return base_features + greed_features
