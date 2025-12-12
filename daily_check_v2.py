"""
Daily Check V2 - Improved daily signal check with Fear Prediction
รวม Synthetic Fear & Greed, Economic Calendar และ Circuit Breaker
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import (
    get_fear_greed_historical, get_market_data, get_current_fear_greed,
    add_economic_calendar_features, get_days_to_event, get_fomc_dates, get_cpi_dates
)
from feature_engineering_v2 import create_features_v2, get_feature_columns_v2
from models_v2 import ModelComparisonV2
from fear_predictor import FearPredictor
from discord_alert import send_discord_alert, send_fear_forecast, send_discord_alert_with_chart
import config


def is_safe_to_trade(close_price, sma_200, vix, rsi, macd_diff, days_to_fomc):
    """
    Circuit Breaker - ตัดวงจรถ้าความเสี่ยงสูงเกินไป
    
    Returns:
        tuple: (is_safe: bool, reason: str)
    """
    # 1. ห้ามสวนเทรนด์ใหญ่: ราคาใต้ SMA200 + VIX สูง = ตลาด Panic
    if close_price < sma_200 and vix > 30:
        return False, "🚫 Circuit Breaker: Bear Market + High VIX (ห้ามสวนเทรนด์)"
    
    # 2. ห้ามรับมีด: RSI ต่ำมาก + Momentum ยังดิ่ง
    if rsi < 25 and macd_diff < -0.5:
        return False, "🚫 Circuit Breaker: Falling Knife (RSI ต่ำ + MACD ดิ่ง)"
    
    # 3. ห้ามเทรดช่วง Crash: VIX > 40 = ตลาดนรก
    if vix > 40:
        return False, "🚫 Circuit Breaker: Market Crash (VIX > 40)"
    
    # 4. ห้ามเทรดก่อน FOMC: ถ้า Days_To_FOMC <= 2
    if days_to_fomc <= 2:
        return False, "🚫 Circuit Breaker: FOMC Soon (งดเทรดก่อนลุงพาวเวลพูด)"
    
    return True, "✅ Safe to trade"


def check_today_v2():
    """Check today's signal with V2 model (Buy + Sell)"""
    print("=" * 70)
    print(f"DAILY CHECK V2 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. Load BUY Model
    buy_model = ModelComparisonV2()
    if not buy_model.load_model('_v2'):
        print("V2 Buy Model not found! Run train_v2.py first.")
        return None
    print("✅ Buy Model loaded successfully")
    
    # 1b. Load SELL Model
    sell_model = ModelComparisonV2()
    sell_model_loaded = sell_model.load_model('_sell_v2')
    if sell_model_loaded:
        print("✅ Sell Model loaded successfully")
    else:
        print("⚠️ Sell Model not found (will skip sell signals)")
    
    # 2. Get Current Fear & Greed (Synthetic - 7 factors!)
    current_fg = get_current_fear_greed()
    if current_fg and current_fg['score']:
        current_fear = current_fg['score']
        print(f"\nCurrent Fear & Greed: {current_fear:.1f} ({current_fg['rating']})")
        print(f"Source: {current_fg.get('source', 'unknown')}")
        if current_fg.get('source') == 'synthetic_7_factors':
            print("✅ Using Synthetic Fear & Greed (7 factors like CNN)")
    else:
        current_fear = None
    
    # 2b. Check Economic Calendar
    today = datetime.now()
    days_to_fomc = get_days_to_event(today, get_fomc_dates())
    days_to_cpi = get_days_to_event(today, get_cpi_dates())
    
    print(f"\n📅 Economic Calendar:")
    print(f"   Days to FOMC: {days_to_fomc}")
    print(f"   Days to CPI:  {days_to_cpi}")
    
    # Warning if event is near
    event_warning = False
    if days_to_fomc <= 1:
        print("   ⚠️ FOMC Meeting Tomorrow/Today - High Volatility Expected!")
        event_warning = True
    if days_to_cpi <= 1:
        print("   ⚠️ CPI Release Tomorrow/Today - High Volatility Expected!")
        event_warning = True
    
    # 3. Load Data
    print("\nLoading market data...")
    fear_data = get_fear_greed_historical()
    market_data = get_market_data()
    
    if fear_data is None or market_data is None:
        print("Error loading data!")
        return None
    
    # 4. Create V2 Features
    data = create_features_v2(market_data, fear_data)
    
    # Get latest
    latest = data.iloc[-1:]
    feature_cols = get_feature_columns_v2('all', pruned=True)  # Use pruned features
    feature_cols = [f for f in feature_cols if f in data.columns]
    X_latest = latest[feature_cols]
    
    # 5. Predict BUY signal with confidence
    buy_prediction, buy_probability = buy_model.predict_with_confidence(X_latest, min_confidence=0.5)
    buy_pred = buy_prediction[0]
    buy_prob = buy_probability[0]
    
    # 5b. Predict SELL signal
    sell_pred = 0
    sell_prob = 0.0
    if sell_model_loaded:
        sell_prediction, sell_probability = sell_model.predict_with_confidence(X_latest, min_confidence=0.5)
        sell_pred = sell_prediction[0]
        sell_prob = sell_probability[0]
    
    # Use current fear if available
    fear_index = current_fear if current_fear else latest['FearIndex'].values[0]
    
    # Get metrics
    spx_price = latest['Close'].values[0]
    vix = latest['VIX'].values[0]
    rsi = latest['RSI'].values[0]
    spx_return = latest['Return'].values[0]
    drawdown = latest['Drawdown_20d'].values[0] * 100
    crash_warning = latest['Crash_Warning'].values[0]
    recovery_signal = latest['Recovery_Signal'].values[0]
    
    # Get greed metrics
    greed_high = latest['Greed_High'].values[0] if 'Greed_High' in latest.columns else 0
    greed_extreme = latest['Greed_Extreme'].values[0] if 'Greed_Extreme' in latest.columns else 0
    top_warning = latest['Top_Warning'].values[0] if 'Top_Warning' in latest.columns else 0
    price_extended = latest['Price_Extended'].values[0] if 'Price_Extended' in latest.columns else 0
    
    # ============================================
    # MARKET REGIME DETECTION (NEW!)
    # ============================================
    sma_200 = latest['SMA_200'].values[0] if 'SMA_200' in latest.columns else spx_price
    is_bull_market = spx_price > sma_200
    bull_bear_pct = ((spx_price - sma_200) / sma_200) * 100
    
    # Regime-adjusted thresholds
    if is_bull_market:
        # Bull Market: ซื้อง่ายกว่า (Buy the Dip)
        buy_prob_threshold = 0.5
        buy_fear_threshold = 25
        regime_name = "🐂 BULL MARKET"
    else:
        # Bear Market: ต้องระวัง (Don't catch falling knife)
        buy_prob_threshold = 0.7  # ต้องมั่นใจมากกว่า
        buy_fear_threshold = 20   # Fear ต้องต่ำกว่า
        regime_name = "🐻 BEAR MARKET"
    
    print(f"\n{'=' * 70}")
    print("TODAY'S ANALYSIS (V2)")
    print(f"{'=' * 70}")
    print(f"Fear & Greed Index:  {fear_index:.1f}")
    print(f"S&P 500 Price:       ${spx_price:,.2f}")
    print(f"S&P 500 Return:      {spx_return*100:.2f}%")
    print(f"Drawdown from High:  {drawdown:.1f}%")
    print(f"VIX:                 {vix:.2f}")
    print(f"RSI:                 {rsi:.2f}")
    
    # Show Market Regime
    print(f"\n--- Market Regime ---")
    print(f"SMA 200:             ${sma_200:,.2f}")
    print(f"Price vs SMA200:     {bull_bear_pct:+.1f}%")
    print(f"Regime:              {regime_name}")
    if not is_bull_market:
        print(f"⚠️ Bear Market Mode: Higher confidence required for BUY signals")
    
    # ============================================
    # CIRCUIT BREAKER CHECK
    # ============================================
    macd_diff = latest['MACD_Diff'].values[0] if 'MACD_Diff' in latest.columns else 0
    circuit_safe, circuit_reason = is_safe_to_trade(
        spx_price, sma_200, vix, rsi, macd_diff, days_to_fomc
    )
    
    print(f"\n--- Circuit Breaker ---")
    print(f"Status:              {circuit_reason}")
    
    print(f"\n--- Fear Zone (BUY) ---")
    print(f"Crash Warning:       {'⚠️ YES' if crash_warning else '✅ NO'}")
    print(f"Recovery Signal:     {'✅ YES' if recovery_signal else '❌ NO'}")
    print(f"BUY Prediction:      {'🟢 BUY SIGNAL' if buy_pred == 1 else '⚪ NO SIGNAL'}")
    print(f"BUY Confidence:      {buy_prob*100:.1f}%")
    
    print(f"\n--- Greed Zone (SELL) ---")
    print(f"Greed High (>75):    {'🔴 YES' if greed_high else '✅ NO'}")
    print(f"Greed Extreme (>85): {'🔴 YES' if greed_extreme else '✅ NO'}")
    print(f"Top Warning:         {'🔴 YES' if top_warning else '✅ NO'}")
    print(f"Price Extended:      {'🔴 YES' if price_extended else '✅ NO'}")
    if sell_model_loaded:
        print(f"SELL Prediction:     {'🔴 SELL SIGNAL' if sell_pred == 1 else '⚪ NO SIGNAL'}")
        print(f"SELL Confidence:     {sell_prob*100:.1f}%")
    
    # 6. Decision Logic
    alert_sent = False
    signal_type = None
    
    # ============================================
    # CHECK SELL SIGNALS FIRST (Greed Zone)
    # ============================================
    if fear_index > config.GREED_EXTREME_THRESHOLD:
        # EXTREME GREED
        print(f"\n🔴 EXTREME GREED DETECTED! (> {config.GREED_EXTREME_THRESHOLD})")
        
        if sell_pred == 1 and sell_prob > 0.6:
            print("🔴 Model CONFIRMS SELL signal with high confidence!")
            signal_type = 'EXTREME_SELL'
        else:
            print("⚠️ Extreme greed but model not confident - watch closely")
            signal_type = 'SELL_WATCH'
            
    elif fear_index > config.GREED_WATCH_THRESHOLD:
        # GREED WATCH ZONE
        print(f"\n🟠 GREED WATCH ZONE (Fear > {config.GREED_WATCH_THRESHOLD})")
        
        if sell_pred == 1 and sell_prob > 0.7:
            print("🔴 Model confirms SELL SIGNAL!")
            signal_type = 'SELL_SIGNAL'
        elif top_warning:
            print("⚠️ Top Warning active - consider taking profits")
            signal_type = 'SELL_WATCH'
        else:
            print("👀 Watching for sell setup...")
            signal_type = 'SELL_WATCH'
    
    # ============================================
    # CHECK BUY SIGNALS (Fear Zone) + CIRCUIT BREAKER
    # ============================================
    elif not circuit_safe:
        # Circuit Breaker ทำงาน - ห้ามซื้อ!
        print(f"\n{circuit_reason}")
        print("🚫 BUY signals BLOCKED by Circuit Breaker")
        signal_type = 'INFO'
    
    elif crash_warning:
        print(f"\n⚠️ CRASH WARNING ACTIVE - VIX very high or big drawdown")
        print("Model recommends WAITING even if Fear is low")
        signal_type = 'INFO'
    
    elif fear_index < config.FEAR_EXTREME_THRESHOLD:
        # EXTREME FEAR
        print(f"\n🚨 EXTREME FEAR DETECTED! (< {config.FEAR_EXTREME_THRESHOLD})")
        
        # Regime-adjusted decision
        if is_bull_market:
            # Bull: ซื้อได้เลยถ้า model บอก
            if buy_pred == 1 and buy_prob > 0.5:
                print("✅ Model CONFIRMS buy signal! (Bull Market - easier entry)")
                signal_type = 'EXTREME'
            else:
                print("⚠️ Model suggests waiting for better setup")
                signal_type = 'WATCH'
        else:
            # Bear: ต้องมั่นใจมากกว่า + Fear ต้องต่ำจริงๆ
            if buy_pred == 1 and buy_prob > 0.7 and fear_index < 15:
                print("✅ Model CONFIRMS buy signal with HIGH confidence! (Bear Market - strict mode)")
                signal_type = 'EXTREME'
            elif buy_pred == 1 and buy_prob > 0.6:
                print("🟡 Moderate signal in Bear Market - consider smaller position")
                signal_type = 'WATCH'
            else:
                print("⚠️ Bear Market: Waiting for stronger confirmation")
                signal_type = 'WATCH'
            
    elif fear_index < buy_fear_threshold:  # Use regime-adjusted threshold
        # WATCH ZONE
        print(f"\n👀 WATCH ZONE (Fear < {buy_fear_threshold})")
        
        # Regime-adjusted decision
        if buy_pred == 1 and buy_prob > buy_prob_threshold:
            if is_bull_market:
                print("✅ Model confirms BUY SIGNAL! (Bull Market)")
                signal_type = 'BUY_SIGNAL'
            else:
                print("🟡 Model suggests BUY but Bear Market - be cautious")
                signal_type = 'WATCH'
        elif buy_pred == 1 and buy_prob > 0.5:
            print("🟡 Moderate confidence - consider smaller position")
            signal_type = 'WATCH'
        else:
            print("⏳ Waiting for better setup...")
            signal_type = 'WATCH'
    else:
        print(f"\n📊 Normal market conditions (Fear: {fear_index:.1f})")
        print("No alert needed.")
    
    # 7. Send Alert (with chart!)
    if signal_type and signal_type != 'INFO':
        # Skip signal if major event tomorrow (optional)
        if event_warning and signal_type in ['WATCH', 'SELL_WATCH']:
            print("\n⚠️ Skipping weak signal due to upcoming economic event")
        else:
            # Determine if it's a sell or buy signal
            is_sell_signal = signal_type in ['EXTREME_SELL', 'SELL_SIGNAL', 'SELL_WATCH']
            
            # Try to send with chart
            try:
                send_discord_alert_with_chart(
                    fear_index=fear_index,
                    probability=sell_prob if is_sell_signal else buy_prob,
                    signal_type=signal_type,
                    data=data,  # Pass data for chart generation
                    spx_price=spx_price,
                    vix=vix,
                    rsi=rsi,
                    additional_info={
                        "Drawdown": f"{drawdown:.1f}%",
                        "Crash Warning": "YES" if crash_warning else "NO",
                        "Recovery Signal": "YES" if recovery_signal else "NO",
                        "Top Warning": "YES" if top_warning else "NO",
                        "Days to FOMC": str(days_to_fomc),
                        "Days to CPI": str(days_to_cpi),
                        "Data Source": "Synthetic 7-Factor",
                        "Model Version": "V2 (Buy + Sell)"
                    }
                )
            except Exception as e:
                # Fallback to text-only alert
                print(f"Chart generation failed: {e}, sending text-only alert")
                send_discord_alert(
                    fear_index=fear_index,
                    probability=sell_prob if is_sell_signal else buy_prob,
                    signal_type=signal_type,
                    spx_price=spx_price,
                    vix=vix,
                    rsi=rsi,
                    additional_info={
                        "Drawdown": f"{drawdown:.1f}%",
                        "Crash Warning": "YES" if crash_warning else "NO",
                        "Recovery Signal": "YES" if recovery_signal else "NO",
                        "Top Warning": "YES" if top_warning else "NO",
                        "Days to FOMC": str(days_to_fomc),
                        "Days to CPI": str(days_to_cpi),
                        "Model Version": "V2 (Buy + Sell)"
                    }
                )
            alert_sent = True
    
    # 8. Fear Forecast
    print(f"\n{'=' * 70}")
    print("FEAR FORECAST (Next 5 Days)")
    print(f"{'=' * 70}")
    
    fear_pred = FearPredictor()
    if fear_pred.load():
        forecast = fear_pred.predict(data)
        forecast_msg = fear_pred.get_forecast_message(forecast)
        print(forecast_msg)
        
        # Send forecast alert if fear might drop to interesting levels
        if forecast['predicted_min_5d'] < 20 or forecast['prob_significant_drop'] > 0.6:
            send_fear_forecast(
                current_fear=forecast['current_fear'],
                predicted_min=forecast['predicted_min_5d'],
                prob_drop=forecast['prob_significant_drop'],
                momentum=forecast['fear_momentum']
            )
    else:
        print("Fear Predictor not trained. Run train_v2.py first.")
    
    print(f"\n{'=' * 70}")
    print(f"Check complete! Alert sent: {alert_sent}")
    print(f"{'=' * 70}")
    
    return {
        'date': datetime.now(),
        'fear_index': fear_index,
        'buy_prediction': buy_pred,
        'buy_probability': buy_prob,
        'sell_prediction': sell_pred,
        'sell_probability': sell_prob,
        'spx_price': spx_price,
        'vix': vix,
        'rsi': rsi,
        'crash_warning': crash_warning,
        'recovery_signal': recovery_signal,
        'top_warning': top_warning,
        'greed_high': greed_high,
        'alert_sent': alert_sent,
        'signal_type': signal_type,
        'fear_forecast': forecast if 'forecast' in dir() else None
    }


if __name__ == "__main__":
    check_today_v2()
