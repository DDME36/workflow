"""
Discord Alert V2 - ส่งแจ้งเตือนผ่าน Discord Webhook
รวมการส่งรูปกราฟ
"""
import requests
from datetime import datetime
import io
import config

# Try to import matplotlib (optional for chart generation)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def send_discord_alert(
    fear_index: float,
    probability: float,
    signal_type: str,
    spx_price: float = None,
    vix: float = None,
    rsi: float = None,
    additional_info: dict = None
):
    """ส่ง Alert ไป Discord"""
    
    if not config.DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL not configured!")
        return False
    
    # Color & Title based on signal type
    # BUY signals (Fear zone)
    if signal_type == 'EXTREME':
        color = 0xFF0000
        title = "🚨 EXTREME FEAR ALERT"
        desc = "ตลาดกลัวสุดขีด! นี่คือโอกาสที่หายากมาก"
    elif signal_type == 'BUY_SIGNAL':
        color = 0x00FF00
        title = "💰 BUY SIGNAL"
        desc = "โมเดลแนะนำให้พิจารณาเข้าซื้อ"
    elif signal_type == 'WATCH':
        color = 0xFFA500
        title = "👀 WATCH ZONE"
        desc = "Fear ต่ำกว่า 25 จับตามองใกล้ชิด"
    # SELL signals (Greed zone) - NEW!
    elif signal_type == 'EXTREME_SELL':
        color = 0x8B0000  # Dark red
        title = "🔴 EXTREME GREED - SELL ALERT"
        desc = "ตลาดโลภสุดขีด! พิจารณาขายทำกำไร"
    elif signal_type == 'SELL_SIGNAL':
        color = 0xFF4500  # Orange red
        title = "📉 SELL SIGNAL"
        desc = "โมเดลแนะนำให้พิจารณาขาย/ทำกำไร"
    elif signal_type == 'SELL_WATCH':
        color = 0xFFD700  # Gold
        title = "⚠️ GREED WATCH ZONE"
        desc = "Greed สูงกว่า 75 - จับตามองใกล้ชิด"
    else:
        color = 0x0099FF
        title = "📊 Market Update"
        desc = "อัพเดทสถานะตลาด"
    
    # Fear level indicator
    if fear_index < 10:
        fear_bar = "🔴🔴🔴🔴🔴 EXTREME"
    elif fear_index < 20:
        fear_bar = "🟠🟠🟠🟠⚪ HIGH FEAR"
    elif fear_index < 30:
        fear_bar = "🟡🟡🟡⚪⚪ FEAR"
    elif fear_index < 50:
        fear_bar = "⚪⚪⚪⚪⚪ NEUTRAL"
    elif fear_index < 70:
        fear_bar = "🟢🟢🟢⚪⚪ GREED"
    else:
        fear_bar = "🟢🟢🟢🟢🟢 EXTREME GREED"
    
    # Build embed
    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "fields": [
            {
                "name": "━━━━━ FEAR & GREED ━━━━━",
                "value": f"```\n{fear_bar}\n      {fear_index:.0f} / 100\n```",
                "inline": False
            },
            {
                "name": "📈 Model Confidence",
                "value": f"```{probability*100:.0f}%```",
                "inline": True
            },
            {
                "name": "💹 S&P 500",
                "value": f"```${spx_price:,.0f}```" if spx_price else "```N/A```",
                "inline": True
            },
            {
                "name": "😱 VIX",
                "value": f"```{vix:.1f}```" if vix else "```N/A```",
                "inline": True
            },
        ],
        "footer": {
            "text": f"Fear & Greed Alert | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    # Add RSI if available
    if rsi:
        rsi_status = "Oversold 📉" if rsi < 30 else "Overbought 📈" if rsi > 70 else "Normal"
        embed["fields"].append({
            "name": "📊 RSI",
            "value": f"```{rsi:.0f} ({rsi_status})```",
            "inline": True
        })
    
    # Add additional info
    if additional_info:
        info_text = "\n".join([f"• {k}: {v}" for k, v in additional_info.items()])
        embed["fields"].append({
            "name": "📋 Additional Info",
            "value": f"```{info_text}```",
            "inline": False
        })
    
    # Action recommendation
    # BUY actions
    if signal_type == 'EXTREME':
        action = "🔥 พิจารณาเข้าซื้อ! โอกาสแบบนี้หายาก"
    elif signal_type == 'BUY_SIGNAL':
        action = "✅ สัญญาณซื้อ - รอจังหวะที่ Fear เริ่มฟื้น"
    elif signal_type == 'WATCH':
        action = "👁️ จับตามอง - รอสัญญาณยืนยันจากโมเดล"
    # SELL actions - NEW!
    elif signal_type == 'EXTREME_SELL':
        action = "🔴 พิจารณาขาย/ทำกำไร! ตลาดอาจถึงจุดสูงสุด"
    elif signal_type == 'SELL_SIGNAL':
        action = "📉 สัญญาณขาย - พิจารณาลดพอร์ต/ทำกำไร"
    elif signal_type == 'SELL_WATCH':
        action = "⚠️ จับตามอง - ตลาดอาจใกล้ถึงจุดสูงสุด"
    else:
        action = "📊 ไม่มีสัญญาณ - รอดูต่อ"
    
    embed["fields"].append({
        "name": "💡 Action",
        "value": action,
        "inline": False
    })
    
    # Add Exit Rules for BUY signals
    if signal_type in ['EXTREME', 'BUY_SIGNAL']:
        exit_rules = """```
📤 EXIT RULES (หลังซื้อแล้ว):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🛑 Stop Loss: ถ้าขาดทุน > 4% → ขายทันที
⏰ Stagnation: ถ้า 3 วันยังไม่กำไร > 1% → ขายทิ้ง
🎯 Take Profit: ถ้า RSI > 70 หรือ Fear > 70 → ขาย
📈 Trailing: ถ้ากำไร > 3% → เลื่อน Stop มาที่ทุน
         ถ้ากำไร > 5% → Trail 2% จากราคาสูงสุด
⏳ Max Hold: ถือไม่เกิน 10 วัน
```"""
        embed["fields"].append({
            "name": "📤 Exit Strategy",
            "value": exit_rules,
            "inline": False
        })
    
    try:
        response = requests.post(
            config.DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 204:
            print(f"Discord alert sent! ({signal_type})")
            return True
        else:
            print(f"Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def send_fear_forecast(
    current_fear: float,
    predicted_min: float,
    prob_drop: float,
    momentum: float
):
    """ส่ง Fear Forecast Alert"""
    
    if not config.DISCORD_WEBHOOK_URL:
        return False
    
    # Determine style
    if predicted_min < 10:
        color = 0xFF0000
        title = "🚨 EXTREME FEAR INCOMING!"
        outlook = "คาดว่าจะลงไปถึง Extreme Fear"
    elif predicted_min < 15:
        color = 0xFFA500
        title = "⚠️ High Fear Expected"
        outlook = "คาดว่าจะลงไปถึง High Fear Zone"
    elif predicted_min < 20:
        color = 0xFFFF00
        title = "👀 Fear Zone Approaching"
        outlook = "คาดว่าจะเข้าสู่ Fear Zone"
    else:
        color = 0x0099FF
        title = "📊 Fear Forecast"
        outlook = "คาดว่าจะทรงตัว"
    
    # Momentum indicator
    if momentum < -5:
        mom_bar = "📈📈📈 ลดลงเร็วมาก"
    elif momentum < -2:
        mom_bar = "📈📈 ลดลง"
    elif momentum > 5:
        mom_bar = "📉📉📉 เพิ่มขึ้นเร็ว"
    elif momentum > 2:
        mom_bar = "📉📉 เพิ่มขึ้น"
    else:
        mom_bar = "➡️ ทรงตัว"
    
    embed = {
        "title": title,
        "description": outlook,
        "color": color,
        "fields": [
            {
                "name": "━━━━━ FORECAST ━━━━━",
                "value": f"```\nปัจจุบัน:  {current_fear:.0f}\nคาดการณ์: {predicted_min:.0f} (ใน 5 วัน)\n```",
                "inline": False
            },
            {
                "name": "📊 โอกาสลงแรง",
                "value": f"```{prob_drop*100:.0f}%```",
                "inline": True
            },
            {
                "name": "📈 Momentum",
                "value": f"```{mom_bar}```",
                "inline": True
            },
        ],
        "footer": {
            "text": f"Fear Forecast | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    # Recommendation
    if predicted_min < 15 and prob_drop > 0.5:
        rec = "💡 เตรียมตัว! อาจมีโอกาสซื้อเร็วๆนี้"
    elif predicted_min < 20:
        rec = "💡 จับตามอง รอจังหวะที่ Fear เริ่มฟื้น"
    else:
        rec = "💡 ยังไม่มีสัญญาณ รอดูต่อ"
    
    embed["fields"].append({
        "name": "💡 Recommendation",
        "value": rec,
        "inline": False
    })
    
    try:
        response = requests.post(
            config.DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 204
    except:
        return False


# ============================================
# CHART GENERATION (NEW!)
# ============================================

def generate_signal_chart(data, signal_type='buy', days=60, save_path=None):
    """
    สร้างกราฟแสดง Signal ล่าสุด
    
    Args:
        data: DataFrame with Close, FearIndex, RSI, VIX
        signal_type: 'buy' or 'sell'
        days: จำนวนวันที่จะแสดง
        save_path: path to save image (optional)
    
    Returns:
        BytesIO object with image data
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for chart generation")
        return None
    
    # Get recent data
    recent = data.tail(days).copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Fear & Greed Signal Chart - {datetime.now().strftime("%Y-%m-%d")}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: S&P 500 Price
    ax1 = axes[0]
    ax1.plot(recent.index, recent['Close'], 'b-', linewidth=1.5, label='S&P 500')
    if 'SMA_20' in recent.columns:
        ax1.plot(recent.index, recent['SMA_20'], 'orange', linewidth=1, alpha=0.7, label='SMA 20')
    if 'SMA_50' in recent.columns:
        ax1.plot(recent.index, recent['SMA_50'], 'green', linewidth=1, alpha=0.7, label='SMA 50')
    
    # Mark signals
    if 'Prediction' in recent.columns:
        buy_signals = recent[recent['Prediction'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    
    if 'Sell_Prediction' in recent.columns:
        sell_signals = recent[recent['Sell_Prediction'] == 1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('S&P 500')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fear & Greed Index
    ax2 = axes[1]
    ax2.fill_between(recent.index, recent['FearIndex'], 50, 
                     where=(recent['FearIndex'] < 50), 
                     color='red', alpha=0.3, label='Fear Zone')
    ax2.fill_between(recent.index, recent['FearIndex'], 50, 
                     where=(recent['FearIndex'] >= 50), 
                     color='green', alpha=0.3, label='Greed Zone')
    ax2.plot(recent.index, recent['FearIndex'], 'k-', linewidth=1.5)
    
    # Add threshold lines
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Fear Threshold')
    ax2.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Greed Threshold')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('Fear & Greed')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI and VIX
    ax3 = axes[2]
    ax3.plot(recent.index, recent['RSI'], 'purple', linewidth=1.5, label='RSI')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI', color='purple')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # VIX on secondary axis
    ax3b = ax3.twinx()
    ax3b.plot(recent.index, recent['VIX'], 'orange', linewidth=1.5, label='VIX')
    ax3b.set_ylabel('VIX', color='orange')
    ax3b.tick_params(axis='y', labelcolor='orange')
    
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    plt.close()
    
    return buf


def send_discord_alert_with_chart(
    fear_index: float,
    probability: float,
    signal_type: str,
    data=None,
    spx_price: float = None,
    vix: float = None,
    rsi: float = None,
    additional_info: dict = None
):
    """
    ส่ง Alert พร้อมรูปกราฟไป Discord
    """
    if not config.DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL not configured!")
        return False
    
    # Generate chart if data is provided
    chart_buf = None
    if data is not None and MATPLOTLIB_AVAILABLE:
        chart_buf = generate_signal_chart(data, signal_type)
    
    # Build embed (same as before)
    if signal_type == 'EXTREME':
        color = 0xFF0000
        title = "🚨 EXTREME FEAR ALERT"
        desc = "ตลาดกลัวสุดขีด! นี่คือโอกาสที่หายากมาก"
    elif signal_type == 'BUY_SIGNAL':
        color = 0x00FF00
        title = "💰 BUY SIGNAL"
        desc = "โมเดลแนะนำให้พิจารณาเข้าซื้อ"
    elif signal_type == 'EXTREME_SELL':
        color = 0x8B0000
        title = "🔴 EXTREME GREED - SELL ALERT"
        desc = "ตลาดโลภสุดขีด! พิจารณาขายทำกำไร"
    elif signal_type == 'SELL_SIGNAL':
        color = 0xFF4500
        title = "📉 SELL SIGNAL"
        desc = "โมเดลแนะนำให้พิจารณาขาย/ทำกำไร"
    else:
        color = 0x0099FF
        title = "📊 Market Update"
        desc = "อัพเดทสถานะตลาด"
    
    # Fear bar
    if fear_index < 25:
        fear_bar = "🔴🔴🔴🔴🔴 FEAR"
    elif fear_index < 50:
        fear_bar = "🟡🟡🟡⚪⚪ NEUTRAL"
    elif fear_index < 75:
        fear_bar = "🟢🟢🟢⚪⚪ GREED"
    else:
        fear_bar = "🟢🟢🟢🟢🟢 EXTREME GREED"
    
    embed = {
        "title": title,
        "description": desc,
        "color": color,
        "fields": [
            {
                "name": "Fear & Greed",
                "value": f"```{fear_bar}\n{fear_index:.0f}/100```",
                "inline": True
            },
            {
                "name": "Confidence",
                "value": f"```{probability*100:.0f}%```",
                "inline": True
            },
            {
                "name": "S&P 500",
                "value": f"```${spx_price:,.0f}```" if spx_price else "```N/A```",
                "inline": True
            },
        ],
        "footer": {
            "text": f"Fear & Greed Alert | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    # Add Exit Rules for BUY signals
    if signal_type in ['EXTREME', 'BUY_SIGNAL']:
        exit_rules = """```
📤 EXIT RULES:
🛑 Stop: ขาดทุน > 4% → ขายทันที
⏰ 3 วันไม่กำไร > 1% → ขายทิ้ง
🎯 RSI > 70 หรือ Fear > 70 → ขาย
📈 กำไร > 3% → Stop ที่ทุน
⏳ ถือไม่เกิน 10 วัน
```"""
        embed["fields"].append({
            "name": "📤 Exit Strategy",
            "value": exit_rules,
            "inline": False
        })
    
    if chart_buf:
        embed["image"] = {"url": "attachment://chart.png"}
    
    try:
        import json
        if chart_buf:
            # Send with file attachment
            payload = {"embeds": [embed]}
            response = requests.post(
                config.DISCORD_WEBHOOK_URL,
                data={"payload_json": json.dumps(payload)},
                files={"file": ("chart.png", chart_buf, "image/png")}
            )
        else:
            # Send without file
            response = requests.post(
                config.DISCORD_WEBHOOK_URL,
                json={"embeds": [embed]},
                headers={"Content-Type": "application/json"}
            )
        
        if response.status_code in [200, 204]:
            print(f"Discord alert with chart sent! ({signal_type})")
            return True
        else:
            print(f"Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def send_daily_summary(
    fear_index: float,
    spx_price: float,
    spx_return: float,
    vix: float,
    rsi: float,
    model_signal: bool,
    probability: float,
    forecast_min: float = None
):
    """ส่ง Daily Summary - รวมทุกอย่างในข้อความเดียว"""
    
    if not config.DISCORD_WEBHOOK_URL:
        return False
    
    # Determine market status
    if fear_index < 10:
        status = "🔴 EXTREME FEAR"
        color = 0xFF0000
    elif fear_index < 25:
        status = "🟠 FEAR ZONE"
        color = 0xFFA500
    elif fear_index < 45:
        status = "🟡 NEUTRAL"
        color = 0xFFFF00
    elif fear_index < 75:
        status = "🟢 GREED"
        color = 0x00FF00
    else:
        status = "🟢 EXTREME GREED"
        color = 0x00FF00
    
    # Signal status
    if model_signal and probability > 0.7:
        signal_text = "🟢 BUY SIGNAL (High Confidence)"
    elif model_signal and probability > 0.5:
        signal_text = "🟡 BUY SIGNAL (Moderate)"
    else:
        signal_text = "⚪ NO SIGNAL"
    
    embed = {
        "title": f"📊 Daily Market Report | {datetime.now().strftime('%Y-%m-%d')}",
        "color": color,
        "fields": [
            {
                "name": f"━━━ MARKET STATUS: {status} ━━━",
                "value": f"```\nFear & Greed:  {fear_index:.0f}/100\nS&P 500:       ${spx_price:,.0f} ({spx_return*100:+.2f}%)\nVIX:           {vix:.1f}\nRSI:           {rsi:.0f}\n```",
                "inline": False
            },
            {
                "name": "🤖 Model Signal",
                "value": f"```{signal_text}\nConfidence: {probability*100:.0f}%```",
                "inline": True
            },
        ],
        "footer": {
            "text": "Fear & Greed Alert System"
        }
    }
    
    # Add forecast if available
    if forecast_min is not None:
        embed["fields"].append({
            "name": "🔮 5-Day Forecast",
            "value": f"```Fear อาจลงไปถึง: {forecast_min:.0f}```",
            "inline": True
        })
    
    try:
        response = requests.post(
            config.DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 204
    except:
        return False


def test_webhook():
    """ทดสอบ Webhook"""
    return send_discord_alert(
        fear_index=18,
        probability=0.75,
        signal_type='WATCH',
        spx_price=4850.00,
        vix=22.5,
        rsi=32,
        additional_info={
            "Drawdown": "-2.5%",
            "Crash Warning": "NO",
            "Recovery Signal": "YES"
        }
    )
