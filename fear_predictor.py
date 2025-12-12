"""
Fear Predictor - ทำนายว่า Fear จะลงไปถึงระดับไหน
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import config


class FearPredictor:
    """ทำนาย Fear Index ในอนาคต"""
    
    def __init__(self):
        self.model_min = None  # ทำนาย minimum fear ใน 5 วัน
        self.model_direction = None  # ทำนายทิศทาง
        self.scaler = StandardScaler()
        self.model_path = f"{config.MODEL_DIR}/fear_predictor.pkl"
        
    def create_features(self, data):
        """สร้าง features สำหรับทำนาย Fear"""
        df = data.copy()
        
        # Fear momentum
        df['Fear_Change_1d'] = df['FearIndex'].diff(1)
        df['Fear_Change_3d'] = df['FearIndex'].diff(3)
        df['Fear_Change_5d'] = df['FearIndex'].diff(5)
        df['Fear_SMA_5'] = df['FearIndex'].rolling(5).mean()
        df['Fear_SMA_10'] = df['FearIndex'].rolling(10).mean()
        df['Fear_Momentum'] = df['Fear_SMA_5'] - df['Fear_SMA_10']
        
        # Fear volatility
        df['Fear_Std_5d'] = df['FearIndex'].rolling(5).std()
        df['Fear_Std_10d'] = df['FearIndex'].rolling(10).std()
        
        # Fear position
        df['Fear_Min_10d'] = df['FearIndex'].rolling(10).min()
        df['Fear_Max_10d'] = df['FearIndex'].rolling(10).max()
        df['Fear_Range'] = df['Fear_Max_10d'] - df['Fear_Min_10d']
        df['Fear_Position'] = (df['FearIndex'] - df['Fear_Min_10d']) / (df['Fear_Range'] + 0.01)
        
        # Market conditions
        df['VIX_Change_5d'] = df['VIX'].diff(5)
        df['SPX_Return_5d'] = df['Close'].pct_change(5)
        df['SPX_Return_10d'] = df['Close'].pct_change(10)
        df['SPX_Drawdown'] = (df['Close'] - df['Close'].rolling(20).max()) / df['Close'].rolling(20).max()
        
        # RSI
        df['RSI_Level'] = df['RSI']
        
        # Targets: minimum fear in next 5 days
        df['Future_Fear_Min_5d'] = df['FearIndex'].rolling(5).min().shift(-5)
        df['Future_Fear_Min_10d'] = df['FearIndex'].rolling(10).min().shift(-10)
        df['Fear_Will_Drop'] = (df['Future_Fear_Min_5d'] < df['FearIndex'] - 5).astype(int)
        
        return df.dropna()
    
    def get_feature_cols(self):
        return [
            'FearIndex', 'Fear_Change_1d', 'Fear_Change_3d', 'Fear_Change_5d',
            'Fear_SMA_5', 'Fear_SMA_10', 'Fear_Momentum',
            'Fear_Std_5d', 'Fear_Std_10d',
            'Fear_Min_10d', 'Fear_Max_10d', 'Fear_Range', 'Fear_Position',
            'VIX', 'VIX_Change_5d',
            'SPX_Return_5d', 'SPX_Return_10d', 'SPX_Drawdown',
            'RSI_Level'
        ]
    
    def train(self, data):
        """Train Fear Predictor"""
        print("Training Fear Predictor...")
        
        df = self.create_features(data)
        feature_cols = self.get_feature_cols()
        
        X = df[feature_cols]
        y_min = df['Future_Fear_Min_5d']
        y_drop = df['Fear_Will_Drop']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train regressor for minimum fear
        self.model_min = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.model_min.fit(X_scaled, y_min)
        
        # Train classifier for direction
        self.model_direction = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model_direction.fit(X_scaled, y_drop)
        
        # Evaluate
        y_pred_min = self.model_min.predict(X_scaled)
        mae = np.abs(y_min - y_pred_min).mean()
        print(f"  Min Fear MAE: {mae:.2f}")
        
        y_pred_drop = self.model_direction.predict(X_scaled)
        acc = (y_drop == y_pred_drop).mean()
        print(f"  Drop Direction Accuracy: {acc*100:.1f}%")
        
        return self
    
    def predict(self, data):
        """ทำนาย Fear ในอนาคต"""
        df = self.create_features(data)
        feature_cols = self.get_feature_cols()
        
        latest = df[feature_cols].iloc[-1:]
        X_scaled = self.scaler.transform(latest)
        
        # Predict minimum fear
        pred_min = self.model_min.predict(X_scaled)[0]
        
        # Predict if fear will drop significantly
        prob_drop = self.model_direction.predict_proba(X_scaled)[0][1]
        
        current_fear = df['FearIndex'].iloc[-1]
        
        return {
            'current_fear': current_fear,
            'predicted_min_5d': pred_min,
            'expected_drop': current_fear - pred_min,
            'prob_significant_drop': prob_drop,
            'fear_momentum': df['Fear_Momentum'].iloc[-1],
            'fear_volatility': df['Fear_Std_5d'].iloc[-1]
        }
    
    def get_forecast_message(self, prediction):
        """สร้างข้อความทำนาย"""
        current = prediction['current_fear']
        pred_min = prediction['predicted_min_5d']
        prob_drop = prediction['prob_significant_drop']
        momentum = prediction['fear_momentum']
        
        messages = []
        
        # Current status
        if current < 10:
            messages.append(f"🔴 EXTREME FEAR NOW ({current:.0f})")
        elif current < 25:
            messages.append(f"🟡 Fear Zone ({current:.0f})")
        else:
            messages.append(f"🟢 Normal ({current:.0f})")
        
        # Prediction
        if pred_min < 10:
            messages.append(f"📉 อาจลงไปถึง EXTREME ({pred_min:.0f}) ใน 5 วัน")
        elif pred_min < 15:
            messages.append(f"📉 อาจลงไปถึง High Fear ({pred_min:.0f}) ใน 5 วัน")
        elif pred_min < 20:
            messages.append(f"📉 อาจลงไปถึง Fear Zone ({pred_min:.0f}) ใน 5 วัน")
        elif pred_min < current - 5:
            messages.append(f"📉 อาจลงอีก ({pred_min:.0f}) ใน 5 วัน")
        else:
            messages.append(f"➡️ น่าจะทรงตัว (~{pred_min:.0f})")
        
        # Probability
        if prob_drop > 0.7:
            messages.append(f"⚠️ โอกาสลงแรง: {prob_drop*100:.0f}%")
        elif prob_drop > 0.5:
            messages.append(f"📊 โอกาสลง: {prob_drop*100:.0f}%")
        
        # Momentum
        if momentum < -3:
            messages.append("📈 Fear กำลังลดลงเร็ว (ดี)")
        elif momentum > 3:
            messages.append("📉 Fear กำลังเพิ่มขึ้น (ระวัง)")
        
        # Recommendation
        if current < 25 and pred_min < 15 and prob_drop > 0.5:
            messages.append("💡 แนะนำ: รอดูก่อน อาจลงอีก")
        elif current < 20 and momentum < 0:
            messages.append("💡 แนะนำ: เริ่มจับตามอง Fear กำลังฟื้น")
        elif current > 25 and pred_min < 20:
            messages.append("💡 แนะนำ: เตรียมตัว อาจมีโอกาสเร็วๆนี้")
        
        return "\n".join(messages)
    
    def save(self):
        """Save models"""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump({
            'model_min': self.model_min,
            'model_direction': self.model_direction,
            'scaler': self.scaler
        }, self.model_path)
        print(f"Fear Predictor saved to {self.model_path}")
    
    def load(self):
        """Load models"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model_min = data['model_min']
            self.model_direction = data['model_direction']
            self.scaler = data['scaler']
            return True
        return False


def analyze_fear_patterns(data):
    """วิเคราะห์ pattern ของ Fear"""
    print("\n" + "="*60)
    print("FEAR PATTERN ANALYSIS")
    print("="*60)
    
    # หา extreme fear events
    extreme_events = []
    in_extreme = False
    start_idx = None
    
    for i, row in data.iterrows():
        if row['FearIndex'] < 15 and not in_extreme:
            in_extreme = True
            start_idx = i
        elif row['FearIndex'] >= 20 and in_extreme:
            in_extreme = False
            extreme_events.append({
                'start': start_idx,
                'end': i,
                'min_fear': data.loc[start_idx:i, 'FearIndex'].min(),
                'duration': (i - start_idx).days,
                'spx_at_min': data.loc[data.loc[start_idx:i, 'FearIndex'].idxmin(), 'Close'],
                'spx_after_10d': data.loc[i:, 'Close'].iloc[10] if len(data.loc[i:]) > 10 else None
            })
    
    if extreme_events:
        print(f"\nFound {len(extreme_events)} Extreme Fear Events:")
        print(f"{'Start':<12} {'Min Fear':>10} {'Duration':>10} {'SPX Return 10d':>15}")
        print("-"*50)
        
        for event in extreme_events[-10:]:  # Last 10
            if event['spx_after_10d']:
                ret = (event['spx_after_10d'] / event['spx_at_min'] - 1) * 100
                print(f"{event['start'].strftime('%Y-%m-%d'):<12} {event['min_fear']:>10.0f} {event['duration']:>10}d {ret:>14.1f}%")
        
        # Statistics
        returns = [(e['spx_after_10d']/e['spx_at_min']-1)*100 for e in extreme_events if e['spx_after_10d']]
        print(f"\nAverage 10d return after extreme fear: {np.mean(returns):.1f}%")
        print(f"Win rate (positive return): {sum(1 for r in returns if r > 0)/len(returns)*100:.0f}%")
    
    return extreme_events
