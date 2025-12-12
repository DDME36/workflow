"""
Backtest V2 - ทดสอบว่า Signal ทำกำไรจริงไหม
รวม Dynamic Exit Strategy และ Walk-Forward Validation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


def is_safe_to_trade(row):
    """
    Circuit Breaker - ตัดวงจรถ้าความเสี่ยงสูงเกินไป
    
    Returns:
        bool: True = ปลอดภัย, False = ห้ามเทรด
    """
    # 1. ห้ามสวนเทรนด์ใหญ่: ราคาใต้ SMA200 + VIX สูง = ตลาด Panic
    if 'SMA_200' in row.index and 'VIX' in row.index:
        if row['Close'] < row['SMA_200'] and row['VIX'] > 30:
            return False
    
    # 2. ห้ามรับมีด: RSI ต่ำมาก + Momentum ยังดิ่ง
    if 'RSI' in row.index and 'MACD_Diff' in row.index:
        if row['RSI'] < 25 and row['MACD_Diff'] < -0.5:
            return False
    
    # 3. ห้ามเทรดช่วง Crash: VIX > 40 = ตลาดนรก
    if 'VIX' in row.index:
        if row['VIX'] > 40:
            return False
    
    # 4. ห้ามเทรดก่อน FOMC: ถ้า Days_To_FOMC <= 2
    if 'Days_To_FOMC' in row.index:
        if row['Days_To_FOMC'] <= 2:
            return False
    
    return True


class Backtester:
    """Backtest Trading Strategy with Dynamic Exit + Circuit Breaker"""
    
    def __init__(self, data, predictions, probabilities):
        self.data = data.copy()
        self.data['Prediction'] = predictions
        self.data['Probability'] = probabilities
        self.results = {}
        
    def run_backtest(self, prob_threshold=0.5, hold_days=5, use_dynamic_exit=True):
        """
        Backtest Strategy with Dynamic Exit:
        - ซื้อเมื่อ Prediction = 1 และ Probability > threshold
        - ขายตาม Dynamic Rules หรือ hold_days
        """
        trades = []
        position = None
        
        for i in range(len(self.data) - hold_days - 5):
            row = self.data.iloc[i]
            
            if position is None:
                # Check for buy signal + Circuit Breaker
                if row['Prediction'] == 1 and row['Probability'] > prob_threshold and is_safe_to_trade(row):
                    entry_price = row['Close']
                    entry_date = self.data.index[i]
                    entry_idx = i
                    
                    # Dynamic Exit Logic
                    if use_dynamic_exit:
                        exit_idx, exit_reason = self._find_dynamic_exit(
                            entry_idx, entry_price, hold_days
                        )
                    else:
                        exit_idx = min(i + hold_days, len(self.data) - 1)
                        exit_reason = 'time_stop'
                    
                    exit_price = self.data.iloc[exit_idx]['Close']
                    exit_date = self.data.index[exit_idx]
                    
                    pnl = (exit_price - entry_price) / entry_price
                    hold_period = exit_idx - entry_idx
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl * 100,
                        'hold_days': hold_period,
                        'exit_reason': exit_reason,
                        'fear_at_entry': row['FearIndex'],
                        'probability': row['Probability']
                    })
        
        if not trades:
            print("No trades generated!")
            return None
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
        win_rate = winning_trades / total_trades * 100
        
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        total_return = trades_df['pnl_pct'].sum()
        avg_return = trades_df['pnl_pct'].mean()
        avg_hold = trades_df['hold_days'].mean()
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max Drawdown
        cumulative = (1 + trades_df['pnl_pct']/100).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'avg_return': avg_return,
            'avg_hold_days': avg_hold,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'trades': trades_df
        }
        
        return self.results
    
    def _find_dynamic_exit(self, entry_idx, entry_price, max_hold_days=10):
        """
        Dynamic Exit Strategy V3 - Aggressive (ลด Drawdown!)
        
        Priority Order:
        1. Stop Loss: ขาดทุน > 4% ให้ cut ทันที (เข้มขึ้น)
        2. Stagnation Kill: 3 วันไม่วิ่ง (< 1%) ขายทิ้ง
        3. Technical Exit: RSI > 70 หรือ Fear > 70
        4. Aggressive Trailing Stop: 3% -> breakeven, 5% -> trail 2%
        5. Max Hold: 10 วัน (ลดจาก 15)
        """
        max_profit = 0
        trailing_stop_active = False
        trailing_stop_price = 0
        
        for day in range(1, max_hold_days + 1):
            idx = entry_idx + day
            if idx >= len(self.data):
                return len(self.data) - 1, 'end_of_data'
            
            current_price = self.data.iloc[idx]['Close']
            current_return = (current_price - entry_price) / entry_price
            current_rsi = self.data.iloc[idx]['RSI'] if 'RSI' in self.data.columns else 50
            current_fear = self.data.iloc[idx]['FearIndex'] if 'FearIndex' in self.data.columns else 50
            
            # Update max profit
            if current_return > max_profit:
                max_profit = current_return
            
            # Rule 1: Stop Loss (-4%) - เข้มขึ้น!
            if current_return < -0.04:
                return idx, 'stop_loss'
            
            # Rule 2: Stagnation Kill (3 วันไม่วิ่ง ขายทิ้ง!)
            if day == 3 and current_return < 0.01:
                return idx, 'stagnation_kill'
            
            # Rule 3: Technical Exit (RSI > 70 or Fear > 70)
            if current_return > 0.005:  # มีกำไรนิดหน่อย
                if current_rsi > 70:
                    return idx, 'rsi_overbought'
                if current_fear > 70:
                    return idx, 'greed_exit'
            
            # Rule 4: Aggressive Trailing Stop
            if current_return > 0.05:  # กำไร > 5%
                # Trail 2% จากราคาปัจจุบัน
                new_stop = current_price * 0.98
                if new_stop > trailing_stop_price:
                    trailing_stop_active = True
                    trailing_stop_price = new_stop
            elif current_return > 0.03:  # กำไร > 3%
                # Lock at breakeven + ค่าคอม
                new_stop = entry_price * 1.005
                if not trailing_stop_active or new_stop > trailing_stop_price:
                    trailing_stop_active = True
                    trailing_stop_price = new_stop
            
            if trailing_stop_active and current_price < trailing_stop_price:
                return idx, 'trailing_stop'
            
            # Rule 5: Max Hold (10 วัน)
            if day >= max_hold_days:
                return idx, 'max_hold'
        
        return entry_idx + max_hold_days, 'max_hold'
    
    def print_results(self):
        """แสดงผล Backtest"""
        if not self.results:
            print("Run backtest first!")
            return
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS (Dynamic Exit)")
        print("=" * 60)
        print(f"Total Trades:    {self.results['total_trades']}")
        print(f"Winning Trades:  {self.results['winning_trades']}")
        print(f"Losing Trades:   {self.results['losing_trades']}")
        print(f"Win Rate:        {self.results['win_rate']:.2f}%")
        print(f"Avg Win:         {self.results['avg_win']:.2f}%")
        print(f"Avg Loss:        {self.results['avg_loss']:.2f}%")
        print(f"Total Return:    {self.results['total_return']:.2f}%")
        print(f"Avg Return/Trade:{self.results['avg_return']:.2f}%")
        print(f"Avg Hold Days:   {self.results['avg_hold_days']:.1f}")
        print(f"Profit Factor:   {self.results['profit_factor']:.2f}")
        print(f"Max Drawdown:    {self.results['max_drawdown']:.2f}%")
        print("=" * 60)
        
        # Exit reasons
        print("\nExit Reasons:")
        for reason, count in self.results['exit_reasons'].items():
            pct = count / self.results['total_trades'] * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Show sample trades
        print("\nSample Trades:")
        print(self.results['trades'].head(10).to_string())
    
    def plot_equity_curve(self, save_path=None):
        """Plot Equity Curve"""
        if not self.results or 'trades' not in self.results:
            return
        
        trades = self.results['trades']
        trades['cumulative_return'] = (1 + trades['pnl_pct']/100).cumprod() - 1
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(trades['entry_date'], trades['cumulative_return'] * 100)
        ax1.set_title('Equity Curve (Cumulative Return %)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return %')
        ax1.grid(True)
        
        # PnL Distribution
        ax2 = axes[0, 1]
        ax2.hist(trades['pnl_pct'], bins=20, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('PnL Distribution')
        ax2.set_xlabel('Return %')
        ax2.set_ylabel('Frequency')
        
        # Fear Index at Entry
        ax3 = axes[1, 0]
        ax3.scatter(trades['fear_at_entry'], trades['pnl_pct'], alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('PnL vs Fear Index at Entry')
        ax3.set_xlabel('Fear Index')
        ax3.set_ylabel('Return %')
        
        # Win Rate by Fear Level
        ax4 = axes[1, 1]
        fear_bins = [0, 10, 15, 20, 25, 100]
        trades['fear_bin'] = pd.cut(trades['fear_at_entry'], bins=fear_bins)
        win_rate_by_fear = trades.groupby('fear_bin').apply(
            lambda x: (x['pnl_pct'] > 0).mean() * 100
        )
        win_rate_by_fear.plot(kind='bar', ax=ax4)
        ax4.set_title('Win Rate by Fear Level')
        ax4.set_xlabel('Fear Index Range')
        ax4.set_ylabel('Win Rate %')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Chart saved to {save_path}")
        
        plt.show()


# ============================================
# WALK-FORWARD VALIDATION (NEW!)
# ============================================

class WalkForwardValidator:
    """
    Walk-Forward Validation สำหรับทดสอบโมเดลแบบ realistic
    Train บน window เก่า -> Test บน window ใหม่ -> เลื่อนไปเรื่อยๆ
    """
    
    def __init__(self, train_years=3, test_years=1):
        self.train_years = train_years
        self.test_years = test_years
        self.results = []
        
    def run_walk_forward(self, data, feature_cols, target_col, model_class):
        """
        Run Walk-Forward Validation
        
        Args:
            data: DataFrame with features and target
            feature_cols: list of feature column names
            target_col: target column name
            model_class: Model class to use (e.g., ModelComparisonV2)
        """
        from datetime import timedelta
        
        print("=" * 70)
        print("WALK-FORWARD VALIDATION")
        print(f"Train Window: {self.train_years} years")
        print(f"Test Window: {self.test_years} year")
        print("=" * 70)
        
        # Get date range
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Calculate windows
        train_days = self.train_years * 252  # Trading days
        test_days = self.test_years * 252
        
        current_start = start_date
        fold = 1
        all_predictions = []
        
        while True:
            train_end = current_start + timedelta(days=train_days * 1.4)  # Approximate
            test_start = train_end
            test_end = test_start + timedelta(days=test_days * 1.4)
            
            if test_end > end_date:
                break
            
            # Get train/test data
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) < 100 or len(test_data) < 20:
                current_start = current_start + timedelta(days=test_days * 1.4)
                continue
            
            print(f"\n--- Fold {fold} ---")
            print(f"Train: {train_data.index.min().date()} to {train_data.index.max().date()} ({len(train_data)} days)")
            print(f"Test:  {test_data.index.min().date()} to {test_data.index.max().date()} ({len(test_data)} days)")
            
            # Prepare data
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Train model
            model = model_class()
            try:
                results = model.train_and_evaluate_v2(
                    X_train, X_test, y_train, y_test, 
                    target_name=target_col
                )
                
                # Get predictions
                predictions, probabilities = model.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                
                # Store results
                fold_result = {
                    'fold': fold,
                    'train_start': train_data.index.min(),
                    'train_end': train_data.index.max(),
                    'test_start': test_data.index.min(),
                    'test_end': test_data.index.max(),
                    'precision': precision,
                    'recall': recall,
                    'n_signals': predictions.sum(),
                    'n_test': len(y_test)
                }
                self.results.append(fold_result)
                
                # Store predictions for backtest
                test_data_copy = test_data.copy()
                test_data_copy['Prediction'] = predictions
                test_data_copy['Probability'] = probabilities
                all_predictions.append(test_data_copy)
                
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Signals: {predictions.sum()}")
                
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
            
            # Move to next window
            current_start = current_start + timedelta(days=test_days * 1.4)
            fold += 1
        
        # Summary
        self.print_summary()
        
        # Combine all predictions for full backtest
        if all_predictions:
            combined = pd.concat(all_predictions)
            return combined
        return None
    
    def print_summary(self):
        """Print Walk-Forward Summary"""
        if not self.results:
            print("No results to show")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 70)
        print("WALK-FORWARD SUMMARY")
        print("=" * 70)
        print(f"Total Folds: {len(df)}")
        print(f"Avg Precision: {df['precision'].mean():.4f} (std: {df['precision'].std():.4f})")
        print(f"Avg Recall: {df['recall'].mean():.4f} (std: {df['recall'].std():.4f})")
        print(f"Total Signals: {df['n_signals'].sum()}")
        print("\nPer-Fold Results:")
        print(df.to_string(index=False))
        
        # Check for consistency
        if df['precision'].std() > 0.15:
            print("\n⚠️ WARNING: High variance in precision - model may not be stable")
        if df['precision'].min() < 0.4:
            print("⚠️ WARNING: Some folds have low precision - check for regime changes")


def compare_thresholds(data, predictions, probabilities):
    """เปรียบเทียบ Threshold ต่างๆ"""
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON")
    print("=" * 60)
    
    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        bt = Backtester(data, predictions, probabilities)
        res = bt.run_backtest(prob_threshold=threshold, use_dynamic_exit=True)
        if res:
            results.append({
                'threshold': threshold,
                'trades': res['total_trades'],
                'win_rate': res['win_rate'],
                'total_return': res['total_return'],
                'profit_factor': res['profit_factor'],
                'avg_hold': res['avg_hold_days']
            })
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        return df
    return None


def compare_exit_strategies(data, predictions, probabilities):
    """เปรียบเทียบ Exit Strategy"""
    print("\n" + "=" * 60)
    print("EXIT STRATEGY COMPARISON")
    print("=" * 60)
    
    results = []
    
    # Fixed hold
    for hold_days in [3, 5, 7, 10]:
        bt = Backtester(data, predictions, probabilities)
        res = bt.run_backtest(prob_threshold=0.5, hold_days=hold_days, use_dynamic_exit=False)
        if res:
            results.append({
                'strategy': f'Fixed {hold_days}d',
                'trades': res['total_trades'],
                'win_rate': res['win_rate'],
                'total_return': res['total_return'],
                'profit_factor': res['profit_factor'],
                'avg_hold': res['avg_hold_days']
            })
    
    # Dynamic exit
    for max_hold in [5, 7, 10]:
        bt = Backtester(data, predictions, probabilities)
        res = bt.run_backtest(prob_threshold=0.5, hold_days=max_hold, use_dynamic_exit=True)
        if res:
            results.append({
                'strategy': f'Dynamic (max {max_hold}d)',
                'trades': res['total_trades'],
                'win_rate': res['win_rate'],
                'total_return': res['total_return'],
                'profit_factor': res['profit_factor'],
                'avg_hold': res['avg_hold_days']
            })
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        return df
    return None
