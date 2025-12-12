"""
Train V2 - Improved training with multiple targets and better evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import get_fear_greed_historical, get_market_data, calculate_synthetic_fear_greed
from feature_engineering_v2 import create_features_v2, get_feature_columns_v2
from models_v2 import ModelComparisonV2
from backtest import Backtester, compare_thresholds, compare_exit_strategies, WalkForwardValidator
import config


def train_v2():
    print("=" * 70)
    print("FEAR & GREED ALERT SYSTEM V2 - IMPROVED TRAINING")
    print("=" * 70)
    print(f"Start Date: {config.START_DATE}")
    print(f"End Date: {datetime.now().strftime('%Y-%m-%d')}")
    print()
    
    # 1. Load Data
    print("Loading data...")
    fear_data = get_fear_greed_historical()
    market_data = get_market_data()
    
    if fear_data is None or market_data is None:
        print("Error loading data!")
        return
    
    # 2. Create V2 Features
    print("Creating V2 features...")
    data = create_features_v2(market_data, fear_data)
    print(f"Dataset: {len(data)} rows")
    
    # 3. Compare different targets - BUY signals (STABLE targets)
    buy_targets = {
        'Stable_Buy': 'Stable Buy (works in all regimes)',
        'Regime_Buy': 'Regime-Aware Buy (adapts to Bull/Bear)',
        'Conservative_Buy': 'Conservative Buy (high precision)',
        'Safe_Entry': 'Safe Entry (5d return > 0, max DD > -3%)',
    }
    
    # SELL targets (STABLE)
    sell_targets = {
        'Good_Exit': 'Good Exit (10d return < -2%, limited upside)',
        'Safe_Exit': 'Safe Exit (Greed > 70, RSI > 60)',
        'Confirmed_Sell': 'Confirmed Sell (4+ signals align)',
    }
    
    targets = buy_targets
    
    # Use PRUNED features (ตัด noise ออก - ลด overfit)
    feature_cols = get_feature_columns_v2('all', pruned=True)
    
    # Filter features that exist
    feature_cols = [f for f in feature_cols if f in data.columns]
    print(f"Using {len(feature_cols)} PRUNED features (reduced from 93)")
    print(f"Features: {feature_cols[:10]}... (showing first 10)")
    
    X = data[feature_cols]
    
    # Time series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    all_results = {}
    best_overall = {'score': 0, 'target': None, 'model': None}
    
    for target_name, target_desc in targets.items():
        print(f"\n{'#' * 70}")
        print(f"TARGET: {target_name}")
        print(f"Description: {target_desc}")
        print(f"{'#' * 70}")
        
        y = data[target_name]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train: {len(y_train)} rows, Positive: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"Test: {len(y_test)} rows, Positive: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        
        if y_train.sum() < 10:
            print("Not enough positive samples, skipping...")
            continue
        
        # Train models
        model = ModelComparisonV2()
        results = model.train_and_evaluate_v2(X_train, X_test, y_train, y_test, target_name=target_name)
        
        all_results[target_name] = {
            'results': results,
            'model': model,
            'best_model_name': model.best_model_name,
            'best_score': results[model.best_model_name]['custom_score'] if model.best_model_name else 0
        }
        
        # Track overall best
        if model.best_model_name and results[model.best_model_name]['custom_score'] > best_overall['score']:
            best_overall = {
                'score': results[model.best_model_name]['custom_score'],
                'target': target_name,
                'model': model,
                'precision': results[model.best_model_name]['precision'],
                'recall': results[model.best_model_name]['recall']
            }
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL TARGETS")
    print("=" * 70)
    print(f"{'Target':<20} {'Best Model':<25} {'Precision':>10} {'Recall':>10} {'Score':>10}")
    print("-" * 75)
    
    for target_name, res in all_results.items():
        if res['best_model_name']:
            r = res['results'][res['best_model_name']]
            print(f"{target_name:<20} {res['best_model_name']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['custom_score']:>10.4f}")
    
    # 5. Save best BUY model
    print("\n" + "=" * 70)
    print("BEST BUY MODEL CONFIGURATION")
    print("=" * 70)
    print(f"Target: {best_overall['target']}")
    print(f"Precision: {best_overall['precision']:.4f}")
    print(f"Recall: {best_overall['recall']:.4f}")
    print(f"Score: {best_overall['score']:.4f}")
    
    if best_overall['model']:
        best_overall['model'].save_model('_v2')
        
        # Save which target was used
        with open(f"{config.MODEL_DIR}/best_target.txt", 'w') as f:
            f.write(best_overall['target'])
    
    # ============================================
    # TRAIN SELL MODELS (NEW!)
    # ============================================
    print("\n" + "#" * 70)
    print("TRAINING SELL MODELS")
    print("#" * 70)
    
    sell_results = {}
    best_sell = {'score': 0, 'target': None, 'model': None}
    
    for target_name, target_desc in sell_targets.items():
        print(f"\n{'#' * 70}")
        print(f"TARGET: {target_name}")
        print(f"Description: {target_desc}")
        print(f"{'#' * 70}")
        
        if target_name not in data.columns:
            print(f"Target {target_name} not found in data, skipping...")
            continue
            
        y = data[target_name]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train: {len(y_train)} rows, Positive: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"Test: {len(y_test)} rows, Positive: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        
        if y_train.sum() < 10:
            print("Not enough positive samples, skipping...")
            continue
        
        # Train models
        model = ModelComparisonV2()
        results = model.train_and_evaluate_v2(X_train, X_test, y_train, y_test, target_name=target_name)
        
        sell_results[target_name] = {
            'results': results,
            'model': model,
            'best_model_name': model.best_model_name,
            'best_score': results[model.best_model_name]['custom_score'] if model.best_model_name else 0
        }
        
        # Track best sell model
        if model.best_model_name and results[model.best_model_name]['custom_score'] > best_sell['score']:
            best_sell = {
                'score': results[model.best_model_name]['custom_score'],
                'target': target_name,
                'model': model,
                'precision': results[model.best_model_name]['precision'],
                'recall': results[model.best_model_name]['recall']
            }
    
    # Save best SELL model
    print("\n" + "=" * 70)
    print("BEST SELL MODEL CONFIGURATION")
    print("=" * 70)
    
    if best_sell['model']:
        print(f"Target: {best_sell['target']}")
        print(f"Precision: {best_sell['precision']:.4f}")
        print(f"Recall: {best_sell['recall']:.4f}")
        print(f"Score: {best_sell['score']:.4f}")
        
        best_sell['model'].save_model('_sell_v2')
        
        with open(f"{config.MODEL_DIR}/best_sell_target.txt", 'w') as f:
            f.write(best_sell['target'])
    else:
        print("No valid sell model found")
    
    # 6. Backtest best BUY model (with Dynamic Exit!)
    print("\n" + "=" * 70)
    print("BACKTESTING BEST BUY MODEL (Dynamic Exit)")
    print("=" * 70)
    
    if best_overall['model']:
        test_data = data.iloc[split_idx:].copy()
        y_test = data[best_overall['target']].iloc[split_idx:]
        
        predictions, probabilities = best_overall['model'].predict(X_test)
        
        # Add future return for backtest
        test_data['Prediction'] = predictions
        test_data['Probability'] = probabilities
        
        bt = Backtester(test_data, predictions, probabilities)
        bt_results = bt.run_backtest(prob_threshold=0.5, use_dynamic_exit=True)
        
        if bt_results:
            bt.print_results()
            
            # Compare thresholds
            print("\nThreshold Comparison:")
            compare_thresholds(test_data, predictions, probabilities)
            
            # Compare exit strategies
            print("\nExit Strategy Comparison:")
            compare_exit_strategies(test_data, predictions, probabilities)
    
    # 6b. Backtest best SELL model
    print("\n" + "=" * 70)
    print("BACKTESTING BEST SELL MODEL")
    print("=" * 70)
    
    if best_sell['model']:
        sell_predictions, sell_probabilities = best_sell['model'].predict(X_test)
        
        test_data['Sell_Prediction'] = sell_predictions
        test_data['Sell_Probability'] = sell_probabilities
        
        # Simple sell backtest stats
        sell_signals = test_data[test_data['Sell_Prediction'] == 1]
        if len(sell_signals) > 0:
            avg_return_5d = sell_signals['Future_Return_5d'].mean() if 'Future_Return_5d' in sell_signals.columns else 0
            avg_return_10d = sell_signals['Future_Return_10d'].mean() if 'Future_Return_10d' in sell_signals.columns else 0
            
            print(f"Total Sell Signals: {len(sell_signals)}")
            print(f"Avg 5d Return after Sell Signal: {avg_return_5d*100:.2f}%")
            print(f"Avg 10d Return after Sell Signal: {avg_return_10d*100:.2f}%")
            print(f"(Negative = good sell signal)")
        else:
            print("No sell signals generated in test period")
    
    # 7. Feature Importance
    print("\n" + "=" * 70)
    print("TOP 20 FEATURE IMPORTANCE")
    print("=" * 70)
    
    if best_overall['model'] and hasattr(best_overall['model'].best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_overall['model'].best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importance.head(20).to_string(index=False))
    
    # 8. Walk-Forward Validation (NEW!)
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)
    
    try:
        wf = WalkForwardValidator(train_years=3, test_years=1)
        wf_results = wf.run_walk_forward(
            data=data,
            feature_cols=feature_cols,
            target_col=best_overall['target'],
            model_class=ModelComparisonV2
        )
        
        if wf_results is not None:
            print("\nWalk-Forward Backtest:")
            wf_predictions = wf_results['Prediction'].values
            wf_probabilities = wf_results['Probability'].values
            bt_wf = Backtester(wf_results, wf_predictions, wf_probabilities)
            bt_wf_results = bt_wf.run_backtest(prob_threshold=0.5, use_dynamic_exit=True)
            if bt_wf_results:
                bt_wf.print_results()
    except Exception as e:
        print(f"Walk-Forward Validation failed: {e}")
    
    # 9. Train Fear Predictor
    print("\n" + "=" * 70)
    print("TRAINING FEAR PREDICTOR")
    print("=" * 70)
    
    from fear_predictor import FearPredictor, analyze_fear_patterns
    
    fear_pred = FearPredictor()
    fear_pred.train(data)
    fear_pred.save()
    
    # Test prediction
    forecast = fear_pred.predict(data)
    print(f"\nCurrent Fear: {forecast['current_fear']:.0f}")
    print(f"Predicted Min (5d): {forecast['predicted_min_5d']:.0f}")
    print(f"Prob of Significant Drop: {forecast['prob_significant_drop']*100:.0f}%")
    print(f"\nForecast Message:")
    print(fear_pred.get_forecast_message(forecast))
    
    # Analyze patterns
    analyze_fear_patterns(data)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Models trained:")
    print(f"  - BUY model: {best_overall['target']} (Precision: {best_overall['precision']:.2%})")
    if best_sell['model']:
        print(f"  - SELL model: {best_sell['target']} (Precision: {best_sell['precision']:.2%})")
    print("\nRun 'python daily_check_v2.py' to check today's signal")
    
    return all_results, best_overall, sell_results, best_sell


if __name__ == "__main__":
    train_v2()
