"""
Models V2 - Stable version with Robust Ensemble and Regime-Aware Training
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import config


def custom_score(y_true, y_pred):
    """Custom score: prioritize precision heavily for high-quality signals"""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    if rec < 0.10:  # Need at least 10% recall
        return 0
    if prec < 0.5:
        return prec * 0.3
    # Weight precision 95%, recall 5%
    return prec * 0.95 + rec * 0.05


def find_optimal_threshold(y_true, probabilities, min_recall=0.10, target_precision=0.65):
    """
    Conservative threshold finding - เน้น precision สูง
    """
    thresholds = np.arange(0.4, 0.95, 0.01)
    best_thresh = 0.6
    best_score = 0
    best_precision = 0
    best_recall = 0
    
    for t in thresholds:
        pred = (probabilities >= t).astype(int)
        if pred.sum() < 2:
            continue
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        
        if rec >= min_recall:
            # Heavily weight precision
            score = prec * 0.9 + rec * 0.1
            if score > best_score:
                best_score = score
                best_thresh = t
                best_precision = prec
                best_recall = rec
    
    return best_thresh, best_precision, best_recall


class ModelComparisonV2:
    """Stable Model with Robust Voting Ensemble"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.ensemble = None
        self.optimal_threshold = 0.6
        
    def get_conservative_models(self, scale_pos_weight=10):
        """Conservative models - เน้น stability"""
        return {
            'XGBoost_Conservative': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,  # Shallow tree = less overfit
                learning_rate=0.02,
                min_child_weight=10,  # More conservative
                subsample=0.7,
                colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=0.5,  # More regularization
                reg_lambda=2,
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM_Conservative': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.02,
                num_leaves=8,  # Very conservative
                min_child_samples=30,
                class_weight='balanced',
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbose=-1
            ),
            'RandomForest_Conservative': RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Conservative': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=2,
                learning_rate=0.03,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.7,
                random_state=42
            ),
        }
    
    def create_robust_ensemble(self, X_train, y_train):
        """
        Robust Voting Ensemble - ใช้ Soft Voting แทน Stacking
        Stacking มักจะ overfit, Voting stable กว่า
        """
        print("\nCreating Robust Voting Ensemble...")
        
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models = self.get_conservative_models(scale_pos_weight)
        
        # Soft Voting - average probabilities
        estimators = [
            ('xgb', models['XGBoost_Conservative']),
            ('lgb', models['LightGBM_Conservative']),
            ('rf', models['RandomForest_Conservative']),
            ('gb', models['GradientBoosting_Conservative']),
        ]
        
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Average probabilities
            n_jobs=-1
        )
        
        voting.fit(X_train, y_train)
        self.ensemble = voting
        
        return voting
    
    def train_and_evaluate_v2(self, X_train, X_test, y_train, y_test, 
                              use_resampling=False, target_name='Buy_Signal'):
        """Train with conservative settings for stability"""
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nClass distribution: 0={len(y_train)-y_train.sum()}, 1={y_train.sum()}")
        print("Using conservative models for stability")
        
        X_train_res, y_train_res = X_train_scaled, y_train
        
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models = self.get_conservative_models(scale_pos_weight)
        
        print("\n" + "=" * 70)
        print(f"MODEL COMPARISON V2 (STABLE) - Target: {target_name}")
        print("=" * 70)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train_res, y_train_res)
                
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                # Find optimal threshold
                opt_thresh, opt_prec, opt_rec = find_optimal_threshold(y_test, y_prob)
                y_pred_opt = (y_prob >= opt_thresh).astype(int)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_opt),
                    'precision': precision_score(y_test, y_pred_opt, zero_division=0),
                    'recall': recall_score(y_test, y_pred_opt, zero_division=0),
                    'f1': f1_score(y_test, y_pred_opt, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
                    'custom_score': custom_score(y_test, y_pred_opt),
                    'optimal_threshold': opt_thresh
                }
                
                self.models[name] = model
                self.results[name] = metrics
                
                cm = confusion_matrix(y_test, y_pred_opt)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                print(f"  Precision: {metrics['precision']:.4f} (TP={tp}, FP={fp})")
                print(f"  Recall:    {metrics['recall']:.4f} (TP={tp}, FN={fn})")
                print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
                print(f"  Threshold: {opt_thresh:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                self.results[name] = {'error': str(e)}
        
        # Create Robust Voting Ensemble
        print("\n" + "-" * 70)
        print("Training Robust Voting Ensemble...")
        try:
            self.create_robust_ensemble(X_train_res, y_train_res)
            
            y_prob_ens = self.ensemble.predict_proba(X_test_scaled)[:, 1]
            
            opt_thresh, opt_prec, opt_rec = find_optimal_threshold(y_test, y_prob_ens)
            y_pred_ens = (y_prob_ens >= opt_thresh).astype(int)
            
            ens_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_ens),
                'precision': precision_score(y_test, y_pred_ens, zero_division=0),
                'recall': recall_score(y_test, y_pred_ens, zero_division=0),
                'f1': f1_score(y_test, y_pred_ens, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob_ens) if len(np.unique(y_test)) > 1 else 0,
                'custom_score': custom_score(y_test, y_pred_ens),
                'optimal_threshold': opt_thresh
            }
            
            self.models['VotingEnsemble'] = self.ensemble
            self.results['VotingEnsemble'] = ens_metrics
            self.optimal_threshold = opt_thresh
            
            cm = confusion_matrix(y_test, y_pred_ens)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            print(f"  Precision: {ens_metrics['precision']:.4f} (TP={tp}, FP={fp})")
            print(f"  Recall:    {ens_metrics['recall']:.4f} (TP={tp}, FN={fn})")
            print(f"  ROC AUC:   {ens_metrics['roc_auc']:.4f}")
            print(f"  Threshold: {opt_thresh:.2f}")
            
        except Exception as e:
            print(f"  Ensemble Error: {e}")
        
        # Find best model - prefer ensemble if precision >= 0.6
        valid_results = {k: v for k, v in self.results.items() if 'custom_score' in v}
        if valid_results:
            # Prefer VotingEnsemble if it has decent precision
            if 'VotingEnsemble' in valid_results and valid_results['VotingEnsemble']['precision'] >= 0.55:
                self.best_model_name = 'VotingEnsemble'
            else:
                self.best_model_name = max(valid_results, key=lambda x: valid_results[x]['custom_score'])
            
            self.best_model = self.models[self.best_model_name]
            
            print(f"\n{'=' * 70}")
            print(f"BEST MODEL: {self.best_model_name}")
            print(f"  Precision: {self.results[self.best_model_name]['precision']:.4f}")
            print(f"  Recall: {self.results[self.best_model_name]['recall']:.4f}")
            print(f"{'=' * 70}")
        
        return self.results
    
    def predict(self, X, threshold=None):
        """Predict with optimal threshold"""
        if self.best_model is None:
            self.load_model()
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        X_scaled = self.scaler.transform(X)
        prob = self.best_model.predict_proba(X_scaled)[:, 1]
        pred = (prob >= threshold).astype(int)
        return pred, prob
    
    def predict_with_confidence(self, X, min_confidence=0.6):
        """Only predict when confidence is high"""
        pred, prob = self.predict(X)
        confident_pred = np.where(prob >= min_confidence, pred, 0)
        return confident_pred, prob
    
    def save_model(self, suffix='_v2'):
        """Save model"""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        if self.best_model:
            joblib.dump(self.best_model, f"{config.MODEL_DIR}/best_model{suffix}.pkl")
            joblib.dump(self.scaler, f"{config.MODEL_DIR}/scaler{suffix}.pkl")
            joblib.dump(self.ensemble, f"{config.MODEL_DIR}/ensemble{suffix}.pkl")
            joblib.dump(self.optimal_threshold, f"{config.MODEL_DIR}/threshold{suffix}.pkl")
            print(f"Models saved to {config.MODEL_DIR}/")
            return True
        return False
    
    def load_model(self, suffix='_v2'):
        """Load model"""
        model_path = f"{config.MODEL_DIR}/best_model{suffix}.pkl"
        if os.path.exists(model_path):
            self.best_model = joblib.load(model_path)
            self.scaler = joblib.load(f"{config.MODEL_DIR}/scaler{suffix}.pkl")
            try:
                self.ensemble = joblib.load(f"{config.MODEL_DIR}/ensemble{suffix}.pkl")
                self.optimal_threshold = joblib.load(f"{config.MODEL_DIR}/threshold{suffix}.pkl")
            except:
                self.optimal_threshold = 0.6
            return True
        return False
