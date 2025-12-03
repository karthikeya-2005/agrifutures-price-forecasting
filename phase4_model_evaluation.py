"""Phase 4: Model Evaluation - Metrics, Validation, Error Analysis, Interpretability"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance plots will be limited.")

class ModelEvaluator:
    def __init__(self, models_dir='models/phase3'):
        """Initialize evaluator with models directory"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        self.results = {}
        
    def load_models(self):
        """Load trained models and metadata"""
        print("="*80)
        print("PHASE 4: MODEL EVALUATION")
        print("="*80 + "\n")
        
        print("Loading models...")
        
        # Load metadata
        metadata_file = self.models_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"  [OK] Loaded metadata")
        
        # Load tree models
        for model_name in ['xgboost', 'lightgbm']:
            model_file = self.models_dir / f'{model_name}.pkl'
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"  [OK] Loaded {model_name}")
        
        # Load scalers
        scaler_file = self.models_dir / 'scalers.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"  [OK] Loaded scalers")
        
        print(f"\nLoaded {len(self.models)} models")
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mean_error = np.mean(y_pred - y_true)
        median_error = np.median(y_pred - y_true)
        std_error = np.std(y_pred - y_true)
        
        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'mean_error': float(mean_error),
            'median_error': float(median_error),
            'std_error': float(std_error),
            'direction_accuracy': float(direction_accuracy)
        }
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all loaded models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80 + "\n")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            y_pred = model.predict(X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            evaluation_results[model_name] = metrics
            
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
        
        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    def error_analysis(self, X_test, y_test, model_name='xgboost'):
        """Perform detailed error analysis"""
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80 + "\n")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        errors = y_pred - y_test
        abs_errors = np.abs(errors)
        
        # Error statistics
        print("Error Statistics:")
        print(f"  Mean Error: {np.mean(errors):.2f}")
        print(f"  Median Error: {np.median(errors):.2f}")
        print(f"  Std Error: {np.std(errors):.2f}")
        print(f"  Max Over-prediction: {np.max(errors):.2f}")
        print(f"  Max Under-prediction: {np.min(errors):.2f}")
        
        # Error by price range
        print("\nError by Price Range:")
        price_ranges = [
            (0, 100, 'Very Low'),
            (100, 500, 'Low'),
            (500, 2000, 'Medium'),
            (2000, 10000, 'High'),
            (10000, float('inf'), 'Very High')
        ]
        
        error_by_range = {}
        for low, high, label in price_ranges:
            mask = (y_test >= low) & (y_test < high)
            if mask.sum() > 0:
                range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                error_by_range[label] = {
                    'mae': float(range_mae),
                    'count': int(mask.sum())
                }
                print(f"  {label} (Rs. {low}-{high}): MAE = {range_mae:.2f}, Count = {mask.sum()}")
        
        # Error distribution
        print("\nError Distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(abs_errors, p):.2f}")
        
        self.results['error_analysis'] = {
            'error_stats': {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'max': float(np.max(errors)),
                'min': float(np.min(errors))
            },
            'error_by_range': error_by_range,
            'error_percentiles': {str(p): float(np.percentile(abs_errors, p)) for p in percentiles}
        }
        
        return self.results['error_analysis']
    
    def feature_importance_analysis(self, model_name='xgboost'):
        """Analyze feature importance"""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80 + "\n")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        feature_cols = self.metadata.get('feature_cols', [])
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            print("  Model does not support feature importance")
            return
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_cols[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))
        
        # Save importance
        importance_file = self.models_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_file, index=False)
        print(f"\n  [OK] Saved feature importance to: {importance_file}")
        
        # Plot importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_file = self.models_dir / 'feature_importance_plot.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved feature importance plot to: {plot_file}")
        
        self.results['feature_importance'] = importance_df.to_dict('records')
        return importance_df
    
    def shap_analysis(self, X_test, model_name='xgboost', sample_size=1000):
        """SHAP analysis for model interpretability"""
        if not SHAP_AVAILABLE:
            print("\nSHAP not available. Skipping SHAP analysis.")
            return
        
        print("\n" + "="*80)
        print("SHAP ANALYSIS (Model Interpretability)")
        print("="*80 + "\n")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        # Sample data for faster computation
        if len(X_test) > sample_size:
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test
        
        print(f"Computing SHAP values for {len(X_sample)} samples...")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            
            shap_file = self.models_dir / 'shap_summary.png'
            plt.savefig(shap_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved SHAP summary plot to: {shap_file}")
            
            # Feature importance from SHAP
            shap_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            print("\nTop 10 Features by SHAP Importance:")
            print(shap_importance.head(10).to_string(index=False))
            
            self.results['shap_analysis'] = {
                'shap_importance': shap_importance.to_dict('records')
            }
            
        except Exception as e:
            print(f"  Error in SHAP analysis: {e}")
    
    def validation_strategies(self, X, y):
        """Implement different validation strategies"""
        print("\n" + "="*80)
        print("VALIDATION STRATEGIES")
        print("="*80 + "\n")
        
        from sklearn.model_selection import KFold, TimeSeriesSplit
        
        validation_results = {}
        
        # 1. K-Fold Cross-Validation
        print("1. K-Fold Cross-Validation (k=5)...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model (using XGBoost as example)
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred_fold = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred_fold)
            kfold_scores.append(r2)
        
        validation_results['kfold'] = {
            'mean_r2': float(np.mean(kfold_scores)),
            'std_r2': float(np.std(kfold_scores)),
            'scores': [float(s) for s in kfold_scores]
        }
        print(f"  Mean R²: {np.mean(kfold_scores):.4f} ± {np.std(kfold_scores):.4f}")
        
        # 2. Time Series Cross-Validation
        print("\n2. Time Series Cross-Validation (n_splits=5)...")
        tscv = TimeSeriesSplit(n_splits=5)
        tscv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred_fold = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred_fold)
            tscv_scores.append(r2)
        
        validation_results['timeseries_cv'] = {
            'mean_r2': float(np.mean(tscv_scores)),
            'std_r2': float(np.std(tscv_scores)),
            'scores': [float(s) for s in tscv_scores]
        }
        print(f"  Mean R²: {np.mean(tscv_scores):.4f} ± {np.std(tscv_scores):.4f}")
        
        self.results['validation'] = validation_results
        return validation_results
    
    def generate_evaluation_report(self, output_file='data/evaluation/evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': list(self.models.keys()),
            'results': self.results,
            'metadata': self.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[SUCCESS] Evaluation report saved to: {output_path}")
        return report
    
    def run_full_evaluation(self, X_test, y_test, X_full=None, y_full=None):
        """Run complete evaluation pipeline"""
        self.load_models()
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Error analysis
        self.error_analysis(X_test, y_test)
        
        # Feature importance
        self.feature_importance_analysis()
        
        # SHAP analysis
        if X_test is not None:
            self.shap_analysis(X_test)
        
        # Validation strategies
        if X_full is not None and y_full is not None:
            self.validation_strategies(X_full, y_full)
        
        # Generate report
        self.generate_evaluation_report()
        
        print("\n" + "="*80)
        print("PHASE 4 COMPLETE: MODEL EVALUATION")
        print("="*80)
        print("\nNext: Phase 5 - System Integration")
        
        return self.results

if __name__ == "__main__":
    # Load test data
    data_file = Path('data/processed/data_with_features.csv')
    if not data_file.exists():
        print(f"[ERROR] Feature-engineered data not found: {data_file}")
        exit(1)
    
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
    print(f"Loaded {len(df):,} records\n")
    
    # Load feature list
    feature_file = Path('data/processed/feature_list.txt')
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['price']]
    
    # Prepare test data (last 20% by date)
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    X_test = test_df[feature_cols].select_dtypes(include=[np.number])
    X_test = X_test.fillna(X_test.mean())
    y_test = test_df['price']
    
    # Align columns with model
    metadata_file = Path('models/phase3/metadata.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        model_features = metadata.get('feature_cols', feature_cols)
        X_test = X_test[[col for col in model_features if col in X_test.columns]]
    
    # Run evaluation
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation(X_test, y_test, 
                                            X_full=df[feature_cols].select_dtypes(include=[np.number]),
                                            y_full=df['price'])

