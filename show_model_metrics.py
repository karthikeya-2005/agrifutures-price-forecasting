"""
Display comprehensive model testing status and metrics
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def format_metric(value, metric_type='price'):
    """Format metric value for display"""
    if metric_type == 'price':
        return f"₹{value:,.2f}"
    elif metric_type == 'percentage':
        return f"{value:.2f}%"
    elif metric_type == 'ratio':
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"

def print_section(title, char="=", width=80):
    """Print a formatted section header"""
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

def print_model_metrics(model_name, metrics, indent="  "):
    """Print metrics for a single model"""
    print(f"\n{indent}📊 {model_name.upper().replace('_', ' ')}")
    print(f"{indent}{'-'*70}")
    
    # Validation metrics
    print(f"{indent}Validation Set:")
    print(f"{indent}  MAE:  {format_metric(metrics.get('val_mae', 0))}")
    print(f"{indent}  RMSE: {format_metric(metrics.get('val_rmse', 0))}")
    print(f"{indent}  MAPE: {format_metric(metrics.get('val_mape', 0), 'percentage')}")
    print(f"{indent}  R²:   {format_metric(metrics.get('val_r2', 0), 'ratio')}")
    
    # Test metrics
    print(f"{indent}Test Set:")
    print(f"{indent}  MAE:  {format_metric(metrics.get('test_mae', 0))}")
    print(f"{indent}  RMSE: {format_metric(metrics.get('test_rmse', 0))}")
    print(f"{indent}  MAPE: {format_metric(metrics.get('test_mape', 0), 'percentage')}")
    print(f"{indent}  R²:   {format_metric(metrics.get('test_r2', 0), 'ratio')}")

def main():
    print_section("MODEL TESTING STATUS AND METRICS", "=", 80)
    
    # Load model metadata
    metadata_file = Path('models/with_weather/metadata.json')
    if not metadata_file.exists():
        print("❌ Model metadata not found!")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Model Information
    print_section("MODEL INFORMATION", "-", 80)
    print(f"  Training Date: {metadata.get('training_date', 'Unknown')}")
    print(f"  Data Source: {metadata.get('data_source', 'Unknown')}")
    print(f"  Total Training Records: {metadata.get('total_records', 0):,}")
    print(f"  Weather Coverage: {metadata.get('weather_coverage_pct', 0)*100:.2f}%")
    print(f"  Number of Features: {len(metadata.get('feature_names', []))}")
    print(f"  Weather Features: {len(metadata.get('weather_features', []))}")
    
    # Model Results
    print_section("MODEL PERFORMANCE METRICS", "-", 80)
    results = metadata.get('results', {})
    
    # Baseline Models
    print("\n📈 BASELINE MODELS (Reference)")
    baseline_models = ['historical_average', 'last_value', 'moving_average_30', 'seasonal_naive']
    for model_name in baseline_models:
        if model_name in results:
            print_model_metrics(model_name, results[model_name])
    
    # Machine Learning Models
    print("\n🤖 MACHINE LEARNING MODELS")
    ml_models = ['xgboost', 'lightgbm']
    for model_name in ml_models:
        if model_name in results:
            print_model_metrics(model_name, results[model_name])
    
    # Neural Network Models
    print("\n🧠 NEURAL NETWORK MODELS")
    nn_models = ['feedforward', 'lstm', 'gru']
    for model_name in nn_models:
        if model_name in results:
            print_model_metrics(model_name, results[model_name])
    
    # Ensemble Models
    print("\n🎯 ENSEMBLE MODELS (Best Performance)")
    ensemble_models = ['ensemble_avg', 'ensemble_weighted']
    for model_name in ensemble_models:
        if model_name in results:
            print_model_metrics(model_name, results[model_name])
    
    # Summary Statistics
    print_section("PERFORMANCE SUMMARY", "-", 80)
    
    # Find best model by test R²
    best_model = None
    best_r2 = -float('inf')
    for model_name, metrics in results.items():
        test_r2 = metrics.get('test_r2', -float('inf'))
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model = model_name
    
    if best_model:
        print(f"\n  🏆 Best Model (by R²): {best_model.replace('_', ' ').title()}")
        best_metrics = results[best_model]
        print(f"     Test R²: {format_metric(best_metrics.get('test_r2', 0), 'ratio')}")
        print(f"     Test MAE: {format_metric(best_metrics.get('test_mae', 0))}")
        print(f"     Test MAPE: {format_metric(best_metrics.get('test_mape', 0), 'percentage')}")
    
    # Find model with lowest MAE
    best_mae_model = None
    best_mae = float('inf')
    for model_name, metrics in results.items():
        test_mae = metrics.get('test_mae', float('inf'))
        if test_mae < best_mae:
            best_mae = test_mae
            best_mae_model = model_name
    
    if best_mae_model:
        print(f"\n  🎯 Lowest MAE: {best_mae_model.replace('_', ' ').title()}")
        best_mae_metrics = results[best_mae_model]
        print(f"     Test MAE: {format_metric(best_mae_metrics.get('test_mae', 0))}")
        print(f"     Test R²: {format_metric(best_mae_metrics.get('test_r2', 0), 'ratio')}")
    
    # Model Comparison Table
    print_section("QUICK COMPARISON TABLE", "-", 80)
    print(f"\n{'Model':<25} {'Test MAE':<15} {'Test RMSE':<15} {'Test MAPE':<15} {'Test R²':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        if 'test_mae' in metrics:
            name = model_name.replace('_', ' ').title()
            mae = format_metric(metrics.get('test_mae', 0))
            rmse = format_metric(metrics.get('test_rmse', 0))
            mape = format_metric(metrics.get('test_mape', 0), 'percentage')
            r2 = format_metric(metrics.get('test_r2', 0), 'ratio')
            print(f"{name:<25} {mae:<15} {rmse:<15} {mape:<15} {r2:<10}")
    
    # Important Notes
    print_section("IMPORTANT NOTES", "-", 80)
    print("""
  ⚠️  CRITICAL ISSUE IDENTIFIED:
  
  The model shows excellent performance metrics (R² = 0.89-0.98, MAE = ₹53-90)
  on the test set, BUT these metrics are misleading because:
  
  1. Test set likely has same price range issues as training data
  2. Model was trained on historical data with much lower prices
  3. Current market prices (₹4,000-5,000) are 3-4x higher than predictions (₹1,450)
  
  📊 METRIC INTERPRETATION:
  
  • MAE (Mean Absolute Error): Average prediction error in ₹
    - Low MAE (₹53-90) suggests good fit to training data
    - BUT training data prices were much lower than current market
  
  • R² (Coefficient of Determination): Proportion of variance explained
    - High R² (0.89-0.98) suggests model fits training data well
    - BUT doesn't account for price scale mismatch with current market
  
  • MAPE (Mean Absolute Percentage Error): Average error as percentage
    - Low MAPE (2-3%) suggests good relative accuracy
    - BUT only accurate relative to training data price range
  
  ✅ CALIBRATION STATUS:
  
  A calibration mechanism has been implemented to adjust predictions
  based on current market prices. This addresses the price scale mismatch.
  
  🔄 RECOMMENDATION:
  
  Model should be retrained with recent data that includes current
  market price ranges for accurate predictions.
    """)
    
    print("\n" + "="*80)
    print("  Report generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

