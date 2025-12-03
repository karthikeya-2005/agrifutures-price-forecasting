"""
Monitoring and Logging System for Price Predictions
Tracks predictions, errors, and model performance
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PredictionMonitor:
    def __init__(self, log_dir='logs/monitoring'):
        """Initialize monitoring system"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('prediction_monitor')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / 'predictions.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # In-memory tracking
        self.prediction_history = []
        self.error_count = defaultdict(int)
        self.model_usage = defaultdict(int)
        
    def log_prediction(self, prediction_result, actual_price=None):
        """Log a prediction"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'state': prediction_result.get('state'),
            'district': prediction_result.get('district'),
            'crop': prediction_result.get('crop'),
            'date': prediction_result.get('date'),
            'predicted_price': prediction_result.get('prediction'),
            'model': prediction_result.get('model'),
            'actual_price': actual_price,
            'error': prediction_result.get('error'),
            'features_used': prediction_result.get('features_used', 0)
        }
        
        # Calculate error if actual price available
        if actual_price is not None and prediction_result.get('prediction') is not None:
            error = abs(prediction_result['prediction'] - actual_price)
            error_pct = (error / actual_price * 100) if actual_price > 0 else 0
            log_entry['absolute_error'] = error
            log_entry['percentage_error'] = error_pct
        
        # Store in memory
        self.prediction_history.append(log_entry)
        
        # Log to file
        self.logger.info(json.dumps(log_entry))
        
        # Track model usage
        if prediction_result.get('model'):
            self.model_usage[prediction_result['model']] += 1
        
        # Track errors
        if prediction_result.get('error'):
            self.error_count[prediction_result['error']] += 1
        
        return log_entry
    
    def get_statistics(self, days=30):
        """Get statistics for the last N days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        recent_predictions = [
            p for p in self.prediction_history
            if pd.to_datetime(p['timestamp']) >= cutoff_date
        ]
        
        if not recent_predictions:
            return {
                'total_predictions': 0,
                'error_rate': 0,
                'average_error': None,
                'model_usage': {}
            }
        
        # Calculate statistics
        total = len(recent_predictions)
        errors = sum(1 for p in recent_predictions if p.get('error'))
        
        # Average error for predictions with actual prices
        predictions_with_actual = [
            p for p in recent_predictions 
            if p.get('actual_price') is not None and p.get('predicted_price') is not None
        ]
        
        avg_error = None
        if predictions_with_actual:
            errors_list = [p.get('absolute_error', 0) for p in predictions_with_actual]
            avg_error = np.mean(errors_list) if errors_list else None
        
        return {
            'total_predictions': total,
            'error_rate': (errors / total * 100) if total > 0 else 0,
            'average_error': avg_error,
            'model_usage': dict(self.model_usage),
            'error_breakdown': dict(self.error_count),
            'predictions_with_actual': len(predictions_with_actual)
        }
    
    def save_daily_report(self):
        """Save daily monitoring report"""
        stats = self.get_statistics(days=1)
        
        report = {
            'date': datetime.now().isoformat(),
            'statistics': stats,
            'total_predictions_all_time': len(self.prediction_history)
        }
        
        report_file = self.log_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def export_history(self, output_file='logs/monitoring/prediction_history.csv'):
        """Export prediction history to CSV"""
        if not self.prediction_history:
            return None
        
        df = pd.DataFrame(self.prediction_history)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def get_model_performance(self, model_name=None):
        """Get performance metrics for a specific model or all models"""
        predictions_with_actual = [
            p for p in self.prediction_history
            if p.get('actual_price') is not None and p.get('predicted_price') is not None
        ]
        
        if model_name:
            predictions_with_actual = [
                p for p in predictions_with_actual
                if p.get('model') == model_name
            ]
        
        if not predictions_with_actual:
            return None
        
        errors = [p.get('absolute_error', 0) for p in predictions_with_actual]
        pct_errors = [p.get('percentage_error', 0) for p in predictions_with_actual]
        
        return {
            'model': model_name or 'all',
            'total_predictions': len(predictions_with_actual),
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])),
            'mape': np.mean(pct_errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors)
        }

# Global monitor instance
_monitor = None

def get_monitor():
    """Get or create global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor

def log_prediction(prediction_result, actual_price=None):
    """Convenience function to log prediction"""
    monitor = get_monitor()
    return monitor.log_prediction(prediction_result, actual_price)

if __name__ == "__main__":
    # Test monitoring system
    monitor = PredictionMonitor()
    
    # Simulate some predictions
    test_predictions = [
        {'state': 'Maharashtra', 'district': 'Pune', 'crop': 'Wheat', 
         'date': '2025-12-01', 'prediction': 2500.0, 'model': 'lightgbm'},
        {'state': 'Maharashtra', 'district': 'Pune', 'crop': 'Wheat',
         'date': '2025-12-02', 'prediction': 2550.0, 'model': 'lightgbm'},
    ]
    
    for pred in test_predictions:
        monitor.log_prediction(pred, actual_price=pred['prediction'] + np.random.normal(0, 50))
    
    # Get statistics
    stats = monitor.get_statistics()
    print("\nMonitoring Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save report
    report = monitor.save_daily_report()
    print(f"\nDaily report saved")

