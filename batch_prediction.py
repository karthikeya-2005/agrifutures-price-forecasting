"""
Batch Prediction System
Process multiple predictions efficiently
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from production_predictor import ProductionPredictor
from monitoring_system import get_monitor
import json
import warnings
warnings.filterwarnings('ignore')

class BatchPredictor:
    def __init__(self, predictor=None):
        """Initialize batch predictor"""
        self.predictor = predictor or ProductionPredictor()
        self.monitor = get_monitor()
    
    def predict_from_dataframe(self, df, model_name='lightgbm', date_column='date'):
        """
        Predict prices from a DataFrame
        
        Args:
            df: DataFrame with columns: state, district, crop, date (or date_column)
            model_name: Model to use
            date_column: Name of date column
        
        Returns:
            DataFrame with predictions added
        """
        results = []
        
        for idx, row in df.iterrows():
            result = self.predictor.predict(
                state=row['state'],
                district=row['district'],
                crop=row['crop'],
                date=row[date_column],
                model_name=model_name
            )
            
            # Log prediction
            self.monitor.log_prediction(result)
            
            # Add to results
            results.append({
                'index': idx,
                'predicted_price': result.get('prediction'),
                'model': result.get('model'),
                'error': result.get('error'),
                'confidence': result.get('confidence')
            })
        
        # Merge results back to original dataframe
        results_df = pd.DataFrame(results)
        df_with_predictions = df.copy()
        df_with_predictions['predicted_price'] = results_df['predicted_price'].values
        df_with_predictions['prediction_model'] = results_df['model'].values
        df_with_predictions['prediction_error'] = results_df['error'].values
        df_with_predictions['prediction_confidence'] = results_df['confidence'].values
        
        return df_with_predictions
    
    def predict_for_commodity(self, crop, start_date, end_date, 
                             states=None, districts=None, model_name='lightgbm'):
        """
        Predict prices for a commodity across locations and dates
        
        Args:
            crop: Crop name
            start_date: Start date
            end_date: End date
            states: List of states (None for all)
            districts: List of districts (None for all)
            model_name: Model to use
        
        Returns:
            DataFrame with predictions
        """
        # Load historical data to get available locations
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
            
            # Filter by crop
            crop_data = df[df['crop'] == crop].copy()
            
            # Get unique locations
            if states:
                crop_data = crop_data[crop_data['state'].isin(states)]
            if districts:
                crop_data = crop_data[crop_data['district'].isin(districts)]
            
            locations = crop_data[['state', 'district']].drop_duplicates()
        else:
            # Fallback: use default locations
            locations = pd.DataFrame({
                'state': ['Maharashtra', 'Uttar Pradesh', 'Punjab'],
                'district': ['Pune', 'Lucknow', 'Amritsar']
            })
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create prediction grid
        predictions = []
        for _, loc in locations.iterrows():
            for date in dates:
                result = self.predictor.predict(
                    state=loc['state'],
                    district=loc['district'],
                    crop=crop,
                    date=date,
                    model_name=model_name
                )
                
                self.monitor.log_prediction(result)
                
                predictions.append({
                    'state': loc['state'],
                    'district': loc['district'],
                    'crop': crop,
                    'date': date,
                    'predicted_price': result.get('prediction'),
                    'model': result.get('model'),
                    'confidence': result.get('confidence')
                })
        
        return pd.DataFrame(predictions)
    
    def predict_for_location(self, state, district, start_date, end_date,
                            crops=None, model_name='lightgbm'):
        """
        Predict prices for a location across commodities and dates
        
        Args:
            state: State name
            district: District name
            start_date: Start date
            end_date: End date
            crops: List of crops (None for all)
            model_name: Model to use
        
        Returns:
            DataFrame with predictions
        """
        # Load historical data to get available crops
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
            
            # Filter by location
            loc_data = df[
                (df['state'] == state) & 
                (df['district'] == district)
            ].copy()
            
            # Get unique crops
            if crops:
                available_crops = [c for c in crops if c in loc_data['crop'].unique()]
            else:
                available_crops = loc_data['crop'].unique().tolist()
        else:
            # Fallback: use common crops
            available_crops = ['Wheat', 'Rice', 'Potato', 'Onion', 'Tomato']
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create prediction grid
        predictions = []
        for crop in available_crops:
            for date in dates:
                result = self.predictor.predict(
                    state=state,
                    district=district,
                    crop=crop,
                    date=date,
                    model_name=model_name
                )
                
                self.monitor.log_prediction(result)
                
                predictions.append({
                    'state': state,
                    'district': district,
                    'crop': crop,
                    'date': date,
                    'predicted_price': result.get('prediction'),
                    'model': result.get('model'),
                    'confidence': result.get('confidence')
                })
        
        return pd.DataFrame(predictions)
    
    def export_predictions(self, predictions_df, output_file):
        """Export predictions to CSV"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        return output_path

if __name__ == "__main__":
    # Test batch prediction
    batch = BatchPredictor()
    
    print("="*80)
    print("BATCH PREDICTION TEST")
    print("="*80 + "\n")
    
    # Test 1: Predict for commodity
    print("Test 1: Predicting for Wheat across locations...")
    wheat_predictions = batch.predict_for_commodity(
        crop='Wheat',
        start_date='2025-12-01',
        end_date='2025-12-07',
        states=['Maharashtra', 'Uttar Pradesh'],
        model_name='lightgbm'
    )
    print(f"Generated {len(wheat_predictions)} predictions")
    print(wheat_predictions.head())
    
    # Test 2: Predict for location
    print("\nTest 2: Predicting for Pune across crops...")
    pune_predictions = batch.predict_for_location(
        state='Maharashtra',
        district='Pune',
        start_date='2025-12-01',
        end_date='2025-12-03',
        crops=['Wheat', 'Rice'],
        model_name='lightgbm'
    )
    print(f"Generated {len(pune_predictions)} predictions")
    print(pune_predictions.head())
    
    # Export
    batch.export_predictions(wheat_predictions, 'data/predictions/wheat_batch_predictions.csv')
    print("\n[SUCCESS] Predictions exported")

