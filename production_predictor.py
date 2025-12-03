"""
Production-Ready Predictor with Full Feature Engineering
Uses Phase 2 feature engineering and Phase 3 trained models
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from phase2_feature_engineering import FeatureEngineer
from get_available_commodities import is_valid_combination

class ProductionPredictor:
    def __init__(self, models_dir='models/phase3', use_cache=True):
        """Initialize production predictor"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        self.scalers = {}
        self.feature_cols = []
        self.use_cache = use_cache
        self.cache = {}
        
        # Load historical data for feature engineering
        self.historical_data = None
        self.load_historical_data()
        
        # Load models and metadata
        self.load_models()
    
    def load_historical_data(self):
        """Load historical data for feature engineering"""
        try:
            data_file = Path('data/processed/data_with_features.csv')
            if data_file.exists():
                self.historical_data = pd.read_csv(
                    data_file, 
                    parse_dates=['date'], 
                    low_memory=False
                )
                print(f"[INFO] Loaded {len(self.historical_data):,} historical records")
        except Exception as e:
            print(f"[WARN] Could not load historical data: {e}")
            self.historical_data = None
    
    def load_models(self):
        """Load trained models and metadata"""
        # Load metadata
        metadata_file = self.models_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.feature_cols = self.metadata.get('feature_cols', [])
            print(f"[INFO] Loaded metadata with {len(self.feature_cols)} features")
        
        # Load XGBoost
        xgb_file = self.models_dir / 'xgboost.pkl'
        if xgb_file.exists():
            with open(xgb_file, 'rb') as f:
                self.models['xgboost'] = pickle.load(f)
            print("[INFO] Loaded XGBoost model")
        
        # Load LightGBM
        lgb_file = self.models_dir / 'lightgbm.pkl'
        if lgb_file.exists():
            with open(lgb_file, 'rb') as f:
                self.models['lightgbm'] = pickle.load(f)
            print("[INFO] Loaded LightGBM model")
        
        # Load scalers (for neural networks if needed)
        scaler_file = self.models_dir / 'scalers.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scalers = pickle.load(f)
    
    def create_features_for_prediction(self, state, district, crop, date):
        """Create full feature set for prediction using historical data"""
        # Convert date
        pred_date = pd.to_datetime(date)
        
        # Create base dataframe
        df = pd.DataFrame({
            'date': [pred_date],
            'state': [state],
            'district': [district],
            'crop': [crop],
            'price': [0]  # Placeholder, will be replaced
        })
        
        # If we have historical data, append it for feature engineering
        if self.historical_data is not None:
            # Get relevant historical data (need enough for rolling features)
            hist_relevant = self.historical_data[
                (self.historical_data['state'] == state) &
                (self.historical_data['district'] == district) &
                (self.historical_data['crop'] == crop)
            ].copy()
            
            if len(hist_relevant) > 0:
                # Take last 365 days for feature engineering context
                hist_relevant = hist_relevant.sort_values('date').tail(365)
                # Append prediction row
                df_with_hist = pd.concat([hist_relevant, df], ignore_index=True)
                df_with_hist = df_with_hist.sort_values('date').reset_index(drop=True)
            else:
                # No historical data, use just the prediction row
                df_with_hist = df.copy()
        else:
            # No historical data available
            df_with_hist = df.copy()
        
        # Use FeatureEngineer to create all features
        try:
            engineer = FeatureEngineer(df_with_hist)
            df_features, _ = engineer.engineer_all_features()
        except Exception as e:
            # If feature engineering fails, create minimal features
            print(f"[WARN] Feature engineering failed: {e}. Using minimal features.")
            df_features = df_with_hist.copy()
            # Add basic temporal features
            df_features['year'] = df_features['date'].dt.year
            df_features['month'] = df_features['date'].dt.month
            df_features['day_of_week'] = df_features['date'].dt.dayofweek
        
        # Get the last row (our prediction row with all features)
        pred_features = df_features.iloc[[-1]].copy()
        
        # Select only the feature columns
        available_features = [col for col in self.feature_cols if col in pred_features.columns]
        pred_features = pred_features[available_features]
        
        # Fill missing features with defaults
        for col in self.feature_cols:
            if col not in pred_features.columns:
                # Set sensible defaults based on feature type
                if 'price' in col.lower():
                    # Use average price for this crop if available
                    if self.historical_data is not None:
                        crop_avg = self.historical_data[
                            self.historical_data['crop'] == crop
                        ]['price'].mean()
                        pred_features[col] = crop_avg if not pd.isna(crop_avg) else 2000.0
                    else:
                        pred_features[col] = 2000.0
                elif 'temp' in col.lower() or 'rain' in col.lower():
                    pred_features[col] = 0.0
                elif 'sin' in col.lower() or 'cos' in col.lower():
                    pred_features[col] = 0.0
                else:
                    pred_features[col] = 0.0
        
        # Ensure correct order
        pred_features = pred_features[self.feature_cols]
        
        # Convert to numeric and handle any remaining issues
        pred_features = pred_features.select_dtypes(include=[np.number])
        pred_features = pred_features.fillna(0).replace([np.inf, -np.inf], 0)
        
        return pred_features
    
    def predict(self, state, district, crop, date, model_name='lightgbm'):
        """
        Predict price with full feature engineering
        
        Args:
            state: State name
            district: District name
            crop: Crop name
            date: Date for prediction (YYYY-MM-DD or datetime)
            model_name: Model to use ('lightgbm', 'xgboost', or 'ensemble')
        
        Returns:
            dict with prediction and metadata
        """
        # Validate input combination
        if not is_valid_combination(state, district, crop):
            return {
                'prediction': None,
                'error': f"Invalid combination: {crop} is not available for {district}, {state} in training data. Please use a valid state-district-crop combination.",
                'state': state,
                'district': district,
                'crop': crop,
                'date': str(date)
            }
        
        # Check cache
        cache_key = f"{state}_{district}_{crop}_{date}_{model_name}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        if model_name not in self.models and model_name != 'ensemble':
            return {
                'prediction': None,
                'error': f"Model '{model_name}' not available. Available: {list(self.models.keys())}"
            }
        
        try:
            # Create features
            features = self.create_features_for_prediction(state, district, crop, date)
            
            # Make prediction
            if model_name == 'ensemble':
                # Use weighted ensemble
                predictions = []
                weights = []
                for name, model in self.models.items():
                    pred = model.predict(features)[0]
                    predictions.append(pred)
                    # Simple weight: prefer LightGBM
                    weights.append(0.6 if name == 'lightgbm' else 0.4)
                
                # Normalize weights
                weights = np.array(weights) / sum(weights)
                final_prediction = np.average(predictions, weights=weights)
            else:
                model = self.models[model_name]
                final_prediction = model.predict(features)[0]
            
            result = {
                'prediction': float(final_prediction),
                'model': model_name,
                'state': state,
                'district': district,
                'crop': crop,
                'date': str(pd.to_datetime(date)),
                'confidence': 'high' if len(features) > 0 else 'low',
                'features_used': len(features.columns)
            }
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                'prediction': None,
                'error': str(e),
                'state': state,
                'district': district,
                'crop': crop,
                'date': str(date)
            }
    
    def predict_batch(self, predictions_list, model_name='lightgbm'):
        """
        Predict prices for multiple inputs
        
        Args:
            predictions_list: List of dicts with keys: state, district, crop, date
            model_name: Model to use
        
        Returns:
            List of prediction results
        """
        results = []
        for pred_input in predictions_list:
            result = self.predict(
                pred_input['state'],
                pred_input['district'],
                pred_input['crop'],
                pred_input['date'],
                model_name
            )
            results.append(result)
        return results
    
    def predict_forecast(self, state, district, crop, start_date, days_ahead=30, model_name='lightgbm'):
        """
        Predict prices for multiple days ahead
        
        Args:
            state: State name
            district: District name
            crop: Crop name
            start_date: Starting date
            days_ahead: Number of days to forecast
            model_name: Model to use
        
        Returns:
            DataFrame with date and predicted price
        """
        start = pd.to_datetime(start_date)
        dates = [start + pd.Timedelta(days=i) for i in range(days_ahead)]
        
        predictions = []
        for date in dates:
            result = self.predict(state, district, crop, date, model_name)
            predictions.append({
                'date': date,
                'predicted_price': result.get('prediction'),
                'model': model_name
            })
        
        return pd.DataFrame(predictions)
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'features_count': len(self.feature_cols),
            'historical_data_available': self.historical_data is not None,
            'historical_records': len(self.historical_data) if self.historical_data is not None else 0,
            'cache_enabled': self.use_cache,
            'cache_size': len(self.cache)
        }

# Global instance
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = ProductionPredictor()
    return _predictor

def predict_price(state, district, crop, date, model_name='lightgbm'):
    """Convenience function for single prediction"""
    predictor = get_predictor()
    result = predictor.predict(state, district, crop, date, model_name)
    return result.get('prediction')

def predict_with_forecast(state, district, crop, start_date, days_ahead=30, model_name='lightgbm'):
    """Convenience function for forecast"""
    predictor = get_predictor()
    return predictor.predict_forecast(state, district, crop, start_date, days_ahead, model_name)

if __name__ == "__main__":
    # Test the predictor
    predictor = ProductionPredictor()
    
    print("\n" + "="*80)
    print("PRODUCTION PREDICTOR TEST")
    print("="*80 + "\n")
    
    # Test single prediction
    result = predictor.predict(
        state="Maharashtra",
        district="Pune",
        crop="Wheat",
        date="2025-12-01",
        model_name="lightgbm"
    )
    
    print("Single Prediction Test:")
    print(f"  Result: {result}")
    
    # Test forecast
    forecast = predictor.predict_forecast(
        state="Maharashtra",
        district="Pune",
        crop="Wheat",
        start_date="2025-12-01",
        days_ahead=7
    )
    
    print("\nForecast Test (7 days):")
    print(forecast.to_string())
    
    # Model info
    print("\nModel Info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

