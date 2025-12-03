"""Phase 2: Advanced Feature Engineering"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df.copy()
        self.feature_names = []
        
    def create_temporal_features(self):
        """Create temporal features"""
        print("Creating temporal features...")
        
        # Basic date features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Cyclical encoding for temporal features
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # Seasonal indicators
        self.df['is_spring'] = self.df['month'].isin([3, 4, 5]).astype(int)
        self.df['is_summer'] = self.df['month'].isin([6, 7, 8]).astype(int)
        self.df['is_monsoon'] = self.df['month'].isin([6, 7, 8, 9]).astype(int)
        self.df['is_winter'] = self.df['month'].isin([12, 1, 2]).astype(int)
        self.df['is_harvest_season'] = self.df['month'].isin([10, 11, 12, 1, 2]).astype(int)
        
        # Indian crop seasons
        self.df['is_kharif'] = self.df['month'].isin([6, 7, 8, 9, 10]).astype(int)
        self.df['is_rabi'] = self.df['month'].isin([11, 12, 1, 2, 3]).astype(int)
        self.df['is_zaid'] = self.df['month'].isin([4, 5]).astype(int)
        
        # Weekend indicator
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        temporal_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                           'week_of_year', 'quarter', 'month_sin', 'month_cos',
                           'day_of_year_sin', 'day_of_year_cos', 'day_of_week_sin',
                           'day_of_week_cos', 'is_spring', 'is_summer', 'is_monsoon',
                           'is_winter', 'is_harvest_season', 'is_kharif', 'is_rabi',
                           'is_zaid', 'is_weekend']
        self.feature_names.extend(temporal_features)
        print(f"  [OK] Created {len(temporal_features)} temporal features")
        
    def create_price_features(self):
        """Create price-based features"""
        print("Creating price features...")
        
        # Sort by date for lag features
        self.df = self.df.sort_values(['crop', 'state', 'district', 'date']).reset_index(drop=True)
        
        # Price range features
        if 'min_price' in self.df.columns and 'max_price' in self.df.columns:
            self.df['price_range'] = self.df['max_price'] - self.df['min_price']
            self.df['price_range_ratio'] = np.where(
                self.df['min_price'] > 0,
                self.df['price_range'] / self.df['min_price'],
                0
            )
            self.df['price_volatility'] = self.df['price_range'] / (self.df['price'] + 1e-6)
        
        # Lag features (previous prices)
        for lag in [1, 7, 30, 90, 365]:
            self.df[f'price_lag_{lag}'] = self.df.groupby(['crop', 'state', 'district'])['price'].shift(lag)
            self.df[f'price_change_{lag}'] = self.df['price'] - self.df[f'price_lag_{lag}']
            self.df[f'price_change_pct_{lag}'] = np.where(
                self.df[f'price_lag_{lag}'] > 0,
                (self.df['price'] - self.df[f'price_lag_{lag}']) / self.df[f'price_lag_{lag}'] * 100,
                0
            )
        
        # Rolling statistics
        for window in [7, 30, 90]:
            self.df[f'price_ma_{window}'] = self.df.groupby(['crop', 'state', 'district'])['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'price_std_{window}'] = self.df.groupby(['crop', 'state', 'district'])['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            self.df[f'price_min_{window}'] = self.df.groupby(['crop', 'state', 'district'])['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            self.df[f'price_max_{window}'] = self.df.groupby(['crop', 'state', 'district'])['price'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            # Price relative to moving average
            self.df[f'price_vs_ma_{window}'] = (self.df['price'] - self.df[f'price_ma_{window}']) / (self.df[f'price_ma_{window}'] + 1e-6)
        
        # Year-over-year features
        self.df['price_yoy'] = self.df.groupby(['crop', 'state', 'district', 'month', 'day'])['price'].transform(
            lambda x: x - x.shift(1) if len(x) > 1 else 0
        )
        
        price_features = [col for col in self.df.columns if 'price' in col.lower() and col != 'price']
        self.feature_names.extend(price_features)
        print(f"  [OK] Created {len(price_features)} price features")
        
    def create_location_features(self):
        """Create location-based features"""
        print("Creating location features...")
        
        # State-level statistics
        state_stats = self.df.groupby('state')['price'].agg(['mean', 'std', 'median']).reset_index()
        state_stats.columns = ['state', 'state_avg_price', 'state_price_std', 'state_price_median']
        self.df = self.df.merge(state_stats, on='state', how='left')
        
        # District-level statistics
        district_stats = self.df.groupby(['state', 'district'])['price'].agg(['mean', 'std']).reset_index()
        district_stats.columns = ['state', 'district', 'district_avg_price', 'district_price_std']
        self.df = self.df.merge(district_stats, on=['state', 'district'], how='left')
        
        # Price relative to location averages
        self.df['price_vs_state_avg'] = (self.df['price'] - self.df['state_avg_price']) / (self.df['state_avg_price'] + 1e-6)
        self.df['price_vs_district_avg'] = (self.df['price'] - self.df['district_avg_price']) / (self.df['district_avg_price'] + 1e-6)
        
        # Location encoding (simple frequency encoding)
        state_counts = self.df['state'].value_counts()
        self.df['state_frequency'] = self.df['state'].map(state_counts)
        
        district_counts = self.df['district'].value_counts()
        self.df['district_frequency'] = self.df['district'].map(district_counts)
        
        location_features = ['state_avg_price', 'state_price_std', 'state_price_median',
                           'district_avg_price', 'district_price_std',
                           'price_vs_state_avg', 'price_vs_district_avg',
                           'state_frequency', 'district_frequency']
        self.feature_names.extend(location_features)
        print(f"  [OK] Created {len(location_features)} location features")
        
    def create_commodity_features(self):
        """Create commodity-based features"""
        print("Creating commodity features...")
        
        # Commodity-level statistics
        crop_stats = self.df.groupby('crop')['price'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
        crop_stats.columns = ['crop', 'crop_avg_price', 'crop_price_std', 'crop_price_median', 'crop_price_min', 'crop_price_max']
        self.df = self.df.merge(crop_stats, on='crop', how='left')
        
        # Price relative to commodity average
        self.df['price_vs_crop_avg'] = (self.df['price'] - self.df['crop_avg_price']) / (self.df['crop_avg_price'] + 1e-6)
        
        # Commodity frequency
        crop_counts = self.df['crop'].value_counts()
        self.df['crop_frequency'] = self.df['crop'].map(crop_counts)
        
        # Commodity price range
        self.df['crop_price_range'] = self.df['crop_price_max'] - self.df['crop_price_min']
        self.df['price_in_crop_range'] = (self.df['price'] - self.df['crop_price_min']) / (self.df['crop_price_range'] + 1e-6)
        
        # Commodity category (simple heuristic based on price)
        self.df['crop_category'] = pd.cut(
            self.df['crop_avg_price'],
            bins=[0, 50, 200, 1000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        commodity_features = ['crop_avg_price', 'crop_price_std', 'crop_price_median',
                             'crop_price_min', 'crop_price_max', 'price_vs_crop_avg',
                             'crop_frequency', 'crop_price_range', 'price_in_crop_range']
        self.feature_names.extend(commodity_features)
        print(f"  [OK] Created {len(commodity_features)} commodity features")
        
    def create_weather_features(self):
        """Create weather-based features"""
        print("Creating weather features...")
        
        weather_features = []
        
        if 'temperature' in self.df.columns:
            # Temperature features
            self.df['temp_ma_7'] = self.df.groupby(['state', 'district'])['temperature'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            self.df['temp_ma_30'] = self.df.groupby(['state', 'district'])['temperature'].transform(
                lambda x: x.rolling(window=30, min_periods=1).mean()
            )
            weather_features.extend(['temp_ma_7', 'temp_ma_30'])
        
        if 'rainfall' in self.df.columns:
            # Rainfall features
            self.df['rainfall_ma_7'] = self.df.groupby(['state', 'district'])['rainfall'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            self.df['rainfall_ma_30'] = self.df.groupby(['state', 'district'])['rainfall'].transform(
                lambda x: x.rolling(window=30, min_periods=1).mean()
            )
            self.df['rainfall_cumulative_30'] = self.df.groupby(['state', 'district'])['rainfall'].transform(
                lambda x: x.rolling(window=30, min_periods=1).sum()
            )
            weather_features.extend(['rainfall_ma_7', 'rainfall_ma_30', 'rainfall_cumulative_30'])
        
        # Fill missing weather with 0 or mean
        if 'temperature' in self.df.columns:
            self.df['temperature'] = self.df['temperature'].fillna(self.df['temperature'].mean())
        if 'rainfall' in self.df.columns:
            self.df['rainfall'] = self.df['rainfall'].fillna(0)
        
        self.feature_names.extend(weather_features)
        if weather_features:
            print(f"  [OK] Created {len(weather_features)} weather features")
        else:
            print("  [WARN] No weather data available")
        
    def create_interaction_features(self):
        """Create interaction features"""
        print("Creating interaction features...")
        
        # Location-commodity interactions
        location_crop_stats = self.df.groupby(['state', 'crop'])['price'].agg(['mean', 'std']).reset_index()
        location_crop_stats.columns = ['state', 'crop', 'state_crop_avg_price', 'state_crop_price_std']
        self.df = self.df.merge(location_crop_stats, on=['state', 'crop'], how='left')
        
        self.df['price_vs_state_crop_avg'] = (self.df['price'] - self.df['state_crop_avg_price']) / (self.df['state_crop_avg_price'] + 1e-6)
        
        # Temporal-location interactions
        monthly_state_stats = self.df.groupby(['state', 'month'])['price'].mean().reset_index()
        monthly_state_stats.columns = ['state', 'month', 'state_monthly_avg_price']
        self.df = self.df.merge(monthly_state_stats, on=['state', 'month'], how='left')
        
        # Temporal-commodity interactions
        monthly_crop_stats = self.df.groupby(['crop', 'month'])['price'].mean().reset_index()
        monthly_crop_stats.columns = ['crop', 'month', 'crop_monthly_avg_price']
        self.df = self.df.merge(monthly_crop_stats, on=['crop', 'month'], how='left')
        
        interaction_features = ['state_crop_avg_price', 'state_crop_price_std',
                               'price_vs_state_crop_avg', 'state_monthly_avg_price',
                               'crop_monthly_avg_price']
        self.feature_names.extend(interaction_features)
        print(f"  [OK] Created {len(interaction_features)} interaction features")
        
    def handle_missing_values(self):
        """Handle missing values in features"""
        print("Handling missing values...")
        
        # Fill numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'price':  # Don't fill target
                if self.df[col].isnull().sum() > 0:
                    # Use median for price-related, mean for others
                    if 'price' in col.lower():
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        print(f"  [OK] Handled missing values")
        
    def get_feature_list(self):
        """Get list of feature columns (excluding target and metadata)"""
        exclude_cols = ['price', 'date', 'state', 'district', 'crop', 
                        'data_source', 'kaggle_dataset', 'crop_category']
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols and col in self.feature_names]
        return feature_cols
    
    def engineer_all_features(self):
        """Run complete feature engineering pipeline"""
        print("="*80)
        print("PHASE 2: FEATURE ENGINEERING")
        print("="*80 + "\n")
        
        self.create_temporal_features()
        self.create_price_features()
        self.create_location_features()
        self.create_commodity_features()
        self.create_weather_features()
        self.create_interaction_features()
        self.handle_missing_values()
        
        feature_cols = self.get_feature_list()
        print(f"\n[SUCCESS] Created {len(feature_cols)} features")
        print(f"Total columns in dataset: {len(self.df.columns)}")
        
        return self.df, feature_cols

if __name__ == "__main__":
    # Load data
    data_file = Path('data/kaggle_combined/all_kaggle_final_complete.csv')
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
    print(f"Loaded {len(df):,} records\n")
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_features, feature_cols = engineer.engineer_all_features()
    
    # Save feature-engineered data
    output_file = Path('data/processed/data_with_features.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Feature-engineered data saved to: {output_file}")
    
    # Save feature list
    feature_list_file = Path('data/processed/feature_list.txt')
    with open(feature_list_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"[SUCCESS] Feature list saved to: {feature_list_file}")

