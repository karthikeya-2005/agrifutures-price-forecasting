"""Phase 3: Model Development - Baseline, Tree-based, Neural Networks, Ensemble"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Neural network imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural network models will be skipped.")

class ModelTrainer:
    def __init__(self, df, feature_cols, target_col='price'):
        """Initialize trainer with data and features"""
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Prepare train/validation/test splits"""
        print("Preparing data splits...")
        
        # Sort by date for time series split
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Time-based split (more appropriate for time series)
        split_date_1 = self.df['date'].quantile(1 - test_size - val_size)
        split_date_2 = self.df['date'].quantile(1 - test_size)
        
        train_df = self.df[self.df['date'] < split_date_1].copy()
        val_df = self.df[(self.df['date'] >= split_date_1) & (self.df['date'] < split_date_2)].copy()
        test_df = self.df[self.df['date'] >= split_date_2].copy()
        
        print(f"  Train: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"  Val:   {len(val_df):,} records ({val_df['date'].min()} to {val_df['date'].max()})")
        print(f"  Test:  {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")
        
        # Prepare features and targets
        X_train = train_df[self.feature_cols].select_dtypes(include=[np.number])
        y_train = train_df[self.target_col]
        
        X_val = val_df[self.feature_cols].select_dtypes(include=[np.number])
        y_val = val_df[self.target_col]
        
        X_test = test_df[self.feature_cols].select_dtypes(include=[np.number])
        y_test = test_df[self.target_col]
        
        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        X_val = X_val.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())
        
        # Align columns
        common_cols = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]
        
        self.feature_cols = common_cols
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_baseline_models(self, train_data, val_data, test_data):
        """Train baseline models"""
        print("\n" + "="*80)
        print("BASELINE MODELS")
        print("="*80 + "\n")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        baselines = {}
        
        # 1. Historical Average
        print("1. Historical Average...")
        hist_avg = y_train.mean()
        baselines['historical_average'] = hist_avg
        
        # 2. Last Value (Naive)
        print("2. Last Value (Naive)...")
        last_value = y_train.iloc[-1]
        baselines['last_value'] = last_value
        
        # 3. Moving Average (30 days)
        print("3. Moving Average (30 days)...")
        ma_30 = y_train.tail(30).mean()
        baselines['moving_average_30'] = ma_30
        
        # 4. Seasonal Naive (same month last year)
        print("4. Seasonal Naive...")
        # Use median price as seasonal naive
        seasonal_naive = y_train.median()
        baselines['seasonal_naive'] = seasonal_naive
        
        # Evaluate baselines
        print("\nEvaluating baselines:")
        for name, pred_value in baselines.items():
            val_pred = np.full(len(y_val), pred_value)
            test_pred = np.full(len(y_test), pred_value)
            
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-6))) * 100
            val_r2 = r2_score(y_val, val_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mape = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-6))) * 100
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"\n{name}:")
            print(f"  Val  - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")
            print(f"  Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")
            
            self.results[name] = {
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_mape': float(val_mape),
                'val_r2': float(val_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'test_r2': float(test_r2)
            }
        
        self.models['baselines'] = baselines
        return baselines
    
    def train_tree_models(self, train_data, val_data, test_data):
        """Train tree-based models"""
        print("\n" + "="*80)
        print("TREE-BASED MODELS")
        print("="*80 + "\n")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        tree_models = {}
        
        # 1. XGBoost
        print("1. Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     verbose=False)
        tree_models['xgboost'] = xgb_model
        
        # 2. LightGBM
        print("2. Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        tree_models['lightgbm'] = lgb_model
        
        # Evaluate tree models
        print("\nEvaluating tree models:")
        for name, model in tree_models.items():
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-6))) * 100
            val_r2 = r2_score(y_val, val_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mape = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-6))) * 100
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"\n{name}:")
            print(f"  Val  - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")
            print(f"  Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")
            
            self.results[name] = {
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_mape': float(val_mape),
                'val_r2': float(val_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'test_r2': float(test_r2)
            }
        
        self.models['tree_models'] = tree_models
        return tree_models
    
    def train_neural_networks(self, train_data, val_data, test_data):
        """Train neural network models"""
        if not TF_AVAILABLE:
            print("\nSkipping neural networks (TensorFlow not available)")
            return {}
        
        print("\n" + "="*80)
        print("NEURAL NETWORK MODELS")
        print("="*80 + "\n")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['neural_net'] = scaler
        
        nn_models = {}
        
        # 1. Feedforward Neural Network
        print("1. Training Feedforward NN...")
        ff_model = Sequential([
            Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        ff_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        ff_model.fit(X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=50,
                    batch_size=256,
                    verbose=0)
        nn_models['feedforward'] = ff_model
        
        # 2. LSTM (requires sequence data)
        print("2. Training LSTM...")
        # Reshape for LSTM (samples, timesteps, features)
        # For simplicity, use single timestep
        X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
        X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        lstm_model.fit(X_train_lstm, y_train,
                      validation_data=(X_val_lstm, y_val),
                      epochs=50,
                      batch_size=256,
                      verbose=0)
        nn_models['lstm'] = lstm_model
        
        # 3. GRU
        print("3. Training GRU...")
        gru_model = Sequential([
            GRU(64, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        gru_model.fit(X_train_lstm, y_train,
                     validation_data=(X_val_lstm, y_val),
                     epochs=50,
                     batch_size=256,
                     verbose=0)
        nn_models['gru'] = gru_model
        
        # Evaluate neural networks
        print("\nEvaluating neural networks:")
        for name, model in nn_models.items():
            if name == 'feedforward':
                val_pred = model.predict(X_val_scaled, verbose=0).flatten()
                test_pred = model.predict(X_test_scaled, verbose=0).flatten()
            else:
                val_pred = model.predict(X_val_lstm, verbose=0).flatten()
                test_pred = model.predict(X_test_lstm, verbose=0).flatten()
            
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-6))) * 100
            val_r2 = r2_score(y_val, val_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mape = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-6))) * 100
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"\n{name}:")
            print(f"  Val  - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")
            print(f"  Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")
            
            self.results[name] = {
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_mape': float(val_mape),
                'val_r2': float(val_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'test_r2': float(test_r2)
            }
        
        self.models['neural_networks'] = nn_models
        return nn_models
    
    def train_ensemble(self, train_data, val_data, test_data):
        """Train ensemble model"""
        print("\n" + "="*80)
        print("ENSEMBLE MODEL")
        print("="*80 + "\n")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Get predictions from all models
        val_predictions = {}
        test_predictions = {}
        
        # Tree models
        if 'tree_models' in self.models:
            for name, model in self.models['tree_models'].items():
                val_predictions[name] = model.predict(X_val)
                test_predictions[name] = model.predict(X_test)
        
        # Neural networks
        if 'neural_networks' in self.models and TF_AVAILABLE:
            scaler = self.scalers.get('neural_net')
            if scaler:
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                for name, model in self.models['neural_networks'].items():
                    if name == 'feedforward':
                        val_predictions[name] = model.predict(X_val_scaled, verbose=0).flatten()
                        test_predictions[name] = model.predict(X_test_scaled, verbose=0).flatten()
                    else:
                        X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
                        X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
                        val_predictions[name] = model.predict(X_val_lstm, verbose=0).flatten()
                        test_predictions[name] = model.predict(X_test_lstm, verbose=0).flatten()
        
        if not val_predictions:
            print("No models available for ensemble")
            return None
        
        # Simple averaging ensemble
        print("Creating averaging ensemble...")
        val_ensemble_pred = np.mean(list(val_predictions.values()), axis=0)
        test_ensemble_pred = np.mean(list(test_predictions.values()), axis=0)
        
        # Weighted ensemble (based on validation performance)
        print("Creating weighted ensemble...")
        weights = {}
        for name, pred in val_predictions.items():
            mae = mean_absolute_error(y_val, pred)
            weights[name] = 1.0 / (mae + 1e-6)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        val_weighted_pred = np.zeros(len(y_val))
        test_weighted_pred = np.zeros(len(y_test))
        
        for name, pred in val_predictions.items():
            val_weighted_pred += weights[name] * pred
        for name, pred in test_predictions.items():
            test_weighted_pred += weights[name] * pred
        
        # Evaluate ensembles
        for name, val_pred, test_pred in [('ensemble_avg', val_ensemble_pred, test_ensemble_pred),
                                          ('ensemble_weighted', val_weighted_pred, test_weighted_pred)]:
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-6))) * 100
            val_r2 = r2_score(y_val, val_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mape = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-6))) * 100
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"\n{name}:")
            print(f"  Val  - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")
            print(f"  Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")
            
            self.results[name] = {
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_mape': float(val_mape),
                'val_r2': float(val_r2),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'test_r2': float(test_r2)
            }
        
        self.models['ensemble'] = {
            'weights': weights,
            'models': list(val_predictions.keys())
        }
        
        return self.models['ensemble']
    
    def save_models(self, output_dir='models/phase3'):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving models to: {output_path}")
        
        # Save tree models
        if 'tree_models' in self.models:
            for name, model in self.models['tree_models'].items():
                model_file = output_path / f'{name}.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  [OK] Saved: {name}.pkl")
        
        # Save neural networks
        if 'neural_networks' in self.models and TF_AVAILABLE:
            nn_dir = output_path / 'neural_networks'
            nn_dir.mkdir(exist_ok=True)
            for name, model in self.models['neural_networks'].items():
                model_file = nn_dir / f'{name}.h5'
                model.save(model_file)
                print(f"  [OK] Saved: neural_networks/{name}.h5")
        
        # Save scalers
        if self.scalers:
            scaler_file = output_path / 'scalers.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers, f)
            print(f"  [OK] Saved: scalers.pkl")
        
        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'results': self.results,
            'ensemble_weights': self.models.get('ensemble', {}).get('weights', {}),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  [OK] Saved: metadata.json")
        
        print(f"\n[SUCCESS] All models saved to: {output_path}")
    
    def train_all_models(self):
        """Train all model types"""
        print("="*80)
        print("PHASE 3: MODEL DEVELOPMENT")
        print("="*80 + "\n")
        
        # Prepare data
        train_data, val_data, test_data = self.prepare_data()
        
        # Train models
        self.train_baseline_models(train_data, val_data, test_data)
        self.train_tree_models(train_data, val_data, test_data)
        self.train_neural_networks(train_data, val_data, test_data)
        self.train_ensemble(train_data, val_data, test_data)
        
        # Save models
        self.save_models()
        
        # Print summary
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('test_r2', ascending=False)
        print(results_df[['test_mae', 'test_rmse', 'test_mape', 'test_r2']].to_string())
        
        best_model = results_df.index[0]
        print(f"\nBest Model: {best_model} (R² = {results_df.loc[best_model, 'test_r2']:.4f})")
        
        print("\n" + "="*80)
        print("PHASE 3 COMPLETE: MODEL DEVELOPMENT")
        print("="*80)
        print("\nNext: Phase 4 - Model Evaluation")
        
        return self.results

if __name__ == "__main__":
    # Load feature-engineered data
    data_file = Path('data/processed/data_with_features.csv')
    if not data_file.exists():
        print(f"[ERROR] Feature-engineered data not found: {data_file}")
        print("Please run phase2_feature_engineering.py first")
        exit(1)
    
    print(f"Loading feature-engineered data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
    print(f"Loaded {len(df):,} records\n")
    
    # Load feature list
    feature_file = Path('data/processed/feature_list.txt')
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: use all numeric columns except target
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['price']]
    
    print(f"Using {len(feature_cols)} features")
    
    # Train models
    trainer = ModelTrainer(df, feature_cols)
    results = trainer.train_all_models()

