"""
Retrain models with weather-enriched data
This script retrains all models using data that includes historical weather information
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from phase2_feature_engineering import FeatureEngineer
from phase3_model_development import ModelTrainer
from phase4_model_evaluation import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrain_with_weather_data():
    """Retrain models with weather-enriched data"""
    logger.info("=" * 80)
    logger.info("RETRAINING MODELS WITH WEATHER-ENRICHED DATA")
    logger.info("=" * 80)
    
    # Check for weather-enriched data
    enriched_file = Path('data/combined/all_sources_consolidated_with_weather.csv')
    consolidated_file = Path('data/combined/all_sources_consolidated.csv')
    
    if enriched_file.exists():
        data_file = enriched_file
        logger.info(f"Using weather-enriched data: {data_file}")
    elif consolidated_file.exists():
        data_file = consolidated_file
        logger.warning(f"Weather-enriched data not found. Using: {data_file}")
        logger.warning("Run enrich_data_with_weather.py first to get better weather relationships.")
    else:
        logger.error("No data file found. Please ensure consolidated data exists.")
        return False
    
    logger.info(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique states: {df['state'].nunique()}")
    logger.info(f"Unique districts: {df['district'].nunique()}")
    logger.info(f"Unique crops: {df['crop'].nunique()}")
    
    # Check weather data availability
    weather_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'temperature', 'rainfall']
    available_weather = [col for col in weather_cols if col in df.columns]
    if available_weather:
        weather_coverage = df[available_weather].notna().any(axis=1).sum()
        logger.info(f"\nWeather data available in columns: {available_weather}")
        logger.info(f"Weather data coverage: {weather_coverage:,} / {len(df):,} records ({weather_coverage/len(df)*100:.1f}%)")
    else:
        logger.warning("No weather columns found in data")
    
    # Phase 2: Feature Engineering
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    engineer = FeatureEngineer(df)
    engineer.engineer_all_features()
    df_with_features = engineer.df
    
    # Save feature-engineered data
    processed_dir = Path('data/processed_with_weather')
    processed_dir.mkdir(exist_ok=True)
    
    output_file = processed_dir / 'data_with_features.csv'
    df_with_features.to_csv(output_file, index=False)
    logger.info(f"\nSaved feature-engineered data to: {output_file}")
    
    # Get feature list
    feature_cols = engineer.get_feature_list()
    logger.info(f"\nTotal features created: {len(feature_cols)}")
    
    # Check which weather features were created
    weather_features = [f for f in feature_cols if any(w in f.lower() for w in ['temp', 'rain', 'weather'])]
    if weather_features:
        logger.info(f"Weather-related features: {len(weather_features)}")
        logger.info(f"  Examples: {weather_features[:5]}")
    else:
        logger.warning("No weather-related features were created")
    
    # Save feature list
    feature_file = processed_dir / 'feature_list.txt'
    with open(feature_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    logger.info(f"Saved feature list to: {feature_file}")
    
    # Phase 3: Model Development
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: MODEL DEVELOPMENT")
    logger.info("=" * 80)
    
    trainer = ModelTrainer(df_with_features, feature_cols, target_col='price')
    
    # Prepare data splits
    train_data, val_data, test_data = trainer.prepare_data(test_size=0.2, val_size=0.1)
    
    logger.info(f"Training set: {len(train_data):,} records")
    logger.info(f"Validation set: {len(val_data):,} records")
    logger.info(f"Test set: {len(test_data):,} records")
    
    # Train baseline models
    logger.info("\nTraining baseline models...")
    trainer.train_baseline_models(train_data, val_data, test_data)
    
    # Train tree-based models
    logger.info("\nTraining tree-based models...")
    trainer.train_tree_models(train_data, val_data, test_data)
    
    # Train neural network models
    logger.info("\nTraining neural network models...")
    trainer.train_neural_networks(train_data, val_data, test_data)
    
    # Train ensemble models
    logger.info("\nTraining ensemble models...")
    trainer.train_ensemble(train_data, val_data, test_data)
    
    # Save models
    model_dir = Path('models/with_weather')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving models to: {model_dir}")
    trainer.save_models(model_dir)
    
    # Save metadata
    import json
    metadata = {
        'feature_names': feature_cols,
        'target_col': 'price',
        'results': trainer.results,
        'training_date': datetime.now().isoformat(),
        'data_source': str(data_file),
        'weather_features': weather_features,
        'total_records': len(df),
        'weather_coverage_pct': (weather_coverage/len(df)*100) if 'weather_coverage' in locals() else 0
    }
    if hasattr(trainer, 'ensemble_weights'):
        metadata['ensemble_weights'] = trainer.ensemble_weights
    
    metadata_file = model_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_file}")
    
    # Phase 4: Model Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Extract test data for evaluation
    X_test, y_test = test_data
    
    # ModelEvaluator loads models from directory
    evaluator = ModelEvaluator(models_dir=str(model_dir))
    evaluator.load_models()
    
    # Evaluate models with test data
    if len(X_test) > 0:
        evaluator.evaluate_models(X_test, y_test)
        # Generate reports if method exists
        if hasattr(evaluator, 'generate_reports'):
            try:
                evaluator.generate_reports(model_dir)
            except Exception as e:
                logger.warning(f"Could not generate reports: {e}")
    
    # Get best model from trainer results
    best_model = None
    best_score = float('inf')
    if hasattr(trainer, 'results') and trainer.results:
        for model_name, metrics in trainer.results.items():
            if 'test_mae' in metrics:
                if metrics['test_mae'] < best_score:
                    best_score = metrics['test_mae']
                    best_model = model_name
    
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {model_dir}")
    if best_model:
        logger.info(f"Best model (lowest test MAE): {best_model} (MAE: {best_score:.2f})")
    else:
        logger.info("Check trainer.results for model performance metrics")
    
    return True

if __name__ == "__main__":
    success = retrain_with_weather_data()
    sys.exit(0 if success else 1)

