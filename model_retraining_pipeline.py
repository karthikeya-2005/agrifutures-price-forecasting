"""
Automated Model Retraining Pipeline
Retrains models with new data on a schedule
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from phase2_feature_engineering import FeatureEngineer
from phase3_model_development import ModelTrainer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retraining.log'),
        logging.StreamHandler()
    ]
)

class ModelRetrainingPipeline:
    def __init__(self, 
                 data_file='data/combined/all_sources_combined.csv',
                 models_dir='models/phase3',
                 retrain_threshold_days=30):
        """Initialize retraining pipeline"""
        self.data_file = Path(data_file)
        self.models_dir = Path(models_dir)
        self.retrain_threshold_days = retrain_threshold_days
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
    def check_retrain_needed(self):
        """Check if retraining is needed based on last training date"""
        metadata_file = self.models_dir / 'metadata.json'
        
        if not metadata_file.exists():
            logging.info("No existing models found. Retraining needed.")
            return True
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            training_date = pd.to_datetime(metadata.get('training_date', '2000-01-01'))
            days_since_training = (datetime.now() - training_date).days
            
            if days_since_training >= self.retrain_threshold_days:
                logging.info(f"Last training: {training_date}. Days since: {days_since_training}. Retraining needed.")
                return True
            else:
                logging.info(f"Last training: {training_date}. Days since: {days_since_training}. No retraining needed.")
                return False
        except Exception as e:
            logging.error(f"Error checking retrain status: {e}")
            return True
    
    def load_latest_data(self):
        """Load latest combined data"""
        if not self.data_file.exists():
            logging.error(f"Data file not found: {self.data_file}")
            return None
        
        logging.info(f"Loading data from: {self.data_file}")
        df = pd.read_csv(self.data_file, parse_dates=['date'], low_memory=False)
        logging.info(f"Loaded {len(df):,} records")
        
        return df
    
    def backup_existing_models(self):
        """Backup existing models before retraining"""
        backup_dir = self.models_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'models_backup_{timestamp}'
        backup_path.mkdir(exist_ok=True)
        
        # Copy model files
        import shutil
        for model_file in self.models_dir.glob('*.pkl'):
            shutil.copy(model_file, backup_path / model_file.name)
        
        for model_file in self.models_dir.glob('*.json'):
            shutil.copy(model_file, backup_path / model_file.name)
        
        if (self.models_dir / 'neural_networks').exists():
            shutil.copytree(
                self.models_dir / 'neural_networks',
                backup_path / 'neural_networks',
                dirs_exist_ok=True
            )
        
        logging.info(f"Backed up models to: {backup_path}")
        return backup_path
    
    def retrain_models(self, df):
        """Retrain all models with new data"""
        logging.info("="*80)
        logging.info("STARTING MODEL RETRAINING")
        logging.info("="*80)
        
        # Backup existing models
        backup_path = self.backup_existing_models()
        
        try:
            # Feature engineering
            logging.info("Step 1: Feature Engineering...")
            engineer = FeatureEngineer(df)
            df_features, feature_cols = engineer.engineer_all_features()
            
            # Save feature-engineered data
            output_file = Path('data/processed/data_with_features.csv')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(output_file, index=False)
            logging.info(f"Saved feature-engineered data: {len(df_features):,} records, {len(feature_cols)} features")
            
            # Model training
            logging.info("Step 2: Training Models...")
            trainer = ModelTrainer(df_features, feature_cols)
            results = trainer.train_all_models()
            
            # Save retraining log
            retrain_log = {
                'retraining_date': datetime.now().isoformat(),
                'data_records': len(df),
                'features_count': len(feature_cols),
                'backup_location': str(backup_path),
                'model_results': results,
                'status': 'success'
            }
            
            log_file = self.log_dir / f'retraining_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(log_file, 'w') as f:
                json.dump(retrain_log, f, indent=2, default=str)
            
            logging.info(f"Retraining complete. Log saved to: {log_file}")
            return True
            
        except Exception as e:
            logging.error(f"Retraining failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Restore backup on failure
            logging.info("Attempting to restore backup...")
            # (Restoration logic would go here)
            
            return False
    
    def run(self, force=False):
        """Run retraining pipeline"""
        if not force and not self.check_retrain_needed():
            logging.info("Retraining not needed at this time.")
            return False
        
        # Load data
        df = self.load_latest_data()
        if df is None:
            return False
        
        # Retrain
        success = self.retrain_models(df)
        
        if success:
            logging.info("="*80)
            logging.info("RETRAINING COMPLETE")
            logging.info("="*80)
        else:
            logging.error("RETRAINING FAILED")
        
        return success

if __name__ == "__main__":
    import sys
    
    # Check for force flag
    force = '--force' in sys.argv
    
    pipeline = ModelRetrainingPipeline()
    success = pipeline.run(force=force)
    
    if success:
        print("\n[SUCCESS] Models retrained successfully!")
    else:
        print("\n[INFO] Retraining not needed or failed. Check logs for details.")

