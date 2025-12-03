"""
Retrain model with consolidated data and evaluate performance
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from phase2_feature_engineering import FeatureEngineer
from phase3_model_development import ModelTrainer
from phase4_model_evaluation import ModelEvaluator

def retrain_and_evaluate():
    """Retrain model with consolidated data and evaluate"""
    print("=" * 80)
    print("RETRAINING MODEL WITH CONSOLIDATED DATA")
    print("=" * 80)
    
    # Load consolidated data
    consolidated_file = Path('data/combined/all_sources_consolidated.csv')
    if not consolidated_file.exists():
        print(f"[ERROR] Consolidated data file not found: {consolidated_file}")
        print("Please run consolidate_location_data.py first.")
        return False
    
    print(f"\nLoading consolidated data from: {consolidated_file}")
    df = pd.read_csv(consolidated_file, parse_dates=['date'], low_memory=False)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique states: {df['state'].nunique()}")
    print(f"Unique districts: {df['district'].nunique()}")
    print(f"Unique crops: {df['crop'].nunique()}")
    
    # Phase 2: Feature Engineering
    print("\n" + "=" * 80)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    engineer = FeatureEngineer(df)
    engineer.engineer_all_features()
    df_with_features = engineer.df  # Get the dataframe with features
    
    # Save feature-engineered data
    processed_dir = Path('data/processed_consolidated')
    processed_dir.mkdir(exist_ok=True)
    
    output_file = processed_dir / 'data_with_features.csv'
    df_with_features.to_csv(output_file, index=False)
    print(f"\nSaved feature-engineered data to: {output_file}")
    
    # Get feature list
    feature_cols = engineer.get_feature_list()
    print(f"\nTotal features created: {len(feature_cols)}")
    
    # Save feature list
    feature_file = processed_dir / 'feature_list.txt'
    with open(feature_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"Saved feature list to: {feature_file}")
    
    # Phase 3: Model Development
    print("\n" + "=" * 80)
    print("PHASE 3: MODEL DEVELOPMENT")
    print("=" * 80)
    
    trainer = ModelTrainer(df_with_features, feature_cols, target_col='price')
    
    # Prepare data splits
    train_data, val_data, test_data = trainer.prepare_data(test_size=0.2, val_size=0.1)
    
    # Train baseline models
    trainer.train_baseline_models(train_data, val_data, test_data)
    
    # Train tree-based models
    trainer.train_tree_models(train_data, val_data, test_data)
    
    # Train neural network models (if available)
    trainer.train_neural_networks(train_data, val_data, test_data)
    
    # Train ensemble models
    trainer.train_ensemble(train_data, val_data, test_data)
    
    # Save models
    model_dir = Path('models/consolidated')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_models(model_dir)
    
    # Save metadata manually
    import json
    metadata = {
        'feature_cols': feature_cols,
        'target_col': 'price',
        'results': trainer.results,
        'training_date': datetime.now().isoformat()
    }
    if hasattr(trainer, 'ensemble_weights'):
        metadata['ensemble_weights'] = trainer.ensemble_weights
    
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModels saved to: {model_dir}")
    
    # Phase 4: Model Evaluation
    print("\n" + "=" * 80)
    print("PHASE 4: MODEL EVALUATION")
    print("=" * 80)
    
    # Use results already computed during training
    evaluation_results = trainer.results
    
    # Generate evaluation report
    report_dir = Path('evaluation/consolidated')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    import json
    with open(report_dir / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {report_dir}")
    
    # Print summary
    print("\nModel Performance Summary (Test Set):")
    print("-" * 80)
    results_df = pd.DataFrame(evaluation_results).T
    if 'test_r2' in results_df.columns:
        results_df = results_df.sort_values('test_r2', ascending=False)
        print(results_df[['test_mae', 'test_rmse', 'test_mape', 'test_r2']].to_string())
        best_model = results_df.index[0]
        print(f"\nBest Model: {best_model}")
        print(f"  Test R²: {results_df.loc[best_model, 'test_r2']:.4f}")
        print(f"  Test MAE: {results_df.loc[best_model, 'test_mae']:.2f}")
        print(f"  Test RMSE: {results_df.loc[best_model, 'test_rmse']:.2f}")
        print(f"  Test MAPE: {results_df.loc[best_model, 'test_mape']:.2f}%")
    
    print(f"\nEvaluation reports saved to: {report_dir}")
    
    # Compare with previous model
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS MODEL")
    print("=" * 80)
    
    try:
        import json
        old_metadata = Path('models/phase3/metadata.json')
        if old_metadata.exists():
            with open(old_metadata, 'r') as f:
                old_results = json.load(f)
            
            print("\nPrevious Model (Phase 3) Results:")
            if 'results' in old_results:
                best_old = max(old_results['results'].items(), 
                             key=lambda x: x[1].get('test_r2', -999))
                print(f"  Best model: {best_old[0]}")
                print(f"  Test R²: {best_old[1].get('test_r2', 'N/A'):.4f}")
                print(f"  Test MAE: {best_old[1].get('test_mae', 'N/A'):.2f}")
                print(f"  Test RMSE: {best_old[1].get('test_rmse', 'N/A'):.2f}")
                print(f"  Test MAPE: {best_old[1].get('test_mape', 'N/A'):.2f}%")
            
            print("\nNew Model (Consolidated Data) Results:")
            best_new = max(evaluation_results.items(),
                          key=lambda x: x[1].get('test_r2', -999) if isinstance(x[1], dict) else -999)
            if isinstance(best_new[1], dict):
                print(f"  Best model: {best_new[0]}")
                print(f"  Test R²: {best_new[1].get('test_r2', 'N/A'):.4f}")
                print(f"  Test MAE: {best_new[1].get('test_mae', 'N/A'):.2f}")
                print(f"  Test RMSE: {best_new[1].get('test_rmse', 'N/A'):.2f}")
                print(f"  Test MAPE: {best_new[1].get('test_mape', 'N/A'):.2f}%")
                
                # Calculate improvement
                if 'results' in old_results and best_old[0] in old_results['results']:
                    old_r2 = old_results['results'][best_old[0]].get('test_r2', 0)
                    new_r2 = best_new[1].get('test_r2', 0)
                    improvement = ((new_r2 - old_r2) / abs(old_r2)) * 100 if old_r2 != 0 else 0
                    print(f"\n  R² Improvement: {improvement:+.2f}%")
            
            # Calculate improvement
            if 'results' in old_results and best_old[0] in old_results['results']:
                old_r2 = old_results['results'][best_old[0]].get('test_r2', 0)
                new_r2 = best_new[1].get('r2_score', 0)
                improvement = ((new_r2 - old_r2) / abs(old_r2)) * 100 if old_r2 != 0 else 0
                print(f"\n  R² Improvement: {improvement:+.2f}%")
    except Exception as e:
        print(f"Could not compare with previous model: {e}")
    
    print("\n" + "=" * 80)
    print("[OK] Model retraining and evaluation complete!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = retrain_and_evaluate()
    sys.exit(0 if success else 1)

