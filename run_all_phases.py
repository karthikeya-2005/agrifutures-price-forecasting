"""Master script to run all phases sequentially"""
import sys
from pathlib import Path

def run_phase(phase_num, phase_name, script_name):
    """Run a phase and handle errors"""
    print("\n" + "="*80)
    print(f"STARTING PHASE {phase_num}: {phase_name}")
    print("="*80)
    
    script_path = Path(script_name)
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_name}")
        return False
    
    try:
        # Import and run
        module_name = script_path.stem
        if phase_num == 1:
            from phase1_data_analysis import DataAnalyzer
            data_file = Path('data/kaggle_combined/all_kaggle_final_complete.csv')
            if not data_file.exists():
                print(f"[ERROR] Data file not found: {data_file}")
                return False
            analyzer = DataAnalyzer(data_file)
            analyzer.run_full_analysis()
        elif phase_num == 2:
            from phase2_feature_engineering import FeatureEngineer
            import pandas as pd
            data_file = Path('data/kaggle_combined/all_kaggle_final_complete.csv')
            df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
            engineer = FeatureEngineer(df)
            engineer.engineer_all_features()
        elif phase_num == 3:
            from phase3_model_development import ModelTrainer
            import pandas as pd
            import numpy as np
            data_file = Path('data/processed/data_with_features.csv')
            if not data_file.exists():
                print(f"[ERROR] Feature-engineered data not found. Run Phase 2 first.")
                return False
            df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
            feature_file = Path('data/processed/feature_list.txt')
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_cols = [line.strip() for line in f if line.strip()]
            else:
                feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                               if col not in ['price']]
            trainer = ModelTrainer(df, feature_cols)
            trainer.train_all_models()
        elif phase_num == 4:
            from phase4_model_evaluation import ModelEvaluator
            import pandas as pd
            import numpy as np
            data_file = Path('data/processed/data_with_features.csv')
            df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
            feature_file = Path('data/processed/feature_list.txt')
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_cols = [line.strip() for line in f if line.strip()]
            else:
                feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                               if col not in ['price']]
            df = df.sort_values('date').reset_index(drop=True)
            split_idx = int(len(df) * 0.8)
            test_df = df.iloc[split_idx:].copy()
            X_test = test_df[feature_cols].select_dtypes(include=[np.number])
            X_test = X_test.fillna(X_test.mean())
            y_test = test_df['price']
            evaluator = ModelEvaluator()
            evaluator.run_full_evaluation(X_test, y_test, 
                                          X_full=df[feature_cols].select_dtypes(include=[np.number]),
                                          y_full=df['price'])
        elif phase_num == 5:
            from phase5_system_integration import SystemIntegrator
            integrator = SystemIntegrator()
            integrator.run_full_integration()
        
        print(f"\n[SUCCESS] Phase {phase_num} completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Phase {phase_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all phases"""
    phases = [
        (1, "Data Analysis & Exploration", "phase1_data_analysis.py"),
        (2, "Feature Engineering", "phase2_feature_engineering.py"),
        (3, "Model Development", "phase3_model_development.py"),
        (4, "Model Evaluation", "phase4_model_evaluation.py"),
        (5, "System Integration", "phase5_system_integration.py")
    ]
    
    print("="*80)
    print("AGRICULTURAL PRICE PREDICTION - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis script will run all 5 phases sequentially.")
    print("Each phase builds on the previous one.")
    print("\nPhases:")
    for phase_num, phase_name, _ in phases:
        print(f"  {phase_num}. {phase_name}")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    for phase_num, phase_name, script_name in phases:
        success = run_phase(phase_num, phase_name, script_name)
        if not success:
            print(f"\n[ERROR] Phase {phase_num} failed. Stopping pipeline.")
            print("Please fix the error and rerun from this phase.")
            return
    
    print("\n" + "="*80)
    print("ALL PHASES COMPLETE!")
    print("="*80)
    print("\nSystem is ready for deployment.")

if __name__ == "__main__":
    main()

