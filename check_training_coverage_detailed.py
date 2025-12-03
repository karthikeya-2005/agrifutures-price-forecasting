"""
Detailed analysis of training data coverage
"""
import pandas as pd
from pathlib import Path
from locations_data import INDIA_STATES, INDIA_DISTRICTS

def analyze_coverage():
    """Analyze training data coverage"""
    print("=" * 80)
    print("COMPREHENSIVE TRAINING DATA COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Load consolidated data
    data_file = Path('data/combined/all_sources_consolidated.csv')
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file, usecols=['state', 'district', 'crop'], low_memory=False)
    
    print(f"\n1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total Records: {len(df):,}")
    print(f"Unique States: {df['state'].nunique()}")
    print(f"Unique Districts: {df['district'].nunique()}")
    print(f"Unique Commodities: {df['crop'].nunique()}")
    print(f"State-District Combinations: {df[['state', 'district']].drop_duplicates().shape[0]:,}")
    print(f"State-District-Crop Combinations: {df[['state', 'district', 'crop']].drop_duplicates().shape[0]:,}")
    
    # State coverage
    print(f"\n2. STATE COVERAGE")
    print("-" * 80)
    training_states = set(df['state'].dropna().unique())
    all_india_states = set(INDIA_STATES)
    missing_states = all_india_states - training_states
    
    print(f"Total Indian States: {len(INDIA_STATES)}")
    print(f"States in Training Data: {len(training_states)}")
    print(f"Missing States: {len(missing_states)}")
    print(f"Coverage: {len(training_states)/len(INDIA_STATES)*100:.1f}%")
    
    if missing_states:
        print(f"\nMissing States ({len(missing_states)}):")
        for state in sorted(missing_states):
            print(f"  - {state}")
    
    # District coverage by state
    print(f"\n3. DISTRICT COVERAGE BY STATE")
    print("-" * 80)
    state_district_coverage = []
    
    for state in sorted(training_states):
        state_df = df[df['state'] == state]
        training_districts = set(state_df['district'].dropna().unique())
        
        # Get expected districts from locations_data
        expected_districts = set(INDIA_DISTRICTS.get(state, []))
        
        coverage_pct = (len(training_districts) / len(expected_districts) * 100) if expected_districts else 0
        
        state_district_coverage.append({
            'State': state,
            'Training Districts': len(training_districts),
            'Expected Districts': len(expected_districts),
            'Coverage %': f"{coverage_pct:.1f}%"
        })
    
    coverage_df = pd.DataFrame(state_district_coverage)
    print(coverage_df.to_string(index=False))
    
    # Commodity coverage by state
    print(f"\n4. COMMODITY COVERAGE BY STATE")
    print("-" * 80)
    state_commodity = df.groupby('state')['crop'].nunique().sort_values(ascending=False)
    print(state_commodity.to_string())
    print(f"\nAverage Commodities per State: {state_commodity.mean():.1f}")
    print(f"Min Commodities: {state_commodity.min()}")
    print(f"Max Commodities: {state_commodity.max()}")
    
    # Detailed state analysis
    print(f"\n5. DETAILED STATE ANALYSIS")
    print("-" * 80)
    detailed = df.groupby('state').agg({
        'district': 'nunique',
        'crop': 'nunique'
    }).sort_values('district', ascending=False)
    detailed.columns = ['Districts', 'Commodities']
    detailed['Combinations'] = df.groupby('state').apply(
        lambda x: x[['district', 'crop']].drop_duplicates().shape[0]
    )
    print(detailed.to_string())
    
    # Summary
    print(f"\n6. SUMMARY")
    print("-" * 80)
    print(f"[OK] States Covered: {len(training_states)}/{len(INDIA_STATES)} ({len(training_states)/len(INDIA_STATES)*100:.1f}%)")
    print(f"[OK] Total Districts: {df['district'].nunique()}")
    print(f"[OK] Total Commodities: {df['crop'].nunique()}")
    print(f"[OK] Total Combinations: {df[['state', 'district', 'crop']].drop_duplicates().shape[0]:,}")
    
    # Check if model is trained on all available combinations
    print(f"\n7. MODEL TRAINING COVERAGE")
    print("-" * 80)
    
    # Check if models exist for combinations
    model_dir = Path('models/consolidated')
    if model_dir.exists():
        print("Model files found in consolidated directory")
        # Count unique combinations in training data
        unique_combos = df[['state', 'district', 'crop']].drop_duplicates()
        print(f"Unique State-District-Crop combinations in data: {len(unique_combos):,}")
        print("\nNote: The model is a global model trained on all available data,")
        print("not individual models per combination. This is more efficient and")
        print("allows the model to learn patterns across all locations and commodities.")
    else:
        print("Note: Models directory not found. Check if models were trained.")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_coverage()

