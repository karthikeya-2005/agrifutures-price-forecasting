"""
Consolidate location data by merging states and districts with identical names
but different spellings or styles
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from location_normalizer import normalize_state_name, normalize_district_name, normalize_location

def find_similar_names(names_list, threshold=0.85):
    """
    Find similar names in a list using fuzzy matching
    Returns groups of similar names
    """
    from difflib import SequenceMatcher
    
    groups = []
    processed = set()
    
    for name1 in names_list:
        if name1 in processed:
            continue
        
        group = [name1]
        processed.add(name1)
        
        for name2 in names_list:
            if name2 in processed or name1 == name2:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            
            # Also check if one contains the other (for cases like "Chennai" vs "Chennai District")
            contains_match = (name1.lower() in name2.lower() or name2.lower() in name1.lower()) and similarity > 0.7
            
            if similarity >= threshold or contains_match:
                group.append(name2)
                processed.add(name2)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups

def consolidate_states(df):
    """Consolidate state names with variations"""
    print("=" * 80)
    print("CONSOLIDATING STATE NAMES")
    print("=" * 80)
    
    unique_states = df['state'].dropna().unique().tolist()
    print(f"\nFound {len(unique_states)} unique state names")
    
    # Create mapping: variation -> standard name
    state_mapping = {}
    
    # Use normalization function
    for state in unique_states:
        normalized = normalize_state_name(state)
        if normalized and normalized != state:
            state_mapping[state] = normalized
            print(f"  Mapping: '{state}' -> '{normalized}'")
    
    # Find similar names that normalization might have missed
    similar_groups = find_similar_names(unique_states, threshold=0.9)
    
    for group in similar_groups:
        # Use the most common one as standard, or the normalized one
        group_counts = df[df['state'].isin(group)]['state'].value_counts()
        most_common = group_counts.index[0]
        
        # Try to normalize the most common
        normalized_most_common = normalize_state_name(most_common) or most_common
        
        for state in group:
            if state != normalized_most_common:
                if state not in state_mapping:
                    state_mapping[state] = normalized_most_common
                    print(f"  Mapping (similar): '{state}' -> '{normalized_most_common}'")
    
    # Apply mapping
    df['state_original'] = df['state'].copy()
    df['state'] = df['state'].map(lambda x: state_mapping.get(x, x))
    
    print(f"\nConsolidated {len(state_mapping)} state name variations")
    print(f"Unique states after consolidation: {df['state'].nunique()}")
    
    return df, state_mapping

def consolidate_districts(df):
    """Consolidate district names with variations"""
    print("\n" + "=" * 80)
    print("CONSOLIDATING DISTRICT NAMES")
    print("=" * 80)
    
    # Group by state first
    district_mapping = {}
    
    for state in df['state'].dropna().unique():
        state_df = df[df['state'] == state]
        unique_districts = state_df['district'].dropna().unique().tolist()
        
        if len(unique_districts) == 0:
            continue
        
        print(f"\nProcessing {state} ({len(unique_districts)} districts)")
        
        # Use normalization function
        for district in unique_districts:
            normalized = normalize_district_name(district, state)
            if normalized and normalized != district:
                key = (state, district)
                district_mapping[key] = normalized
                print(f"  Mapping: '{district}' -> '{normalized}'")
        
        # Find similar names within the state
        similar_groups = find_similar_names(unique_districts, threshold=0.88)
        
        for group in similar_groups:
            # Use the most common one as standard
            group_counts = state_df[state_df['district'].isin(group)]['district'].value_counts()
            most_common = group_counts.index[0]
            
            # Try to normalize the most common
            normalized_most_common = normalize_district_name(most_common, state) or most_common
            
            for district in group:
                if district != normalized_most_common:
                    key = (state, district)
                    if key not in district_mapping:
                        district_mapping[key] = normalized_most_common
                        print(f"  Mapping (similar): '{district}' -> '{normalized_most_common}'")
    
    # Apply mapping
    df['district_original'] = df['district'].copy()
    
    def map_district(row):
        key = (row['state'], row['district'])
        return district_mapping.get(key, row['district'])
    
    df['district'] = df.apply(map_district, axis=1)
    
    print(f"\nConsolidated {len(district_mapping)} district name variations")
    print(f"Unique districts after consolidation: {df.groupby('state')['district'].nunique().sum()}")
    
    return df, district_mapping

def consolidate_crops(df):
    """Consolidate crop/commodity names with variations"""
    print("\n" + "=" * 80)
    print("CONSOLIDATING CROP/COMMODITY NAMES")
    print("=" * 80)
    
    unique_crops = df['crop'].dropna().unique().tolist()
    print(f"\nFound {len(unique_crops)} unique crop names")
    
    crop_mapping = {}
    
    # Common variations
    common_variations = {
        'Rice': ['Paddy', 'Paddy Rice', 'Rice (Paddy)'],
        'Wheat': ['Wheat Grain', 'Wheat (Grain)'],
        'Onion': ['Onions', 'Onion (Dry)'],
        'Tomato': ['Tomatoes', 'Tomato (Fresh)'],
        'Potato': ['Potatoes', 'Potato (Fresh)'],
        'Chili': ['Chilli', 'Chillies', 'Red Chili', 'Red Chilli', 'Green Chili', 'Green Chilli'],
        'Turmeric': ['Turmeric (Dry)', 'Turmeric Powder'],
        'Ginger': ['Ginger (Fresh)', 'Ginger (Dry)'],
        'Garlic': ['Garlic (Dry)', 'Garlic (Fresh)'],
    }
    
    # Create reverse mapping
    for standard, variations in common_variations.items():
        for variation in variations:
            if variation in unique_crops and standard in unique_crops:
                crop_mapping[variation] = standard
                print(f"  Mapping: '{variation}' -> '{standard}'")
    
    # Find similar crop names
    similar_groups = find_similar_names(unique_crops, threshold=0.85)
    
    for group in similar_groups:
        # Use the most common one as standard
        group_counts = df[df['crop'].isin(group)]['crop'].value_counts()
        most_common = group_counts.index[0]
        
        for crop in group:
            if crop != most_common and crop not in crop_mapping:
                crop_mapping[crop] = most_common
                print(f"  Mapping (similar): '{crop}' -> '{most_common}'")
    
    # Apply mapping
    df['crop_original'] = df['crop'].copy()
    df['crop'] = df['crop'].map(lambda x: crop_mapping.get(x, x))
    
    print(f"\nConsolidated {len(crop_mapping)} crop name variations")
    print(f"Unique crops after consolidation: {df['crop'].nunique()}")
    
    return df, crop_mapping

def clean_and_consolidate_data(input_file, output_file):
    """
    Main function to clean and consolidate location data
    """
    print("=" * 80)
    print("DATA CONSOLIDATION AND FEATURE ENGINEERING")
    print("=" * 80)
    print(f"\nLoading data from: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file, low_memory=False, parse_dates=['date'], infer_datetime_format=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original unique states: {df['state'].nunique()}")
    print(f"Original unique districts: {df['district'].nunique()}")
    print(f"Original unique crops: {df['crop'].nunique()}")
    
    # Remove rows with missing essential data
    initial_count = len(df)
    df = df.dropna(subset=['state', 'district', 'crop', 'price', 'date'])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"\nRemoved {removed} rows with missing essential data")
    
    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] > 0]  # Remove invalid prices
    
    # Consolidate states
    df, state_mapping = consolidate_states(df)
    
    # Consolidate districts
    df, district_mapping = consolidate_districts(df)
    
    # Consolidate crops
    df, crop_mapping = consolidate_crops(df)
    
    # Remove duplicates (same state, district, crop, date)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['state', 'district', 'crop', 'date'], keep='first')
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed} duplicate records")
    
    # Save mappings for reference
    mappings_dir = Path('data/mappings')
    mappings_dir.mkdir(exist_ok=True)
    
    import json
    with open(mappings_dir / 'state_mapping.json', 'w') as f:
        json.dump(state_mapping, f, indent=2)
    
    with open(mappings_dir / 'district_mapping.json', 'w') as f:
        # Convert tuple keys to strings
        district_mapping_str = {f"{k[0]}_{k[1]}": v for k, v in district_mapping.items()}
        json.dump(district_mapping_str, f, indent=2)
    
    with open(mappings_dir / 'crop_mapping.json', 'w') as f:
        json.dump(crop_mapping, f, indent=2)
    
    print(f"\nSaved mappings to: {mappings_dir}")
    
    # Save consolidated data
    print(f"\nSaving consolidated data to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("CONSOLIDATION SUMMARY")
    print("=" * 80)
    print(f"Final data shape: {df.shape}")
    print(f"Final unique states: {df['state'].nunique()}")
    print(f"Final unique districts: {df['district'].nunique()}")
    print(f"Final unique crops: {df['crop'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df):,}")
    
    return df, state_mapping, district_mapping, crop_mapping

if __name__ == "__main__":
    input_file = Path('data/combined/all_sources_combined.csv')
    output_file = Path('data/combined/all_sources_consolidated.csv')
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    df, state_mapping, district_mapping, crop_mapping = clean_and_consolidate_data(
        input_file, output_file
    )
    
    print("\n[OK] Data consolidation complete!")

