"""
Module to get available commodities for specific state-district combinations
Based on the actual training data
"""
import json
import logging
import pandas as pd
from pathlib import Path
from location_normalizer import normalize_state_name, normalize_district_name, normalize_location

logger = logging.getLogger(__name__)

# Load mappings
LOCATION_COMMODITY_MAPPING = {}
STATE_DISTRICT_MAPPING = {}
STATE_DISTRICT_CROP_MAPPING = {}

def load_mappings():
    """Load all mappings from JSON files with error handling"""
    global LOCATION_COMMODITY_MAPPING, STATE_DISTRICT_MAPPING, STATE_DISTRICT_CROP_MAPPING
    
    try:
        # Load location-commodity mapping (state_district -> [crops])
        mapping_file = Path('data/location_commodity_mapping.json')
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    LOCATION_COMMODITY_MAPPING = json.load(f)
            except Exception as e:
                print(f"[WARN] Error loading location_commodity_mapping.json: {e}")
                LOCATION_COMMODITY_MAPPING = {}
        else:
            LOCATION_COMMODITY_MAPPING = {}
        
        # Load state-district mapping
        state_district_file = Path('data/state_district_mapping.json')
        if state_district_file.exists():
            try:
                with open(state_district_file, 'r', encoding='utf-8') as f:
                    STATE_DISTRICT_MAPPING = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading state_district_mapping.json: {e}")
                STATE_DISTRICT_MAPPING = {}
        else:
            STATE_DISTRICT_MAPPING = {}
        
        # Create state-district-crop mapping for faster lookups
        STATE_DISTRICT_CROP_MAPPING = {}
        if LOCATION_COMMODITY_MAPPING:
            for key, crops in LOCATION_COMMODITY_MAPPING.items():
                try:
                    # Key format: "State_District" (with potential space issues)
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        state = parts[0].strip()
                        district = parts[1].strip()
                        state_key = f"{state}_{district}"
                        STATE_DISTRICT_CROP_MAPPING[state_key] = crops
                except Exception as e:
                    # Skip invalid entries
                    continue
    except Exception as e:
        logger.error(f"Error in load_mappings: {e}", exc_info=True)
        LOCATION_COMMODITY_MAPPING = {}
        STATE_DISTRICT_MAPPING = {}
        STATE_DISTRICT_CROP_MAPPING = {}

def reload_mappings():
    """Reload mappings from files (useful when data is updated)"""
    load_mappings()

# Load on import
load_mappings()

def get_available_commodities(state, district):
    """
    Get list of available commodities for a specific state and district
    Based on the training data
    
    Args:
        state: State name
        district: District name
    
    Returns:
        List of available commodity names (sorted)
    """
    # Try consolidated data first (preferred) - THIS IS THE PRIMARY SOURCE
    try:
        data_file = Path('data/combined/all_sources_consolidated.csv')
        if data_file.exists():
            # Read only needed columns for performance
            df = pd.read_csv(data_file, usecols=['state', 'district', 'crop'], low_memory=False)
            # Filter out Unknown entries
            df = df[(df['state'] != 'Unknown') & (df['district'] != 'Unknown') & (df['crop'] != 'Unknown')]
            df = df.dropna(subset=['state', 'district', 'crop'])
            
            # Try exact match first (case-insensitive) with original names
            state_lower = str(state).strip().lower()
            district_lower = str(district).strip().lower()
            
            filtered = df[
                (df['state'].astype(str).str.strip().str.lower() == state_lower) &
                (df['district'].astype(str).str.strip().str.lower() == district_lower)
            ]
            
            # If no exact match, try with normalized names
            if filtered.empty:
                normalized_state, normalized_district = normalize_location(state, district)
                if normalized_state and normalized_district:
                    filtered = df[
                        (df['state'].astype(str).str.strip().str.lower() == normalized_state.strip().lower()) &
                        (df['district'].astype(str).str.strip().str.lower() == normalized_district.strip().lower())
                    ]
            
            if not filtered.empty:
                crops = sorted(filtered['crop'].dropna().unique().tolist())
                # Remove any invalid crop names
                crops = [c for c in crops if c and str(c).strip().lower() not in ['unknown', 'nan', '']]
                if crops:
                    return crops
    except Exception as e:
        print(f"[WARN] Error loading from consolidated CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Try exact match in mapping
    key = f"{state}_{district}"
    if key in STATE_DISTRICT_CROP_MAPPING:
        crops = sorted(STATE_DISTRICT_CROP_MAPPING[key])
        crops = [c for c in crops if c and str(c).lower() != 'unknown']
        return crops
    
    # Try case-insensitive search in LOCATION_COMMODITY_MAPPING
    for mapping_key, crops in LOCATION_COMMODITY_MAPPING.items():
        parts = mapping_key.split('_', 1)
        if len(parts) == 2:
            if parts[0].strip().lower() == state.lower() and parts[1].strip().lower() == district.lower():
                crops = sorted(crops)
                crops = [c for c in crops if c and str(c).lower() != 'unknown']
                return crops
    
    # Fallback to original CSV
    try:
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            df = pd.read_csv(data_file, usecols=['state', 'district', 'crop'], low_memory=False)
            filtered = df[
                (df['state'].str.strip().str.lower() == state.lower()) &
                (df['district'].str.strip().str.lower() == district.lower())
            ]
            if not filtered.empty:
                crops = sorted(filtered['crop'].dropna().unique().tolist())
                crops = [c for c in crops if c and str(c).lower() != 'unknown']
                return crops
    except Exception as e:
        logger.warning(f"Error loading from CSV: {e}")
    
    return []

def get_available_districts(state):
    """
    Get list of available districts for a specific state
    Based on the training data (consolidated)
    Only returns districts that actually have training data
    
    Args:
        state: State name (will be normalized for matching)
    
    Returns:
        List of available district names (sorted)
    """
    # Try consolidated data first (preferred) - THIS IS THE PRIMARY SOURCE
    try:
        data_file = Path('data/combined/all_sources_consolidated.csv')
        if data_file.exists():
            # Read only needed columns for performance
            df = pd.read_csv(data_file, usecols=['state', 'district'], low_memory=False)
            # Filter out Unknown
            df = df[(df['state'] != 'Unknown') & (df['district'] != 'Unknown')]
            df = df.dropna(subset=['state', 'district'])
            
            # Try exact match first
            filtered = df[df['state'].astype(str).str.strip().str.lower() == state.strip().lower()]
            
            # If no exact match, try with normalized state
            if filtered.empty:
                normalized_state = normalize_state_name(state)
                if normalized_state:
                    filtered = df[df['state'].astype(str).str.strip().str.lower() == normalized_state.strip().lower()]
            
            if not filtered.empty:
                districts = sorted(filtered['district'].dropna().unique().tolist())
                # Remove 'Unknown' and invalid entries
                districts = [d for d in districts if d and str(d).strip().lower() not in ['unknown', 'nan', '']]
                if districts:
                    return districts
    except Exception as e:
        print(f"[WARN] Error loading districts from consolidated CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to mapping
    if state in STATE_DISTRICT_MAPPING:
        districts = STATE_DISTRICT_MAPPING[state]
        districts = [d for d in districts if d and d.lower() != 'unknown']
        return districts
    
    # Try case-insensitive search in mapping
    for s, districts in STATE_DISTRICT_MAPPING.items():
        if s.lower() == state.lower():
            districts = [d for d in districts if d and d.lower() != 'unknown']
            return districts
    
    # Fallback to original CSV
    try:
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            df = pd.read_csv(data_file, usecols=['state', 'district'], low_memory=False)
            filtered = df[df['state'].str.strip().str.lower() == state.lower()]
            if not filtered.empty:
                districts = sorted(filtered['district'].dropna().unique().tolist())
                districts = [d for d in districts if d and d.lower() != 'unknown']
                return districts
    except Exception as e:
        print(f"[WARN] Error loading districts from CSV: {e}")
    
    return []

def get_available_states():
    """
    Get list of all available states from training data (consolidated)
    Only returns states that actually have training data
    
    Returns:
        List of available state names (sorted)
    """
    # Try consolidated data first (preferred) - THIS IS THE PRIMARY SOURCE
    try:
        data_file = Path('data/combined/all_sources_consolidated.csv')
        if data_file.exists():
            # Read only state column for performance
            df = pd.read_csv(data_file, usecols=['state'], low_memory=False)
            states = sorted(df['state'].dropna().unique().tolist())
            # Remove 'Unknown' and invalid entries
            states = [s for s in states if s and str(s).lower() != 'unknown' and str(s).lower() != 'nan']
            if states:
                return states
    except Exception as e:
        print(f"[WARN] Error loading states from consolidated CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to original combined data
    if STATE_DISTRICT_MAPPING:
        states = sorted(STATE_DISTRICT_MAPPING.keys())
        # Remove 'Unknown' if present
        states = [s for s in states if s and s.lower() != 'unknown']
        return states
    
    # Try loading from original CSV
    try:
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            df = pd.read_csv(data_file, usecols=['state'], low_memory=False)
            states = sorted(df['state'].dropna().unique().tolist())
            # Remove 'Unknown' if present
            states = [s for s in states if s and s.lower() != 'unknown']
            return states
    except Exception as e:
        print(f"[WARN] Error loading states from CSV: {e}")
    
    return []

def get_location_info(state, district):
    """
    Get detailed information about a location
    
    Args:
        state: State name
        district: District name
    
    Returns:
        Dictionary with location information
    """
    commodities = get_available_commodities(state, district)
    
    info = {
        'state': state,
        'district': district,
        'commodity_count': len(commodities),
        'commodities': commodities,
        'has_location_specific': len(commodities) > 0,
        'location_specific_count': len(commodities)
    }
    
    return info

def is_valid_combination(state, district, crop):
    """
    Check if a state-district-crop combination exists in training data
    Uses the same logic as get_available_commodities to ensure consistency
    
    Args:
        state: State name
        district: District name
        crop: Crop/commodity name
    
    Returns:
        True if combination exists, False otherwise
    """
    crop = str(crop).strip()
    if not crop or crop.lower() in ['unknown', 'nan', '']:
        return False
    
    # Check consolidated data directly (same as get_available_commodities)
    try:
        data_file = Path('data/combined/all_sources_consolidated.csv')
        if data_file.exists():
            # Read only needed columns for performance
            df = pd.read_csv(data_file, usecols=['state', 'district', 'crop'], low_memory=False)
            # Filter out Unknown entries
            df = df[(df['state'] != 'Unknown') & (df['district'] != 'Unknown') & (df['crop'] != 'Unknown')]
            df = df.dropna(subset=['state', 'district', 'crop'])
            
            # Try exact match first (case-insensitive) with original names
            state_lower = str(state).strip().lower()
            district_lower = str(district).strip().lower()
            crop_lower = crop.lower()
            
            filtered = df[
                (df['state'].astype(str).str.strip().str.lower() == state_lower) &
                (df['district'].astype(str).str.strip().str.lower() == district_lower) &
                (df['crop'].astype(str).str.strip().str.lower() == crop_lower)
            ]
            
            if not filtered.empty:
                return True
            
            # If no exact match, try with normalized names
            normalized_state, normalized_district = normalize_location(state, district)
            if normalized_state and normalized_district:
                filtered = df[
                    (df['state'].astype(str).str.strip().str.lower() == normalized_state.strip().lower()) &
                    (df['district'].astype(str).str.strip().str.lower() == normalized_district.strip().lower()) &
                    (df['crop'].astype(str).str.strip().str.lower() == crop_lower)
                ]
                
                if not filtered.empty:
                    return True
    except Exception as e:
        logger.debug(f"Error checking combination in consolidated CSV: {e}")
    
    # Fallback: Use get_available_commodities and check if crop is in the list
    # This ensures consistency with the dropdown
    try:
        commodities = get_available_commodities(state, district)
        # Case-insensitive match
        for c in commodities:
            if str(c).strip().lower() == crop_lower:
                return True
    except Exception as e:
        logger.debug(f"Error checking combination via get_available_commodities: {e}")
    
    return False

def reload_mappings():
    """Reload mappings from files (useful if data is updated)"""
    load_mappings()

