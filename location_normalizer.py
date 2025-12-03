"""
Location Name Normalization Module
Handles variations in Indian state and district names and maps them to standardized names
used in the training data.
"""
import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# State name variations and mappings
STATE_VARIATIONS = {
    # Common abbreviations
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JK': 'Jammu and Kashmir',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OR': 'Odisha',
    'OD': 'Odisha',
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TS': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UK': 'Uttarakhand',
    'WB': 'West Bengal',
    
    # Common variations
    'Tamilnadu': 'Tamil Nadu',
    'Tamil nadu': 'Tamil Nadu',
    'TamilNadu': 'Tamil Nadu',
    'Andhra': 'Andhra Pradesh',
    'Arunachal': 'Arunachal Pradesh',
    'Himachal': 'Himachal Pradesh',
    'Madhya': 'Madhya Pradesh',
    'Uttar': 'Uttar Pradesh',
    'West': 'West Bengal',
    'J&K': 'Jammu and Kashmir',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Jammu and kashmir': 'Jammu and Kashmir',
    'Jammu & kashmir': 'Jammu and Kashmir',
    'Orissa': 'Odisha',
    'Pondicherry': 'Puducherry',
    'Pondichery': 'Puducherry',
}

# District name variations (common ones)
DISTRICT_VARIATIONS = {
    # Common spelling variations
    'Kasargod': 'Kasaragod',
    'Kasargode': 'Kasaragod',
    'Kasaragode': 'Kasaragod',
    'Mewat': 'Nuh',  # Mewat was renamed to Nuh
    'Gurgaon': 'Gurugram',
    'Bangalore': 'Bengaluru Urban',
    'Bangalore Urban': 'Bengaluru Urban',
    'Bangalore Rural': 'Bengaluru Rural',
    'Bangaluru': 'Bengaluru Urban',
    'Calcutta': 'Kolkata',
    'Calicut': 'Kozhikode',
    'Cochin': 'Ernakulam',
    'Trivandrum': 'Thiruvananthapuram',
    'Trichy': 'Tiruchirappalli',
    'Trichirappalli': 'Tiruchirappalli',
    'Madras': 'Chennai',
    'Bombay': 'Mumbai',
    'Poona': 'Pune',
    'Allahabad': 'Prayagraj',
    'Prayag': 'Prayagraj',
    'Benares': 'Varanasi',
    'Banaras': 'Varanasi',
    'Kashi': 'Varanasi',
    'Mysore': 'Mysuru',
    'Mangalore': 'Dakshina Kannada',
    'Mangaluru': 'Dakshina Kannada',
    'Bellary': 'Ballari',
    'Belgaum': 'Belagavi',
    'Bijapur': 'Vijayapura',
    'Gulbarga': 'Kalaburagi',
    'Hubli': 'Dharwad',
    'Shimoga': 'Shivamogga',
    'Tumkur': 'Tumakuru',
    'Chikmagalur': 'Chikkamagaluru',
    'Chikballapur': 'Chikkaballapur',
    'Chitradurga': 'Chitradurga',
    'Raichur': 'Raichur',
    'Kolar': 'Kolar',
    'Mandya': 'Mandya',
    'Hassan': 'Hassan',
    'Davanagere': 'Davanagere',
    'Udupi': 'Udupi',
    'Dharwad': 'Dharwad',
    'Gadag': 'Gadag',
    'Haveri': 'Haveri',
    'Bagalkot': 'Bagalkot',
    'Vijayapura': 'Vijayapura',
    'Ballari': 'Ballari',
    'Belagavi': 'Belagavi',
    'Bidar': 'Bidar',
    'Chamarajanagar': 'Chamarajanagar',
    'Chikkaballapur': 'Chikkaballapur',
    'Chikkamagaluru': 'Chikkamagaluru',
    'Chitradurga': 'Chitradurga',
    'Dakshina Kannada': 'Dakshina Kannada',
    'Davanagere': 'Davanagere',
    'Dharwad': 'Dharwad',
    'Gadag': 'Gadag',
    'Hassan': 'Hassan',
    'Haveri': 'Haveri',
    'Kodagu': 'Kodagu',
    'Kolar': 'Kolar',
    'Koppal': 'Koppal',
    'Mandya': 'Mandya',
    'Mysuru': 'Mysuru',
    'Raichur': 'Raichur',
    'Ramanagara': 'Ramanagara',
    'Shivamogga': 'Shivamogga',
    'Tumakuru': 'Tumakuru',
    'Udupi': 'Udupi',
    'Uttara Kannada': 'Uttara Kannada',
    'Vijayapura': 'Vijayapura',
    'Yadgir': 'Yadgir',
}

# Load actual state/district names from training data
TRAINING_STATES = set()
TRAINING_DISTRICTS = {}
STATE_DISTRICT_MAPPING = {}

def load_training_data_names():
    """Load actual state and district names from training data"""
    global TRAINING_STATES, TRAINING_DISTRICTS, STATE_DISTRICT_MAPPING
    
    try:
        data_file = Path('data/combined/all_sources_combined.csv')
        if data_file.exists():
            import pandas as pd
            df = pd.read_csv(data_file, usecols=['state', 'district'], low_memory=False, nrows=100000)
            
            # Get unique states
            TRAINING_STATES = set(df['state'].dropna().unique())
            
            # Get state-district mapping
            for state in TRAINING_STATES:
                districts = df[df['state'] == state]['district'].dropna().unique().tolist()
                TRAINING_DISTRICTS[state] = set(districts)
                STATE_DISTRICT_MAPPING[state] = districts
    except Exception as e:
        print(f"[WARN] Could not load training data names: {e}")

# Load on import
load_training_data_names()

def normalize_state_name(state: str) -> Optional[str]:
    """
    Normalize state name to match training data
    
    Args:
        state: State name (can be variation, abbreviation, etc.)
    
    Returns:
        Normalized state name or None if not found
    """
    if not state:
        return None
    
    state = str(state).strip()
    
    # Check exact match first (case-insensitive)
    for training_state in TRAINING_STATES:
        if training_state.lower() == state.lower():
            return training_state
    
    # Check variations
    state_upper = state.upper()
    if state_upper in STATE_VARIATIONS:
        normalized = STATE_VARIATIONS[state_upper]
        if normalized in TRAINING_STATES:
            return normalized
    
    # Check case-insensitive variations
    state_lower = state.lower()
    for variation, standard in STATE_VARIATIONS.items():
        if variation.lower() == state_lower:
            if standard in TRAINING_STATES:
                return standard
    
    # Try fuzzy matching (contains)
    state_lower = state.lower()
    for training_state in TRAINING_STATES:
        if state_lower in training_state.lower() or training_state.lower() in state_lower:
            return training_state
    
    # Try removing spaces and matching
    state_no_space = re.sub(r'\s+', '', state.lower())
    for training_state in TRAINING_STATES:
        training_no_space = re.sub(r'\s+', '', training_state.lower())
        if state_no_space == training_no_space:
            return training_state
    
    return None

def normalize_district_name(district: str, state: Optional[str] = None) -> Optional[str]:
    """
    Normalize district name to match training data
    
    Args:
        district: District name (can be variation, old name, etc.)
        state: Optional state name to narrow down search
    
    Returns:
        Normalized district name or None if not found
    """
    if not district:
        return None
    
    district = str(district).strip()
    
    # If state is provided, search only in that state's districts
    districts_to_search = []
    if state:
        normalized_state = normalize_state_name(state)
        if normalized_state and normalized_state in TRAINING_DISTRICTS:
            districts_to_search = list(TRAINING_DISTRICTS[normalized_state])
        else:
            # If state not found, search all districts
            districts_to_search = [d for districts in TRAINING_DISTRICTS.values() for d in districts]
    else:
        # Search all districts
        districts_to_search = [d for districts in TRAINING_DISTRICTS.values() for d in districts]
    
    # Check exact match first (case-insensitive)
    for training_district in districts_to_search:
        if training_district.lower() == district.lower():
            return training_district
    
    # Check variations
    district_lower = district.lower()
    if district in DISTRICT_VARIATIONS:
        normalized = DISTRICT_VARIATIONS[district]
        if normalized in districts_to_search:
            return normalized
    
    # Check case-insensitive variations
    for variation, standard in DISTRICT_VARIATIONS.items():
        if variation.lower() == district_lower:
            if standard in districts_to_search:
                return standard
    
    # Try fuzzy matching (contains)
    for training_district in districts_to_search:
        if district_lower in training_district.lower() or training_district.lower() in district_lower:
            return training_district
    
    # Try removing spaces and special characters
    district_clean = re.sub(r'[^\w]', '', district.lower())
    for training_district in districts_to_search:
        training_clean = re.sub(r'[^\w]', '', training_district.lower())
        if district_clean == training_clean:
            return training_district
    
    return None

def normalize_location(state: str, district: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize both state and district names
    
    Args:
        state: State name
        district: District name
    
    Returns:
        Tuple of (normalized_state, normalized_district) or (None, None) if not found
    """
    normalized_state = normalize_state_name(state)
    normalized_district = normalize_district_name(district, normalized_state)
    
    return normalized_state, normalized_district

def get_all_states() -> List[str]:
    """Get all available states from training data"""
    return sorted(list(TRAINING_STATES))

def get_all_districts(state: Optional[str] = None) -> List[str]:
    """Get all available districts, optionally filtered by state"""
    if state:
        normalized_state = normalize_state_name(state)
        if normalized_state and normalized_state in STATE_DISTRICT_MAPPING:
            return sorted(STATE_DISTRICT_MAPPING[normalized_state])
        return []
    else:
        all_districts = set()
        for districts in TRAINING_DISTRICTS.values():
            all_districts.update(districts)
        return sorted(list(all_districts))

def reload_training_data():
    """Reload training data names (useful if data is updated)"""
    load_training_data_names()

