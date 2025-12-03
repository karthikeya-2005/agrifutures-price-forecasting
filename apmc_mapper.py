"""
APMC to District/State Mapper
Fetches APMC information from e-NAM and maps each APMC to its corresponding district and state.
Uses geocoding and fuzzy matching to find locations when APMC names don't explicitly contain district/state info.
"""
import requests
import pandas as pd
import json
import re
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import logging
from pathlib import Path
from cache_manager import get_cache_manager
from location_normalizer import normalize_state_name, normalize_district_name, get_all_states, get_all_districts
from enam_fetcher import EnamFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to store APMC mapping
APMC_MAPPING_FILE = Path('data/apmc_mapping.json')

class APMCMapper:
    """Maps APMC names to their corresponding districts and states"""
    
    def __init__(self):
        self.mapping: Dict[str, Dict] = {}
        self.enam_fetcher = EnamFetcher()
        self.cache_manager = get_cache_manager()
        self._load_mapping()
    
    def _load_mapping(self):
        """Load existing APMC mapping from file"""
        if APMC_MAPPING_FILE.exists():
            try:
                with open(APMC_MAPPING_FILE, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
                logger.info(f"Loaded {len(self.mapping)} APMC mappings from cache")
            except Exception as e:
                logger.warning(f"Failed to load APMC mapping: {e}")
                self.mapping = {}
        else:
            self.mapping = {}
    
    def _save_mapping(self):
        """Save APMC mapping to file"""
        try:
            APMC_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(APMC_MAPPING_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.mapping)} APMC mappings to {APMC_MAPPING_FILE}")
        except Exception as e:
            logger.error(f"Failed to save APMC mapping: {e}")
    
    def fetch_apmcs_from_enam(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch all APMCs from e-NAM by getting data from all endpoints
        Returns DataFrame with APMC, state, district columns
        """
        logger.info("Fetching APMC list from e-NAM...")
        all_apmcs = []
        
        try:
            # Fetch data from e-NAM (without filters to get all APMCs)
            df = self.enam_fetcher.fetch_live_prices()
            
            if df is not None and not df.empty:
                # Extract unique APMC-state-district combinations
                if 'apmc' in df.columns or 'district' in df.columns:
                    # If APMC column exists, use it
                    if 'apmc' in df.columns:
                        apmc_col = 'apmc'
                    else:
                        # If district column contains APMC names, use it
                        apmc_col = 'district'
                    
                    # Get unique combinations
                    if 'state' in df.columns:
                        unique_combos = df[[apmc_col, 'state']].drop_duplicates()
                        for _, row in unique_combos.iterrows():
                            apmc_name = str(row[apmc_col]).strip()
                            state_name = str(row['state']).strip() if pd.notna(row['state']) else None
                            all_apmcs.append({
                                'apmc': apmc_name,
                                'state': state_name,
                                'district': None  # Will be mapped later
                            })
                    else:
                        # Only APMC names available
                        unique_apmcs = df[apmc_col].drop_duplicates()
                        for apmc in unique_apmcs:
                            if pd.notna(apmc):
                                all_apmcs.append({
                                    'apmc': str(apmc).strip(),
                                    'state': None,
                                    'district': None
                                })
                
                logger.info(f"Found {len(all_apmcs)} unique APMCs from e-NAM")
                return pd.DataFrame(all_apmcs)
            
        except Exception as e:
            logger.warning(f"Error fetching APMCs from e-NAM: {e}")
        
        return pd.DataFrame(all_apmcs)
    
    def _geocode_apmc(self, apmc_name: str) -> Optional[Tuple[str, str]]:
        """
        Use geocoding to find district and state for an APMC
        Returns (district, state) or None
        """
        # Check cache first
        cached = self.cache_manager.get('apmc_geocode', apmc_name)
        if cached is not None:
            return cached
        
        try:
            base_url = "https://nominatim.openstreetmap.org/search"
            # Try with "APMC" suffix first
            query = f"{apmc_name} APMC, India"
            params = {
                "q": query,
                "format": "json",
                "limit": 5,
                "addressdetails": 1
            }
            
            response = requests.get(
                base_url, 
                params=params, 
                timeout=10, 
                headers={"User-Agent": "AgrifuturesApp/1.0"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    # Try to extract district and state from address
                    for result in data:
                        address = result.get('address', {})
                        district = None
                        state = None
                        
                        # Try different address components
                        if 'city_district' in address:
                            district = address['city_district']
                        elif 'county' in address:
                            district = address['county']
                        elif 'municipality' in address:
                            district = address['municipality']
                        
                        if 'state' in address:
                            state = address['state']
                        elif 'region' in address:
                            state = address['region']
                        
                        if district and state:
                            # Normalize the names
                            normalized_state = normalize_state_name(state)
                            normalized_district = normalize_district_name(district, normalized_state)
                            
                            if normalized_state and normalized_district:
                                result_tuple = (normalized_district, normalized_state)
                                self.cache_manager.set('apmc_geocode', result_tuple, apmc_name)
                                return result_tuple
                    
                    # If no exact match, try without "APMC" suffix
                    query2 = f"{apmc_name}, India"
                    params['q'] = query2
                    response2 = requests.get(
                        base_url, 
                        params=params, 
                        timeout=10, 
                        headers={"User-Agent": "AgrifuturesApp/1.0"}
                    )
                    
                    if response2.status_code == 200:
                        data2 = response2.json()
                        if data2:
                            for result in data2:
                                address = result.get('address', {})
                                district = address.get('city_district') or address.get('county') or address.get('municipality')
                                state = address.get('state') or address.get('region')
                                
                                if district and state:
                                    normalized_state = normalize_state_name(state)
                                    normalized_district = normalize_district_name(district, normalized_state)
                                    
                                    if normalized_state and normalized_district:
                                        result_tuple = (normalized_district, normalized_state)
                                        self.cache_manager.set('apmc_geocode', result_tuple, apmc_name)
                                        return result_tuple
        except Exception as e:
            logger.debug(f"Geocoding failed for {apmc_name}: {e}")
        
        # Cache None result
        self.cache_manager.set('apmc_geocode', None, ttl=86400, apmc_name=apmc_name)
        return None
    
    def _extract_location_from_apmc_name(self, apmc_name: str) -> Optional[Tuple[str, str]]:
        """
        Try to extract district and state from APMC name
        Many APMCs are named like "District Name APMC" or "City Name APMC"
        """
        apmc_clean = apmc_name.strip()
        
        # Remove common suffixes
        apmc_clean = re.sub(r'\s*(APMC|Market|Mandi|Yard)\s*$', '', apmc_clean, flags=re.IGNORECASE)
        apmc_clean = apmc_clean.strip()
        
        # Try to match with known districts
        all_districts = get_all_districts()
        for district in all_districts:
            district_lower = district.lower()
            apmc_lower = apmc_clean.lower()
            
            # Check if district name is in APMC name
            if district_lower in apmc_lower or apmc_lower in district_lower:
                # Find which state this district belongs to
                all_states = get_all_states()
                for state in all_states:
                    districts_in_state = get_all_districts(state)
                    if district in districts_in_state:
                        return (district, state)
        
        # Try to match with state names
        all_states = get_all_states()
        for state in all_states:
            state_lower = state.lower()
            apmc_lower = apmc_clean.lower()
            
            if state_lower in apmc_lower:
                # Found state, now try to find district
                # Remove state name and try to match remaining with districts
                remaining = re.sub(state_lower, '', apmc_lower, flags=re.IGNORECASE).strip()
                if remaining:
                    districts_in_state = get_all_districts(state)
                    for district in districts_in_state:
                        if remaining in district.lower() or district.lower() in remaining:
                            return (district, state)
                
                # If no district match, return state only (district will be None)
                return (None, state)
        
        return None
    
    def map_apmc(self, apmc_name: str, state: Optional[str] = None, district: Optional[str] = None) -> Dict:
        """
        Map an APMC to its district and state
        Returns dict with 'district', 'state', 'confidence' keys
        """
        if not apmc_name:
            return {'district': None, 'state': None, 'confidence': 'none'}
        
        apmc_name = str(apmc_name).strip()
        
        # Check if already mapped
        if apmc_name in self.mapping:
            return self.mapping[apmc_name]
        
        # If state and district are already provided, use them
        if state and district:
            normalized_state = normalize_state_name(state)
            normalized_district = normalize_district_name(district, normalized_state)
            
            if normalized_state and normalized_district:
                result = {
                    'district': normalized_district,
                    'state': normalized_state,
                    'confidence': 'provided'
                }
                self.mapping[apmc_name] = result
                self._save_mapping()
                return result
        
        # Strategy 1: Extract from APMC name
        location = self._extract_location_from_apmc_name(apmc_name)
        if location:
            district_found, state_found = location
            if state_found:
                result = {
                    'district': district_found,
                    'state': state_found,
                    'confidence': 'name_extraction'
                }
                self.mapping[apmc_name] = result
                self._save_mapping()
                return result
        
        # Strategy 2: If state is provided, search districts in that state
        if state:
            normalized_state = normalize_state_name(state)
            if normalized_state:
                districts_in_state = get_all_districts(normalized_state)
                apmc_lower = apmc_name.lower()
                
                for district in districts_in_state:
                    if district.lower() in apmc_lower or apmc_lower in district.lower():
                        result = {
                            'district': district,
                            'state': normalized_state,
                            'confidence': 'state_constrained'
                        }
                        self.mapping[apmc_name] = result
                        self._save_mapping()
                        return result
        
        # Strategy 3: Geocode the APMC
        geocode_result = self._geocode_apmc(apmc_name)
        if geocode_result:
            district_found, state_found = geocode_result
            result = {
                'district': district_found,
                'state': state_found,
                'confidence': 'geocoded'
            }
            self.mapping[apmc_name] = result
            self._save_mapping()
            return result
        
        # Strategy 4: Fuzzy match with all districts
        all_districts = get_all_districts()
        apmc_lower = apmc_name.lower()
        best_match = None
        best_score = 0
        
        for district in all_districts:
            district_lower = district.lower()
            # Simple substring matching score
            if district_lower in apmc_lower:
                score = len(district_lower) / len(apmc_lower)
                if score > best_score:
                    best_score = score
                    districts_in_state = []
                    for state in get_all_states():
                        if district in get_all_districts(state):
                            districts_in_state.append(state)
                    if districts_in_state:
                        best_match = (district, districts_in_state[0])
        
        if best_match and best_score > 0.3:  # Threshold for confidence
            district_found, state_found = best_match
            result = {
                'district': district_found,
                'state': state_found,
                'confidence': 'fuzzy_match'
            }
            self.mapping[apmc_name] = result
            self._save_mapping()
            return result
        
        # No match found
        result = {
            'district': None,
            'state': None,
            'confidence': 'none'
        }
        self.mapping[apmc_name] = result
        self._save_mapping()
        return result
    
    def map_apmc_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map APMCs in a DataFrame to districts and states
        Expects DataFrame with 'apmc' column (or 'district' column containing APMC names)
        Adds/updates 'district' and 'state' columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Determine APMC column
        apmc_col = None
        if 'apmc' in df.columns:
            apmc_col = 'apmc'
        elif 'district' in df.columns:
            # Check if district column contains APMC names
            apmc_col = 'district'
        
        if not apmc_col:
            logger.warning("No APMC column found in DataFrame")
            return df
        
        # Ensure district and state columns exist
        if 'district' not in df.columns:
            df['district'] = None
        if 'state' not in df.columns:
            df['state'] = None
        
        # Map each APMC
        mapped_count = 0
        for idx, row in df.iterrows():
            apmc_name = str(row[apmc_col]).strip() if pd.notna(row[apmc_col]) else None
            if not apmc_name:
                continue
            
            existing_state = str(row['state']).strip() if pd.notna(row['state']) else None
            existing_district = str(row['district']).strip() if pd.notna(row['district']) else None
            
            # Map the APMC
            mapping = self.map_apmc(apmc_name, state=existing_state, district=existing_district)
            
            # Update DataFrame
            if mapping['district']:
                df.at[idx, 'district'] = mapping['district']
            if mapping['state']:
                df.at[idx, 'state'] = mapping['state']
            
            if mapping['district'] or mapping['state']:
                mapped_count += 1
        
        logger.info(f"Mapped {mapped_count}/{len(df)} APMCs to districts/states")
        return df
    
    def build_mapping_from_enam(self, limit: Optional[int] = None):
        """
        Build APMC mapping by fetching all APMCs from e-NAM and mapping them
        """
        logger.info("Building APMC mapping from e-NAM...")
        
        # Fetch APMCs from e-NAM
        apmc_df = self.fetch_apmcs_from_enam(limit=limit)
        
        if apmc_df.empty:
            logger.warning("No APMCs found from e-NAM")
            return
        
        # Map each APMC
        logger.info(f"Mapping {len(apmc_df)} APMCs...")
        mapped_df = self.map_apmc_dataframe(apmc_df)
        
        # Save mapping
        self._save_mapping()
        
        # Report statistics
        mapped_count = mapped_df['district'].notna().sum()
        logger.info(f"Successfully mapped {mapped_count}/{len(mapped_df)} APMCs")
        
        return mapped_df


# Global mapper instance
_mapper_instance = None

def get_apmc_mapper() -> APMCMapper:
    """Get global APMC mapper instance"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = APMCMapper()
    return _mapper_instance

def map_apmc_to_location(apmc_name: str, state: Optional[str] = None, district: Optional[str] = None) -> Dict:
    """Convenience function to map a single APMC"""
    mapper = get_apmc_mapper()
    return mapper.map_apmc(apmc_name, state, district)

