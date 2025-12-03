"""
Enhanced Predictor with Weather and Current Market Conditions
Fetches current market data and weather, then makes predictions with forecasts
"""

import os
import joblib
import pickle
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocessing import create_features
from geolocation_fetcher import get_coordinates
from weather_data_fetcher import fetch_weather_data
from get_available_commodities import is_valid_combination
from location_normalizer import normalize_state_name, normalize_district_name, normalize_location
from cache_manager import get_cache_manager, cached

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_DIR_ADVANCED = "models_advanced"
MODEL_DIR_BY_COMMODITY = "models_by_commodity"

def _fetch_enam_wrapper(state, crop):
    """Wrapper for e-NAM fetch to use in parallel execution"""
    try:
        from enam_fetcher import fetch_enam_data
        df = fetch_enam_data(state=state, commodity=crop, limit=500)
        if df is not None and not df.empty:
            logger.info(f"Fetched {len(df)} records from e-NAM")
            return ('enam', df)
    except Exception as e:
        logger.debug(f"e-NAM fetch failed: {e}")
    return ('enam', None)

def _fetch_commodityonline_wrapper(state, district, crop):
    """Wrapper for Commodity Online fetch to use in parallel execution"""
    try:
        from commodityonline_fetcher import fetch_commodityonline_data
        df = fetch_commodityonline_data(commodity=crop, state=state, district=district, limit=500)
        if df is not None and not df.empty:
            logger.info(f"Fetched {len(df)} records from Commodity Online")
            return ('commodityonline', df)
    except Exception as e:
        logger.debug(f"Commodity Online fetch failed: {e}")
    return ('commodityonline', None)

def _fetch_ncdex_wrapper(state, district, crop):
    """Wrapper for NCDEX fetch to use in parallel execution"""
    try:
        from ncdex_fetcher import fetch_ncdex_data
        df = fetch_ncdex_data(commodity=crop, limit=500)
        if df is not None and not df.empty:
            # NCDEX data doesn't have state/district, but has good commodity-level prices
            df['state'] = state
            df['district'] = district
            logger.info(f"Fetched {len(df)} records from NCDEX")
            return ('ncdex', df)
    except Exception as e:
        logger.debug(f"NCDEX fetch failed: {e}")
    return ('ncdex', None)

def _fetch_gov_data_wrapper():
    """Wrapper for Data.gov.in fetch to use in parallel execution"""
    try:
        from enhanced_market_data_fetcher import fetch_data_gov_data
        df = fetch_data_gov_data(limit=2000)
        if df is not None and not df.empty:
            logger.info(f"Fetched {len(df)} records from Data.gov.in")
            return ('gov', df)
    except Exception as e:
        logger.debug(f"Data.gov.in fetch failed: {e}")
    return ('gov', None)

def _fetch_agmarknet_wrapper():
    """Wrapper for Agmarknet fetch to use in parallel execution"""
    try:
        from market_data_fetcher import fetch_agmarknet_data
        df = fetch_agmarknet_data(use_api=True)
        if df is not None and not df.empty:
            logger.info(f"Fetched {len(df)} records from Agmarknet")
            return ('agmarknet', df)
    except Exception as e:
        logger.debug(f"Agmarknet fetch failed: {e}")
    return ('agmarknet', None)

def fetch_current_market_conditions(state, district, crop, progress_callback=None, use_previous_week_fallback=True):
    """
    Fetch current market conditions (recent prices) for the commodity
    PRIORITY: e-NAM is the PRIMARY source - tried first with all available methods
    If current data unavailable, falls back to previous week's data for predictions
    
    Args:
        state: State name
        district: District name
        crop: Crop/commodity name
        progress_callback: Optional callback function(status, source) for progress updates
        use_previous_week_fallback: If True, use previous week's data if current data unavailable
    """
    # Check cache first
    cache_manager = get_cache_manager()
    cached_result = cache_manager.get('market_data', state, district, crop)
    if cached_result is not None:
        logger.info("Using cached market data")
        if progress_callback:
            progress_callback("Using cached data", "cache")
        return cached_result
    
    try:
        # PRIORITY 1: Try e-NAM FIRST as PRIMARY source
        # Try e-NAM with all available methods before other sources
        if progress_callback:
            progress_callback("Fetching from e-NAM (PRIMARY source)...", "enam")
        
        logger.info(f"Attempting to fetch from e-NAM (PRIMARY source) for {crop} in {district}, {state}")
        enam_data = None
        
        # Try e-NAM with multiple strategies
        try:
            source_key, enam_data = _fetch_enam_wrapper(state, crop)
            if enam_data is not None and not enam_data.empty:
                logger.info(f"✅ Successfully fetched {len(enam_data)} records from e-NAM (PRIMARY source)")
                # If e-NAM has data, prioritize it and combine with others
                all_data = [enam_data]
            else:
                logger.warning("e-NAM returned no data, trying other sources")
                all_data = []
        except Exception as e:
            logger.warning(f"e-NAM fetch failed: {e}, trying other sources")
            all_data = []
        
        # PRIORITY 2: Try other sources in parallel (only if e-NAM didn't return data or to supplement)
        sources_completed = 1 if enam_data is not None and not enam_data.empty else 0
        total_sources = 5
        
        # Use ThreadPoolExecutor for parallel API calls to other sources
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit other fetch tasks (e-NAM already tried)
            futures = {
                executor.submit(_fetch_commodityonline_wrapper, state, district, crop): 'Commodity Online',
                executor.submit(_fetch_ncdex_wrapper, state, district, crop): 'NCDEX',
                executor.submit(_fetch_gov_data_wrapper): 'Data.gov.in',
                executor.submit(_fetch_agmarknet_wrapper): 'Agmarknet'
            }
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                source_name = futures[future]
                sources_completed += 1
                
                if progress_callback:
                    progress_callback(f"Fetching from {source_name}... ({sources_completed}/{total_sources})", source_name)
                
                try:
                    source_key, df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                        logger.info(f"Successfully fetched data from {source_name}")
                except Exception as e:
                    logger.debug(f"Error in {source_name} fetch: {e}")
        
        # Combine all data sources
        if all_data:
            df = pd.concat(all_data, ignore_index=True).drop_duplicates()
            # Prioritize e-NAM data if available (mark it as primary source)
            if enam_data is not None and not enam_data.empty:
                logger.info("Using e-NAM as PRIMARY data source")
        else:
            # No current data available - try previous week's data as fallback
            if use_previous_week_fallback:
                logger.info("No current data available, attempting to fetch previous week's data...")
                if progress_callback:
                    progress_callback("Fetching previous week's data...", "fallback")
                
                previous_week_data = _fetch_previous_week_data(state, district, crop, progress_callback)
                if previous_week_data:
                    logger.info("✅ Using previous week's data for predictions")
                    # Mark as fallback data
                    previous_week_data['is_fallback'] = True
                    previous_week_data['data_source'] = 'previous_week_fallback'
                    cache_manager.set('market_data', previous_week_data, ttl=3600, state=state, district=district, crop=crop)
                    return previous_week_data
            
            result = None
            # Cache None result for shorter time to allow retries
            cache_manager.set('market_data', None, ttl=300, state=state, district=district, crop=crop)
            return None
        
        if df is None or df.empty:
            return None
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # Ensure price column is numeric
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]  # Remove invalid prices
        
        # Filter for the specific commodity, state, district (case-insensitive)
        filtered = df[
            (df['crop'].str.strip().str.lower() == crop.strip().lower()) &
            (df['state'].str.strip().str.lower() == state.strip().lower()) &
            (df['district'].str.strip().str.lower() == district.strip().lower())
        ].copy()
        
        if not filtered.empty:
            # Get most recent data (last 30 days)
            filtered = filtered.sort_values('date', ascending=False)
            recent = filtered.head(30).copy()
            
            if len(recent) > 0:
                # Ensure price is numeric
                recent['price'] = pd.to_numeric(recent['price'], errors='coerce')
                recent = recent.dropna(subset=['price'])
                
                if len(recent) == 0:
                    return None
                
                # Current price is the most recent (first after sorting descending by date)
                # Ensure we're using the actual most recent price
                current_price = float(recent['price'].iloc[0])
                
                # Calculate 7-day average from the most recent 7 days only
                # Recent is sorted descending, so head(7) gives the 7 most recent days
                if len(recent) >= 7:
                    # Take the 7 most recent days (already sorted descending)
                    avg_7d = float(recent['price'].head(7).mean())
                elif len(recent) > 0:
                    # If less than 7 days available, use all available
                    avg_7d = float(recent['price'].mean())
                else:
                    avg_7d = current_price
                
                # Calculate 30-day average (or available days, max 30)
                if len(recent) > 0:
                    avg_30d = float(recent['price'].mean())
                else:
                    avg_30d = current_price
                
                # Determine trend - compare first (most recent) with last (oldest in recent set)
                if len(recent) >= 2:
                    most_recent = float(recent['price'].iloc[0])
                    oldest_recent = float(recent['price'].iloc[-1])
                    if most_recent > oldest_recent:
                        trend = 'increasing'
                    elif most_recent < oldest_recent:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
                
                # Calculate volatility (standard deviation)
                if len(recent) > 1:
                    volatility = float(recent['price'].std())
                else:
                    volatility = 0.0
                
                # Check if data is from e-NAM (primary source)
                data_source_type = 'location_specific'
                if enam_data is not None and not enam_data.empty:
                    # Check if this location-specific data came from e-NAM
                    enam_filtered = enam_data[
                        (enam_data['crop'].str.strip().str.lower() == crop.strip().lower()) &
                        (enam_data['state'].str.strip().str.lower() == state.strip().lower()) &
                        (enam_data['district'].str.strip().str.lower() == district.strip().lower())
                    ]
                    if not enam_filtered.empty:
                        data_source_type = 'enam_primary'
                        logger.info("✅ Using e-NAM as PRIMARY source for location-specific data")
                
                result = {
                    'current_price': current_price,
                    'min_price': float(recent['min_price'].iloc[0]) if 'min_price' in recent.columns and len(recent) > 0 else None,
                    'max_price': float(recent['max_price'].iloc[0]) if 'max_price' in recent.columns and len(recent) > 0 else None,
                    'avg_price_7d': avg_7d,
                    'avg_price_30d': avg_30d,
                    'price_trend': trend,
                    'volatility': volatility,
                    'recent_prices': recent[['date', 'price']].to_dict('records'),
                    'data_source': data_source_type,
                    'records_count': len(recent),
                    'is_fallback': False
                }
                # Cache the result
                cache_manager.set('market_data', result, state=state, district=district, crop=crop)
                return result
        
        # Fallback: Try to get any recent data for the commodity (state-level)
        commodity_data = df[
            (df['crop'].str.strip().str.lower() == crop.strip().lower()) &
            (df['state'].str.strip().str.lower() == state.strip().lower())
        ].copy()
        
        if not commodity_data.empty:
            commodity_data = commodity_data.sort_values('date', ascending=False)
            recent = commodity_data.head(30).copy()
            
            if len(recent) > 0:
                # Ensure price is numeric
                recent['price'] = pd.to_numeric(recent['price'], errors='coerce')
                recent = recent.dropna(subset=['price'])
                
                if len(recent) == 0:
                    # Try final fallback
                    pass
                else:
                    current_price = float(recent['price'].iloc[0])
                    
                    # Calculate 7-day average from the most recent 7 days
                    if len(recent) >= 7:
                        avg_7d = float(recent['price'].head(7).mean())
                    else:
                        avg_7d = float(recent['price'].mean())
                    
                    avg_30d = float(recent['price'].mean())
                    
                    # Determine trend
                    if len(recent) >= 2:
                        most_recent = float(recent['price'].iloc[0])
                        oldest_recent = float(recent['price'].iloc[-1])
                        if most_recent > oldest_recent:
                            trend = 'increasing'
                        elif most_recent < oldest_recent:
                            trend = 'decreasing'
                        else:
                            trend = 'stable'
                    else:
                        trend = 'stable'
                    
                    volatility = float(recent['price'].std()) if len(recent) > 1 else 0.0
                
                # Check if data is from e-NAM (primary source)
                data_source_type = 'state_level'
                if enam_data is not None and not enam_data.empty:
                    enam_filtered = enam_data[
                        (enam_data['crop'].str.strip().str.lower() == crop.strip().lower()) &
                        (enam_data['state'].str.strip().str.lower() == state.strip().lower())
                    ]
                    if not enam_filtered.empty:
                        data_source_type = 'enam_primary'
                        logger.info("✅ Using e-NAM as PRIMARY source for state-level data")
                
                result = {
                    'current_price': current_price,
                    'min_price': float(recent['min_price'].iloc[0]) if 'min_price' in recent.columns and len(recent) > 0 else None,
                    'max_price': float(recent['max_price'].iloc[0]) if 'max_price' in recent.columns and len(recent) > 0 else None,
                    'avg_price_7d': avg_7d,
                    'avg_price_30d': avg_30d,
                    'price_trend': trend,
                    'volatility': volatility,
                    'recent_prices': recent[['date', 'price']].to_dict('records'),
                    'data_source': data_source_type,
                    'records_count': len(recent),
                    'is_fallback': False
                }
                # Cache the result
                cache_manager.set('market_data', result, state=state, district=district, crop=crop)
                return result
        
        # Final fallback: commodity-level data with error handling
        try:
            commodity_only = df[df['crop'].astype(str).str.strip().str.lower() == crop.strip().lower()].copy()
            if not commodity_only.empty:
                commodity_only = commodity_only.sort_values('date', ascending=False)
                recent = commodity_only.head(30)
                
                if len(recent) > 0:
                    try:
                        current_price = float(recent['price'].iloc[0])
                        if current_price <= 0 or pd.isna(current_price):
                            return None
                        
                        avg_7d = float(recent['price'].head(7).mean()) if len(recent) >= 7 else float(recent['price'].mean())
                        avg_30d = float(recent['price'].mean())
                        
                        if pd.isna(avg_7d) or pd.isna(avg_30d):
                            return None
                        
                        trend = 'stable'
                        volatility = float(recent['price'].std()) if len(recent) > 1 else 0.0
                        if pd.isna(volatility):
                            volatility = 0.0
                        
                        # Get min/max with error handling
                        min_price = None
                        max_price = None
                        try:
                            if 'min_price' in recent.columns and len(recent) > 0:
                                min_val = recent['min_price'].iloc[0]
                                if pd.notna(min_val):
                                    min_price = float(min_val)
                        except Exception:
                            pass
                        
                        try:
                            if 'max_price' in recent.columns and len(recent) > 0:
                                max_val = recent['max_price'].iloc[0]
                                if pd.notna(max_val):
                                    max_price = float(max_val)
                        except Exception:
                            pass
                        
                        # Get recent prices
                        recent_prices = []
                        try:
                            if 'date' in recent.columns and 'price' in recent.columns:
                                recent_prices = recent[['date', 'price']].to_dict('records')
                        except Exception:
                            pass
                        
                        # Check if data is from e-NAM (primary source)
                        data_source_type = 'commodity_level'
                        if enam_data is not None and not enam_data.empty:
                            enam_filtered = enam_data[
                                enam_data['crop'].str.strip().str.lower() == crop.strip().lower()
                            ]
                            if not enam_filtered.empty:
                                data_source_type = 'enam_primary'
                                logger.info("✅ Using e-NAM as PRIMARY source for commodity-level data")
                        
                        result = {
                            'current_price': current_price,
                            'min_price': min_price,
                            'max_price': max_price,
                            'avg_price_7d': avg_7d,
                            'avg_price_30d': avg_30d,
                            'price_trend': trend,
                            'volatility': volatility,
                            'recent_prices': recent_prices,
                            'data_source': data_source_type,
                            'records_count': len(recent),
                            'is_fallback': False
                        }
                        # Cache the result
                        cache_manager.set('market_data', result, state=state, district=district, crop=crop)
                        return result
                    except (ValueError, IndexError, KeyError) as e:
                        logger.warning(f"Error processing commodity-level data: {e}")
                        return None
        except Exception as e:
            logger.warning(f"Error in commodity-level fallback: {e}")
                
    except Exception as e:
        logger.error(f"Error fetching current market conditions: {e}", exc_info=True)
    
    # Final fallback: Try previous week's data if current data unavailable
    if use_previous_week_fallback:
        logger.info("Attempting to fetch previous week's data as final fallback...")
        if progress_callback:
            progress_callback("Fetching previous week's data (fallback)...", "fallback")
        
        previous_week_data = _fetch_previous_week_data(state, district, crop, progress_callback)
        if previous_week_data:
            logger.info("✅ Using previous week's data for predictions (fallback)")
            previous_week_data['is_fallback'] = True
            previous_week_data['data_source'] = 'previous_week_fallback'
            cache_manager.set('market_data', previous_week_data, ttl=3600, state=state, district=district, crop=crop)
            return previous_week_data
    
    # Cache None result for shorter time to allow retries
    cache_manager.set('market_data', None, ttl=300, state=state, district=district, crop=crop)
    return None

def _fetch_previous_week_data(state, district, crop, progress_callback=None):
    """
    Fetch previous week's market data as fallback when current data is unavailable
    PRIORITY: Tries e-NAM first, then other sources, for data from 7-14 days ago
    
    Args:
        state: State name
        district: District name
        crop: Crop/commodity name
        progress_callback: Optional callback function
    
    Returns:
        Dictionary with market conditions from previous week, or None if unavailable
    """
    from datetime import timedelta
    
    try:
        today = datetime.now().date()
        week_ago_start = today - timedelta(days=14)  # 2 weeks ago
        week_ago_end = today - timedelta(days=7)     # 1 week ago
        
        logger.info(f"Fetching previous week's data ({week_ago_start} to {week_ago_end})")
        
        all_data = []
        
        # PRIORITY: Try e-NAM first for previous week's data
        if progress_callback:
            progress_callback("Fetching previous week from e-NAM (PRIMARY)...", "enam_fallback")
        
        try:
            from enam_fetcher import fetch_enam_data
            # Try to fetch with date range for previous week
            enam_df = fetch_enam_data(
                state=state,
                commodity=crop,
                limit=500
            )
            
            if enam_df is not None and not enam_df.empty:
                # Filter for previous week's dates
                if 'date' in enam_df.columns:
                    enam_df['date'] = pd.to_datetime(enam_df['date'], errors='coerce')
                    previous_week_enam = enam_df[
                        (enam_df['date'] >= pd.Timestamp(week_ago_start)) &
                        (enam_df['date'] <= pd.Timestamp(week_ago_end))
                    ].copy()
                    
                    if not previous_week_enam.empty:
                        all_data.append(previous_week_enam)
                        logger.info(f"✅ Found {len(previous_week_enam)} records from e-NAM for previous week")
        except Exception as e:
            logger.debug(f"e-NAM previous week fetch failed: {e}")
        
        # Try other sources for previous week's data
        if progress_callback:
            progress_callback("Fetching previous week from other sources...", "fallback")
        
        # Try to fetch from all sources and filter for previous week
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_fetch_commodityonline_wrapper, state, district, crop): 'Commodity Online',
                executor.submit(_fetch_ncdex_wrapper, state, district, crop): 'NCDEX',
                executor.submit(_fetch_gov_data_wrapper): 'Data.gov.in',
                executor.submit(_fetch_agmarknet_wrapper): 'Agmarknet'
            }
            
            for future in as_completed(futures):
                try:
                    source_key, df = future.result()
                    if df is not None and not df.empty and 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        previous_week_df = df[
                            (df['date'] >= pd.Timestamp(week_ago_start)) &
                            (df['date'] <= pd.Timestamp(week_ago_end))
                        ].copy()
                        
                        if not previous_week_df.empty:
                            all_data.append(previous_week_df)
                            logger.info(f"Found {len(previous_week_df)} records from {futures[future]} for previous week")
                except Exception as e:
                    logger.debug(f"Error fetching previous week data: {e}")
        
        # Combine and process previous week's data
        if all_data:
            df = pd.concat(all_data, ignore_index=True).drop_duplicates()
            
            # Filter for location and commodity
            filtered = df[
                (df['crop'].str.strip().str.lower() == crop.strip().lower()) &
                (df['state'].str.strip().str.lower() == state.strip().lower()) &
                (df['district'].str.strip().str.lower() == district.strip().lower())
            ].copy()
            
            # If no location-specific data, try state-level
            if filtered.empty:
                filtered = df[
                    (df['crop'].str.strip().str.lower() == crop.strip().lower()) &
                    (df['state'].str.strip().str.lower() == state.strip().lower())
                ].copy()
            
            # If still empty, use commodity-level
            if filtered.empty:
                filtered = df[df['crop'].str.strip().str.lower() == crop.strip().lower()].copy()
            
            if not filtered.empty:
                # Sort by date and get most recent from previous week
                filtered = filtered.sort_values('date', ascending=False)
                recent = filtered.head(7).copy()  # Get up to 7 days from previous week
                
                if len(recent) > 0:
                    recent['price'] = pd.to_numeric(recent['price'], errors='coerce')
                    recent = recent.dropna(subset=['price'])
                    
                    if len(recent) > 0:
                        # Use the most recent price from previous week as "current"
                        current_price = float(recent['price'].iloc[0])
                        avg_7d = float(recent['price'].mean())
                        avg_30d = avg_7d  # Use same for 30d if only have week's data
                        
                        trend = 'stable'
                        if len(recent) >= 2:
                            most_recent = float(recent['price'].iloc[0])
                            oldest = float(recent['price'].iloc[-1])
                            if most_recent > oldest:
                                trend = 'increasing'
                            elif most_recent < oldest:
                                trend = 'decreasing'
                        
                        volatility = float(recent['price'].std()) if len(recent) > 1 else 0.0
                        
                        result = {
                            'current_price': current_price,
                            'min_price': float(recent['min_price'].iloc[0]) if 'min_price' in recent.columns and len(recent) > 0 else None,
                            'max_price': float(recent['max_price'].iloc[0]) if 'max_price' in recent.columns and len(recent) > 0 else None,
                            'avg_price_7d': avg_7d,
                            'avg_price_30d': avg_30d,
                            'price_trend': trend,
                            'volatility': volatility,
                            'recent_prices': recent[['date', 'price']].to_dict('records'),
                            'data_source': 'previous_week_fallback',
                            'records_count': len(recent),
                            'is_fallback': True,
                            'fallback_date_range': f"{week_ago_start} to {week_ago_end}"
                        }
                        
                        logger.info(f"✅ Successfully fetched previous week's data: {len(recent)} records")
                        return result
        
        logger.warning("No previous week's data available from any source")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching previous week's data: {e}", exc_info=True)
        return None

def load_model(state, district, crop):
    """
    Load model with fallback strategy
    Prioritizes models trained with weather data
    """
    # Strategy 0: Try weather-enriched models first (best models)
    weather_model_dir = Path('models/with_weather')
    if weather_model_dir.exists():
        # Try consolidated weather model
        consolidated_model = weather_model_dir / 'xgboost.pkl'  # or lightgbm.pkl
        if consolidated_model.exists():
            try:
                model = joblib.load(consolidated_model)
                metadata_file = weather_model_dir / 'metadata.json'
                metadata = None
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                logger.info("Using weather-enriched model")
                return model, metadata
            except Exception as e:
                logger.debug(f"Error loading weather-enriched model: {e}")
    
    # Strategy 1: Try advanced models (commodity-level)
    crop_clean = crop.replace('/', '_').replace('\\', '_')
    advanced_model_dir = Path(MODEL_DIR_ADVANCED) / crop_clean
    advanced_model_file = advanced_model_dir / 'model.pkl'
    
    if advanced_model_file.exists():
        try:
            with open(advanced_model_file, 'rb') as f:
                model = pickle.load(f)
            metadata_file = advanced_model_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            return model, None
        except Exception as e:
            logger.debug(f"Error loading advanced model: {e}")
    
    # Strategy 2: Try commodity-location models
    location_clean = f"{state}_{district}".replace('/', '_').replace('\\', '_')
    commodity_model_dir = Path(MODEL_DIR_BY_COMMODITY) / crop_clean / location_clean
    commodity_model_file = commodity_model_dir / 'model.pkl'
    
    if commodity_model_file.exists():
        try:
            with open(commodity_model_file, 'rb') as f:
                model = pickle.load(f)
            metadata_file = commodity_model_dir / 'metadata.json'
            metadata = None
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            return model, metadata
        except Exception as e:
            logger.debug(f"Error loading commodity-location model: {e}")
    
    # Strategy 3: Try ALL_LOCATIONS models
    all_locations_dir = Path(MODEL_DIR_BY_COMMODITY) / crop_clean / 'ALL_LOCATIONS'
    all_locations_file = all_locations_dir / 'model.pkl'
    
    if all_locations_file.exists():
        try:
            with open(all_locations_file, 'rb') as f:
                model = pickle.load(f)
            metadata_file = all_locations_dir / 'metadata.json'
            metadata = None
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            return model, metadata
        except Exception as e:
            logger.debug(f"Error loading ALL_LOCATIONS model: {e}")
    
    # Strategy 4: Try old format
    old_model_path = os.path.join(MODEL_DIR, f"{state}_{district}_{crop}.joblib")
    if os.path.exists(old_model_path):
        try:
            model = joblib.load(old_model_path)
            return model, None
        except Exception as e:
            logger.debug(f"Error loading old format model: {e}")
    
    return None, None

def predict_with_forecast(state, district, crop, target_date, days_ahead=7, progress_callback=None):
    """
    Predict price with forecast for multiple days ahead (max 7 days)
    Returns predictions and current market conditions
    
    Args:
        state: State name
        district: District name
        crop: Crop/commodity name
        target_date: Target date for prediction
        days_ahead: Number of days to forecast (max 7)
        progress_callback: Optional callback function(status, step) for progress updates
    """
    # Enforce maximum of 7 days for forecast horizon
    days_ahead = min(days_ahead, 7)
    # Try to normalize state and district names, but use original if normalization fails
    # This allows predictions to work even if location_normalizer doesn't have all locations
    normalized_state, normalized_district = normalize_location(state, district)
    
    # Use normalized names if available, otherwise use original names
    # The data matching functions handle both cases
    if normalized_state:
        state_for_prediction = normalized_state
    else:
        state_for_prediction = state
    
    if normalized_district:
        district_for_prediction = normalized_district
    else:
        district_for_prediction = district
    
    # Validate input combination (try both normalized and original)
    is_valid = is_valid_combination(state_for_prediction, district_for_prediction, crop)
    if not is_valid:
        # Try with original names
        is_valid = is_valid_combination(state, district, crop)
        if is_valid:
            # Use original names if validation passes with them
            state_for_prediction = state
            district_for_prediction = district
    
    # Don't fail completely if validation fails - the crop came from dropdown which only shows valid crops
    # Silently proceed - dropdown ensures only valid combinations are shown
    # Log at debug level only for internal tracking
    if not is_valid:
        logger.debug(f"Validation check failed for {crop} in {district}, {state}, but continuing (dropdown ensures validity)")
    
    # Use the names that work (normalized if available, otherwise original)
    state = state_for_prediction
    district = district_for_prediction
    
    # Fetch current market conditions with error handling
    if progress_callback:
        progress_callback("Fetching market data from multiple sources...", "market_data")
    try:
        market_conditions = fetch_current_market_conditions(state, district, crop, progress_callback=progress_callback)
    except Exception as e:
        logger.warning(f"Error fetching market conditions: {e}")
        market_conditions = None
    
    # Get coordinates for weather with error handling
    if progress_callback:
        progress_callback("Getting location coordinates...", "coordinates")
    try:
        lat, lon = get_coordinates(state, district)
    except Exception as e:
        logger.warning(f"Error getting coordinates: {e}")
        lat, lon = None, None
    
    # OPTIMIZATION: Fetch weather data once for the entire forecast period (max 7 days)
    # This reduces API calls from N calls to 1 call
    current_weather = None
    forecast_weather_data = None
    
    if lat and lon:
        if progress_callback:
            progress_callback("Fetching weather data...", "weather")
        try:
            # Check cache for weather data
            cache_manager = get_cache_manager()
            today = datetime.now().date()
            max_forecast_date = min(target_date + timedelta(days=days_ahead - 1), today + timedelta(days=7))
            start_date = today.strftime('%Y-%m-%d')
            end_date = max_forecast_date.strftime('%Y-%m-%d')
            
            # Try cache first
            cached_weather = cache_manager.get('weather_data', lat, lon, start_date, end_date)
            if cached_weather is not None:
                weather_data = cached_weather
            else:
                # Single API call for entire forecast period
                weather_data = fetch_weather_data(lat, lon, start_date, end_date)
                # Cache the result
                if weather_data:
                    cache_manager.set('weather_data', weather_data, lat=lat, lon=lon, start_date=start_date, end_date=end_date)
            
            if weather_data and 'daily' in weather_data:
                daily = weather_data['daily']
                forecast_weather_data = daily  # Store for reuse
                
                # Use most recent weather data (current conditions)
                if 'temperature_2m_max' in daily and len(daily['temperature_2m_max']) > 0:
                    current_weather = {
                        'temperature_2m_max': daily['temperature_2m_max'][-1] if daily['temperature_2m_max'] else 25.0,
                        'temperature_2m_min': daily['temperature_2m_min'][-1] if 'temperature_2m_min' in daily and daily['temperature_2m_min'] else 20.0,
                        'precipitation_sum': daily['precipitation_sum'][-1] if 'precipitation_sum' in daily and daily['precipitation_sum'] else 0.0
                    }
        except Exception as e:
            logger.warning(f"Could not fetch weather: {e}")
    
    # Generate forecast for multiple days (max 7)
    if progress_callback:
        progress_callback("Loading prediction model...", "model")
    forecast_dates = [target_date + timedelta(days=i) for i in range(days_ahead)]
    
    predictions = []
    current_date = datetime.now().date()
    
    # OPTIMIZATION: Reuse weather data instead of making individual API calls
    total_days = len(forecast_dates)
    for i, forecast_date in enumerate(forecast_dates):
        if progress_callback:
            progress_callback(f"Generating predictions... ({i+1}/{total_days})", "prediction")
        # Use cached weather data if available
        weather_features = current_weather
        
        # For future dates, try to get forecast weather from cached data
        if forecast_weather_data and forecast_date > current_date:
            try:
                # Calculate days from today
                days_from_today = (forecast_date - current_date).days
                if days_from_today >= 0 and days_from_today < len(forecast_weather_data.get('temperature_2m_max', [])):
                    weather_features = {
                        'temperature_2m_max': forecast_weather_data['temperature_2m_max'][days_from_today] if 'temperature_2m_max' in forecast_weather_data else 25.0,
                        'temperature_2m_min': forecast_weather_data['temperature_2m_min'][days_from_today] if 'temperature_2m_min' in forecast_weather_data else 20.0,
                        'precipitation_sum': forecast_weather_data['precipitation_sum'][days_from_today] if 'precipitation_sum' in forecast_weather_data else 0.0
                    }
            except:
                # Fallback to current weather if forecast weather fails
                weather_features = current_weather
        
        # Prepare input features
        input_features = pd.DataFrame({
            'date': pd.to_datetime([forecast_date]),
            'state': [state],
            'district': [district],
            'crop': [crop]
        })
        
        # Add market conditions to features if available
        # For future dates, use projected values based on previous predictions
        if market_conditions:
            try:
                days_ahead = (forecast_date - current_date).days
                
                # For current date or near future, use actual market conditions
                if days_ahead <= 0:
                    # Today or past - use actual current price
                    if market_conditions.get('current_price'):
                        input_features['current_price'] = float(market_conditions['current_price'])
                    if market_conditions.get('avg_price_7d'):
                        input_features['price_ma_7'] = float(market_conditions['avg_price_7d'])
                    if market_conditions.get('avg_price_30d'):
                        input_features['price_ma_30'] = float(market_conditions['avg_price_30d'])
                else:
                    # For future dates, use projected values
                    # Use the most recent prediction as the "current" price for this date
                    if len(predictions) > 0:
                        # Use the last predicted price as baseline
                        projected_price = predictions[-1]['price']
                        input_features['current_price'] = float(projected_price)
                        
                        # For moving averages, blend actual with projected
                        if market_conditions.get('avg_price_7d'):
                            # Gradually shift from actual to projected
                            actual_7d = float(market_conditions['avg_price_7d'])
                            weight = max(0.3, 1.0 - (days_ahead * 0.1))  # Decay factor
                            input_features['price_ma_7'] = (weight * actual_7d) + ((1 - weight) * projected_price)
                        else:
                            input_features['price_ma_7'] = float(projected_price)
                            
                        if market_conditions.get('avg_price_30d'):
                            actual_30d = float(market_conditions['avg_price_30d'])
                            weight = max(0.5, 1.0 - (days_ahead * 0.05))  # Slower decay for 30d
                            input_features['price_ma_30'] = (weight * actual_30d) + ((1 - weight) * projected_price)
                        else:
                            input_features['price_ma_30'] = float(projected_price)
                    else:
                        # First future prediction - use current market conditions
                        if market_conditions.get('current_price'):
                            input_features['current_price'] = float(market_conditions['current_price'])
                        if market_conditions.get('avg_price_7d'):
                            input_features['price_ma_7'] = float(market_conditions['avg_price_7d'])
                        if market_conditions.get('avg_price_30d'):
                            input_features['price_ma_30'] = float(market_conditions['avg_price_30d'])
                
                # Volatility can remain constant (market volatility doesn't change quickly)
                if market_conditions.get('volatility'):
                    input_features['price_std_30'] = float(market_conditions['volatility'])
            except (ValueError, TypeError) as e:
                # If conversion fails, skip market condition features
                logger.warning(f"Error processing market conditions for {forecast_date}: {e}")
                pass
        
        # Make prediction
        price = predict_price(state, district, crop, input_features, weather_features)
        
        if price:
            price_value = float(price)
            predictions.append({
                'date': forecast_date,
                'price': price_value,
                'weather': weather_features
            })
            # Debug logging (can be removed in production)
            logger.debug(f"Prediction for {forecast_date}: ₹{price_value:,.2f} (days_ahead: {(forecast_date - current_date).days})")
        else:
            logger.warning(f"Failed to generate prediction for {forecast_date}")
    
    # Validate and return results
    result = {
        'predictions': predictions,
        'market_conditions': market_conditions if market_conditions else {},
        'location': {'state': state, 'district': district, 'lat': lat, 'lon': lon},
        'price_unit': 'quintal',  # Standard unit for Indian agricultural commodities
        'price_unit_display': '₹ per quintal',  # Display format
        'quantity': 1,  # 1 quintal = 100 kg
        'quantity_unit': 'quintal'
    }
    
    # Add error information if no predictions were generated
    if not predictions:
        result['error'] = "No predictions generated. This may be due to missing model or invalid inputs."
    
    return result

def predict_price(state, district, crop, input_features, weather_features=None):
    """
    Predict price for given inputs
    
    Args:
        state: State name (will be normalized if possible)
        district: District name (will be normalized if possible)
        crop: Crop name
        input_features: DataFrame with date, state, district, crop columns
        weather_features: Optional dict with weather data
    
    Returns:
        Predicted price or None if error
    """
    # Try to normalize state and district names, but use original if normalization fails
    normalized_state, normalized_district = normalize_location(state, district)
    
    # Use normalized names if available, otherwise use original names
    if normalized_state:
        state_for_prediction = normalized_state
    else:
        state_for_prediction = state
    
    if normalized_district:
        district_for_prediction = normalized_district
    else:
        district_for_prediction = district
    
    # Try validation with both normalized and original names
    is_valid = is_valid_combination(state_for_prediction, district_for_prediction, crop)
    if not is_valid:
        # Try with original names
        is_valid = is_valid_combination(state, district, crop)
        if is_valid:
            # Use original names if validation passes with them
            state_for_prediction = state
            district_for_prediction = district
    
    # Don't fail completely if validation fails - continue with prediction
    # The crop came from dropdown which only shows valid crops
    if not is_valid:
        logger.warning(f"Validation warning for {crop} in {district}, {state}, but continuing...")
    
    # Use the names that work (normalized if available, otherwise original)
    state = state_for_prediction
    district = district_for_prediction
    
    # Predict price using pre-trained models with weather and market conditions
    model_result = load_model(state, district, crop)
    if model_result is None:
        return None
    
    if isinstance(model_result, tuple):
        model, metadata = model_result
    else:
        model = model_result
        metadata = None
    
    if model is None:
        return None

    # Fetch current market conditions to use in prediction
    market_conditions = fetch_current_market_conditions(state, district, crop)
    
    # Prepare features with advanced feature engineering
    try:
        if metadata and 'feature_names' in metadata:
            data_with_features = create_advanced_features_for_prediction(
                input_features, state, district, crop, weather_features, metadata, market_conditions
            )
        else:
            data_with_features = create_features(input_features)
            if weather_features:
                for key, value in weather_features.items():
                    data_with_features[key] = value
    except Exception as e:
        data_with_features = create_features(input_features)
        if weather_features:
            for key, value in weather_features.items():
                data_with_features[key] = value
    
    # Get feature columns based on model expectations
    if metadata and 'feature_names' in metadata:
        expected_features = metadata['feature_names']
    elif hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = [col for col in data_with_features.columns 
                           if col not in ['price', 'date', 'state', 'district', 'crop', 'market', 'variety', 'grade', 'time']]
    
    # Get commodity-specific defaults for missing features
    commodity_defaults = get_commodity_default_prices(crop)
    historical_data = load_historical_commodity_data(crop, state, district)
    if historical_data:
        commodity_defaults = historical_data
    
    # Ensure all expected features exist
    for feat in expected_features:
        if feat not in data_with_features.columns:
            if 'temperature' in feat.lower():
                data_with_features[feat] = 25.0
            elif 'precipitation' in feat.lower() or 'rain' in feat.lower():
                data_with_features[feat] = 0.0
            elif 'price' in feat.lower() and 'change' in feat.lower():
                data_with_features[feat] = 0.0
            elif 'price' in feat.lower() and 'ma' in feat.lower():
                if market_conditions and market_conditions.get('avg_price_7d'):
                    data_with_features[feat] = market_conditions['avg_price_7d']
                else:
                    data_with_features[feat] = commodity_defaults['avg']
            elif 'price' in feat.lower() and ('std' in feat.lower() or 'volatility' in feat.lower()):
                if market_conditions and market_conditions.get('volatility'):
                    data_with_features[feat] = market_conditions['volatility']
                else:
                    data_with_features[feat] = commodity_defaults.get('std', commodity_defaults['avg'] * 0.1)
            elif 'seasonal' in feat.lower():
                if market_conditions and market_conditions.get('avg_price_30d'):
                    data_with_features[feat] = market_conditions['avg_price_30d']
                else:
                    data_with_features[feat] = commodity_defaults['avg']
            elif 'lat' in feat.lower() or 'lon' in feat.lower():
                lat, lon = get_coordinates(state, district)
                data_with_features['lat'] = lat if lat else 20.0
                data_with_features['lon'] = lon if lon else 77.0
            elif 'min_price' in feat.lower() or 'max_price' in feat.lower():
                if market_conditions:
                    if 'min_price' in feat.lower():
                        data_with_features[feat] = market_conditions.get('min_price', commodity_defaults['min'])
                    else:
                        data_with_features[feat] = market_conditions.get('max_price', commodity_defaults['max'])
                else:
                    if 'min_price' in feat.lower():
                        data_with_features[feat] = commodity_defaults['min']
                    else:
                        data_with_features[feat] = commodity_defaults['max']
            else:
                data_with_features[feat] = 0.0
    
    # Select and reorder features to match model (XGBoost is strict about feature order)
    # Ensure all expected features exist and are in the correct order
    missing_features = [f for f in expected_features if f not in data_with_features.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}. Adding defaults.")
        for feat in missing_features:
            data_with_features[feat] = 0.0
    
    # Reorder columns to match expected_features exactly (order matters for XGBoost)
    # Create DataFrame with features in the exact order expected by the model
    X_dict = {}
    for feat in expected_features:
        if feat in data_with_features.columns:
            X_dict[feat] = data_with_features[feat].values[0] if len(data_with_features) > 0 else data_with_features[feat].iloc[0]
        else:
            X_dict[feat] = 0.0
    
    # Create DataFrame with features in exact order
    X = pd.DataFrame([X_dict], columns=expected_features)
    
    # Ensure numeric types (but preserve order)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                X[col] = 0.0
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean()).fillna(0)
    
    try:
        # Disable feature validation for XGBoost if order is correct
        prediction = model.predict(X, validate_features=False)
        raw_prediction = float(prediction[0]) if len(prediction) > 0 else None
        
        if raw_prediction is None:
            return None
        
        # CALIBRATION: Adjust prediction based on current market conditions
        # This addresses the issue where models trained on old data predict prices
        # that are much lower than current market prices
        calibrated_prediction = calibrate_prediction(
            raw_prediction, 
            state, 
            district, 
            crop, 
            market_conditions,
            metadata
        )
        
        return calibrated_prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return None

def calibrate_prediction(raw_prediction, state, district, crop, market_conditions, metadata):
    """
    Calibrate model prediction based on current market conditions.
    
    This function addresses the issue where models trained on historical data
    may predict prices that are significantly different from current market prices.
    
    Strategy:
    1. If current market price is available, use it as an anchor
    2. Calculate adjustment factor based on ratio of current price to prediction
    3. Apply conservative adjustment: scale prediction by ratio, but cap the adjustment
    4. This preserves model's relative patterns while anchoring to current market reality
    """
    if raw_prediction is None or raw_prediction <= 0:
        return raw_prediction
    
    # If we have current market data, use it for calibration
    if market_conditions and market_conditions.get('current_price'):
        current_price = float(market_conditions['current_price'])
        
        if current_price > 0:
            # Calculate the ratio between current price and prediction
            price_ratio = current_price / raw_prediction
            
            # If the ratio is very different (model is way off), apply calibration
            # Only adjust if ratio is > 1.3 or < 0.7 (model is significantly off)
            if price_ratio > 1.3:
                # Model is predicting too low - likely trained on old/incorrect data
                # Apply conservative adjustment: scale by ratio but cap at 2.5x
                # Use 70% of the adjustment to avoid over-correction
                capped_ratio = min(price_ratio, 2.5)  # Cap at 2.5x to avoid extreme adjustments
                adjustment = 1.0 + (capped_ratio - 1.0) * 0.7  # Apply 70% of adjustment
                calibrated = raw_prediction * adjustment
                
                logger.info(f"Calibration applied (upward): Raw=₹{raw_prediction:.2f}, Current=₹{current_price:.2f}, "
                          f"Ratio={price_ratio:.2f}x, Adjustment={adjustment:.2f}x, Calibrated=₹{calibrated:.2f}")
                return calibrated
            elif price_ratio < 0.7:
                # Model is predicting too high - apply downward adjustment
                capped_ratio = max(price_ratio, 0.4)  # Cap at 2.5x downward
                adjustment = 1.0 + (capped_ratio - 1.0) * 0.7  # Apply 70% of adjustment
                calibrated = raw_prediction * adjustment
                
                logger.info(f"Calibration applied (downward): Raw=₹{raw_prediction:.2f}, Current=₹{current_price:.2f}, "
                          f"Ratio={price_ratio:.2f}x, Adjustment={adjustment:.2f}x, Calibrated=₹{calibrated:.2f}")
                return calibrated
    
    # If we have 7-day or 30-day averages, use them as secondary calibration
    if market_conditions:
        avg_price = None
        if market_conditions.get('avg_price_7d'):
            avg_price = float(market_conditions['avg_price_7d'])
        elif market_conditions.get('avg_price_30d'):
            avg_price = float(market_conditions['avg_price_30d'])
        
        if avg_price and avg_price > 0:
            price_ratio = avg_price / raw_prediction
            if price_ratio > 1.3 or price_ratio < 0.7:
                # Apply lighter calibration using averages (50% of adjustment)
                capped_ratio = min(max(price_ratio, 0.5), 2.0)
                adjustment = 1.0 + (capped_ratio - 1.0) * 0.5
                calibrated = raw_prediction * adjustment
                logger.debug(f"Secondary calibration using averages: ₹{raw_prediction:.2f} -> ₹{calibrated:.2f} "
                           f"(ratio={price_ratio:.2f}x, adjustment={adjustment:.2f}x)")
                return calibrated
    
    # No calibration needed or no market data available
    return raw_prediction

def get_commodity_default_prices(crop):
    """
    Get commodity-specific default price ranges based on typical Indian market prices (per quintal)
    These are used when market conditions are not available
    """
    # Typical price ranges for Indian commodities (in ₹ per quintal)
    commodity_prices = {
        'Wheat': {'min': 1800, 'max': 2200, 'avg': 2000},
        'Rice': {'min': 1800, 'max': 2500, 'avg': 2150},
        'Maize': {'min': 1500, 'max': 2000, 'avg': 1750},
        'Cotton': {'min': 5000, 'max': 7000, 'avg': 6000},
        'Sugarcane': {'min': 2500, 'max': 3500, 'avg': 3000},
        'Groundnut': {'min': 5000, 'max': 7000, 'avg': 6000},
        'Tomato': {'min': 800, 'max': 2000, 'avg': 1400},
        'Potato': {'min': 600, 'max': 1200, 'avg': 900},
        'Onion': {'min': 1000, 'max': 2500, 'avg': 1750},
        'Soyabean': {'min': 3500, 'max': 5000, 'avg': 4250},
        'Mustard Seed': {'min': 4000, 'max': 6000, 'avg': 5000},
        'Sunflower': {'min': 4500, 'max': 6500, 'avg': 5500},
        'Barley': {'min': 1500, 'max': 2000, 'avg': 1750},
        'Sorghum': {'min': 1800, 'max': 2500, 'avg': 2150},
        'Gram': {'min': 4000, 'max': 6000, 'avg': 5000},
        'Turmeric': {'min': 6000, 'max': 10000, 'avg': 8000},
        'Chili': {'min': 8000, 'max': 15000, 'avg': 11500},
        'Grapes': {'min': 3000, 'max': 6000, 'avg': 4500},
        'Cashew': {'min': 8000, 'max': 12000, 'avg': 10000},
        'Coconut': {'min': 3000, 'max': 5000, 'avg': 4000},
        'Arecanut': {'min': 15000, 'max': 25000, 'avg': 20000},
        'Coffee': {'min': 8000, 'max': 12000, 'avg': 10000},
        'Pomegranate': {'min': 4000, 'max': 8000, 'avg': 6000},
        'Pigeon Pea': {'min': 5000, 'max': 8000, 'avg': 6500},
        'Ragi': {'min': 2500, 'max': 4000, 'avg': 3250},
        'Jowar': {'min': 2000, 'max': 3000, 'avg': 2500},
    }
    
    # Try exact match first
    if crop in commodity_prices:
        return commodity_prices[crop]
    
    # Try case-insensitive match
    crop_lower = crop.lower()
    for key, value in commodity_prices.items():
        if key.lower() == crop_lower:
            return value
    
    # Try partial match (e.g., "Groundnut" matches "Groundnut")
    for key, value in commodity_prices.items():
        if crop_lower in key.lower() or key.lower() in crop_lower:
            return value
    
    # Default fallback - use average of all commodities
    return {'min': 3000, 'max': 5000, 'avg': 4000}

def load_historical_commodity_data(crop, state=None, district=None):
    """
    Try to load historical data for the commodity to get realistic price averages
    """
    try:
        import glob
        historical_files = []
        
        # Check historical_data_extensive directory
        if Path('historical_data_extensive').exists():
            historical_files.extend(glob.glob('historical_data_extensive/*.csv'))
        
        # Check historical_data directory
        if Path('historical_data').exists():
            historical_files.extend(glob.glob('historical_data/*.csv'))
        
        # Check fetched_data directory
        if Path('fetched_data').exists():
            historical_files.extend(glob.glob('fetched_data/*.csv'))
        
        for file_path in historical_files:
            try:
                df = pd.read_csv(file_path, nrows=10000)  # Read sample for speed
                
                # Standardize column names
                if 'commodity' in df.columns:
                    df['crop'] = df['commodity']
                if 'crop' not in df.columns:
                    continue
                
                # Filter for commodity
                commodity_df = df[df['crop'].str.contains(crop, case=False, na=False)]
                
                if not commodity_df.empty and 'price' in commodity_df.columns:
                    # Calculate statistics
                    prices = pd.to_numeric(commodity_df['price'], errors='coerce').dropna()
                    if len(prices) > 0:
                        return {
                            'min': float(prices.min()),
                            'max': float(prices.max()),
                            'avg': float(prices.mean()),
                            'std': float(prices.std())
                        }
            except:
                continue
    except:
        pass
    
    return None

def create_advanced_features_for_prediction(input_features, state, district, crop, weather_features, metadata, market_conditions=None):
    """
    Create advanced features matching the trained model structure
    Includes current market conditions and commodity-specific defaults
    """
    import pandas as pd
    
    # Start with input features
    df = input_features.copy()
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime([datetime.now()])
    
    # Add temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Seasonal indicators
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # Add location info
    df['state'] = state
    df['district'] = district
    df['crop'] = crop
    
    # Get coordinates
    lat, lon = get_coordinates(state, district)
    df['lat'] = lat if lat else 20.0
    df['lon'] = lon if lon else 77.0
    
    # Add weather features
    if weather_features:
        for key, value in weather_features.items():
            df[key] = value
    else:
        df['temperature_2m_max'] = 25.0
        df['temperature_2m_min'] = 20.0
        df['precipitation_sum'] = 0.0
    
    # Get commodity-specific price defaults
    commodity_defaults = get_commodity_default_prices(crop)
    
    # Try to load historical data for more accurate defaults
    historical_data = load_historical_commodity_data(crop, state, district)
    if historical_data:
        commodity_defaults = historical_data
    
    # Add current market conditions if available, otherwise use commodity-specific defaults
    if market_conditions and market_conditions.get('current_price'):
        current_price = market_conditions.get('current_price', commodity_defaults['avg'])
        min_price = market_conditions.get('min_price', commodity_defaults['min'])
        max_price = market_conditions.get('max_price', commodity_defaults['max'])
        avg_7d = market_conditions.get('avg_price_7d', current_price)
        avg_30d = market_conditions.get('avg_price_30d', current_price)
        volatility = market_conditions.get('volatility', commodity_defaults.get('std', current_price * 0.1))
    else:
        # Use commodity-specific defaults
        current_price = commodity_defaults['avg']
        min_price = commodity_defaults['min']
        max_price = commodity_defaults['max']
        avg_7d = commodity_defaults['avg']
        avg_30d = commodity_defaults['avg']
        volatility = commodity_defaults.get('std', commodity_defaults['avg'] * 0.1)
    
    # Set price features
    df['min_price'] = min_price
    df['max_price'] = max_price
    df['price_ma_7'] = avg_7d
    df['price_ma_30'] = avg_30d
    df['price_std_7'] = volatility
    df['price_std_30'] = volatility
    df['price_volatility'] = volatility  # Add price_volatility feature
    df['price_range'] = max_price - min_price
    df['price_range_ratio'] = (max_price - min_price) / max_price if max_price > 0 else 0.0
    df['seasonal_avg_price'] = avg_30d
    
    # Calculate price changes
    df['price_change'] = ((current_price - avg_7d) / avg_7d * 100) if avg_7d > 0 else 0.0
    df['price_change_7d'] = ((current_price - avg_7d) / avg_7d * 100) if avg_7d > 0 else 0.0
    df['price_change_30d'] = ((current_price - avg_30d) / avg_30d * 100) if avg_30d > 0 else 0.0
    
    return df

