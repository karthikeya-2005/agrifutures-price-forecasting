"""
Enrich consolidated data with historical weather data
Fetches weather data for all unique state-district combinations and merges with price data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from geolocation_fetcher import get_coordinates
from weather_data_fetcher import fetch_weather_data
from cache_manager import get_cache_manager
import logging
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enrich_data_with_weather(input_file, output_file, batch_size=100, max_retries=3):
    """
    Enrich consolidated data with historical weather data
    
    Args:
        input_file: Path to consolidated CSV file
        output_file: Path to save enriched CSV file
        batch_size: Number of records to process in each batch
        max_retries: Maximum retry attempts for API calls
    """
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['date'], low_memory=False)
    logger.info(f"Loaded {len(df):,} records")
    
    # Check if weather data already exists
    if 'temperature_2m_max' in df.columns and df['temperature_2m_max'].notna().sum() > 0:
        logger.info("Weather data already exists in dataset")
        # Check if we need to fill missing values
        missing_weather = df[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']].isna().any(axis=1).sum()
        if missing_weather == 0:
            logger.info("All weather data is complete. Skipping enrichment.")
            return df
        else:
            logger.info(f"Found {missing_weather:,} records with missing weather data. Enriching...")
    
    # Get unique state-district combinations
    location_groups = df.groupby(['state', 'district']).size().reset_index(name='count')
    logger.info(f"Found {len(location_groups)} unique state-district combinations")
    
    # Get date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Create coordinate cache
    cache_manager = get_cache_manager()
    coordinates_cache = {}
    
    # Process each location
    weather_data_list = []
    processed_locations = set()
    
    for idx, row in tqdm(location_groups.iterrows(), total=len(location_groups), desc="Processing locations"):
        state = row['state']
        district = row['district']
        
        # Skip Unknown locations
        if state == 'Unknown' or district == 'Unknown':
            continue
        
        location_key = f"{state}_{district}"
        if location_key in processed_locations:
            continue
        
        # Get coordinates (with caching)
        if location_key not in coordinates_cache:
            try:
                lat, lon = get_coordinates(state, district)
                if lat and lon:
                    coordinates_cache[location_key] = (lat, lon)
                else:
                    logger.warning(f"Could not get coordinates for {district}, {state}")
                    continue
            except Exception as e:
                logger.warning(f"Error getting coordinates for {district}, {state}: {e}")
                continue
        else:
            lat, lon = coordinates_cache[location_key]
        
        # Get date range for this location
        location_df = df[(df['state'] == state) & (df['district'] == district)]
        if location_df.empty:
            continue
        
        loc_min_date = location_df['date'].min()
        loc_max_date = location_df['date'].max()
        
        # Fetch weather data in chunks (API limit is typically 90 days)
        current_date = loc_min_date
        all_weather = []
        
        while current_date <= loc_max_date:
            end_date = min(current_date + timedelta(days=89), loc_max_date)
            
            for attempt in range(max_retries):
                try:
                    weather = fetch_weather_data(
                        lat, lon,
                        current_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if weather and 'daily' in weather:
                        daily = weather['daily']
                        dates = pd.date_range(
                            start=current_date,
                            end=end_date,
                            freq='D'
                        )
                        
                        # Extract weather data
                        temp_max = daily.get('temperature_2m_max', [])
                        temp_min = daily.get('temperature_2m_min', [])
                        precip = daily.get('precipitation_sum', [])
                        
                        # Create DataFrame for this period
                        for i, date in enumerate(dates):
                            if i < len(temp_max):
                                all_weather.append({
                                    'date': date,
                                    'state': state,
                                    'district': district,
                                    'temperature_2m_max': temp_max[i] if i < len(temp_max) else None,
                                    'temperature_2m_min': temp_min[i] if i < len(temp_min) else None,
                                    'precipitation_sum': precip[i] if i < len(precip) else None
                                })
                    
                    # Success, break retry loop
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"Error fetching weather for {district}, {state} ({current_date}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch weather for {district}, {state} ({current_date}) after {max_retries} attempts")
            
            # Rate limiting - be respectful to API
            time.sleep(1)
            current_date = end_date + timedelta(days=1)
        
        if all_weather:
            weather_df = pd.DataFrame(all_weather)
            weather_data_list.append(weather_df)
            processed_locations.add(location_key)
            logger.info(f"Fetched weather data for {district}, {state}: {len(weather_df)} days")
    
    if not weather_data_list:
        logger.warning("No weather data was fetched. Returning original data.")
        return df
    
    # Combine all weather data
    logger.info("Combining weather data...")
    all_weather_df = pd.concat(weather_data_list, ignore_index=True)
    logger.info(f"Total weather records: {len(all_weather_df):,}")
    
    # Merge with original data
    logger.info("Merging weather data with price data...")
    
    # Convert date columns to ensure matching
    df['date'] = pd.to_datetime(df['date']).dt.date
    all_weather_df['date'] = pd.to_datetime(all_weather_df['date']).dt.date
    
    # Merge on date, state, and district
    df_enriched = df.merge(
        all_weather_df,
        on=['date', 'state', 'district'],
        how='left'
    )
    
    # Convert date back to datetime
    df_enriched['date'] = pd.to_datetime(df_enriched['date'])
    
    # Calculate average temperature if we have min and max
    if 'temperature_2m_max' in df_enriched.columns and 'temperature_2m_min' in df_enriched.columns:
        df_enriched['temperature'] = (
            df_enriched['temperature_2m_max'] + df_enriched['temperature_2m_min']
        ) / 2
    
    # Map precipitation_sum to rainfall (for compatibility with existing code)
    if 'precipitation_sum' in df_enriched.columns:
        df_enriched['rainfall'] = df_enriched['precipitation_sum']
    
    # Fill missing weather with forward/backward fill within same location
    logger.info("Filling missing weather data...")
    for state_district in df_enriched.groupby(['state', 'district']):
        state, district = state_district[0]
        mask = (df_enriched['state'] == state) & (df_enriched['district'] == district)
        df_enriched.loc[mask, 'temperature_2m_max'] = df_enriched.loc[mask, 'temperature_2m_max'].ffill().bfill()
        df_enriched.loc[mask, 'temperature_2m_min'] = df_enriched.loc[mask, 'temperature_2m_min'].ffill().bfill()
        df_enriched.loc[mask, 'precipitation_sum'] = df_enriched.loc[mask, 'precipitation_sum'].fillna(0)
        if 'temperature' in df_enriched.columns:
            df_enriched.loc[mask, 'temperature'] = df_enriched.loc[mask, 'temperature'].ffill().bfill()
        if 'rainfall' in df_enriched.columns:
            df_enriched.loc[mask, 'rainfall'] = df_enriched.loc[mask, 'rainfall'].fillna(0)
    
    # Statistics
    weather_coverage = df_enriched[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']].notna().all(axis=1).sum()
    logger.info(f"Weather data coverage: {weather_coverage:,} / {len(df_enriched):,} records ({weather_coverage/len(df_enriched)*100:.1f}%)")
    
    # Save enriched data
    logger.info(f"Saving enriched data to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df_enriched):,} records with weather data")
    
    return df_enriched

if __name__ == "__main__":
    input_file = Path('data/combined/all_sources_consolidated.csv')
    output_file = Path('data/combined/all_sources_consolidated_with_weather.csv')
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        exit(1)
    
    logger.info("="*80)
    logger.info("ENRICHING DATA WITH HISTORICAL WEATHER")
    logger.info("="*80)
    
    enriched_df = enrich_data_with_weather(input_file, output_file)
    
    logger.info("="*80)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("="*80)

