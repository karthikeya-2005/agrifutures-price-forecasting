"""
Agmarknet 2.0 API Fetcher
Uses the new API endpoints discovered from the website
"""

import requests
import pandas as pd
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AgmarknetAPIClient:
    """Client for Agmarknet 2.0 API"""
    
    def __init__(self):
        self.base_url = "https://api.agmarknet.gov.in/v1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.agmarknet.gov.in/'
        }
    
    def get_filters(self) -> Optional[Dict]:
        """Get filter data (commodities, states, markets, etc.)"""
        try:
            response = requests.get(
                f"{self.base_url}/dashboard-filters/?dashboard_name=marketwise_price_arrival",
                headers=self.headers,
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and 'data' in data:
                    return data['data']
            return None
        except Exception as e:
            logger.error(f"Failed to get filters: {e}")
            return None
    
    def get_price_data(self, 
                      commodity_id: Optional[int] = None,
                      state_id: Optional[int] = None,
                      district_id: Optional[int] = None,
                      market_id: Optional[int] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 1000,
                      dashboard_name: str = "marketwise_price_arrival") -> Optional[pd.DataFrame]:
        """
        Get price data from Agmarknet API
        
        Args:
            commodity_id: Commodity ID (from filters)
            state_id: State ID (from filters)
            district_id: District ID (from filters)
            market_id: Market ID (from filters)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum records to fetch
        
        Returns:
            DataFrame with price data or None
        """
        # Use GET request with query parameters (discovered from browser network requests)
        endpoint = f"{self.base_url}/dashboard-data"
        
        # Get filter data first to get valid IDs
        filters_data = self.get_filters()
        
        # Build query parameters based on discovered structure
        params = {
            'dashboard': dashboard_name,
            'format': 'json',
            'limit': str(limit),
        }
        
        # Add date (use today if not provided, or yesterday for better results)
        if end_date:
            params['date'] = end_date
        elif start_date:
            params['date'] = start_date
        else:
            from datetime import datetime, timedelta
            # Use yesterday for better chance of data
            params['date'] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Add filters - use array format as discovered: [id] or single id
        if commodity_id:
            params['commodity'] = f'[{commodity_id}]'
        else:
            params['commodity'] = '[100001]'  # All commodities
        
        if state_id:
            params['state'] = str(state_id)
        else:
            # Get first available state from filters
            if filters_data and 'state_data' in filters_data and len(filters_data['state_data']) > 0:
                params['state'] = str(filters_data['state_data'][0].get('state_id', '100006'))
            else:
                params['state'] = '100006'  # Default state
        
        if district_id:
            params['district'] = f'[{district_id}]'
        else:
            params['district'] = '[100007]'  # All districts
        
        if market_id:
            params['market'] = f'[{market_id}]'
        else:
            params['market'] = '[100009]'  # All markets
        
        # Default values based on discovered API call
        params['group'] = '[100000]'  # All groups
        params['variety'] = '100021'  # Default variety (or get from filters)
        params['grades'] = '[4]'  # Default grade
        
        # Try GET request with query parameters
        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle response
                if isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        records = data['data']
                    elif 'status' in data and data.get('status') and 'data' in data:
                        records = data['data'] if isinstance(data['data'], list) else []
                    else:
                        records = []
                elif isinstance(data, list):
                    records = data
                else:
                    records = []
                
                if records:
                    df = pd.DataFrame(records)
                    
                    # Standardize column names
                    rename_map = {
                        'date': 'date',
                        'arrival_date': 'date',
                        'commodity': 'crop',
                        'commodity_name': 'crop',
                        'cmdt_name': 'crop',
                        'state': 'state',
                        'state_name': 'state',
                        'district': 'district',
                        'district_name': 'district',
                        'market': 'market',
                        'market_name': 'market',
                        'modal_price': 'price',
                        'price': 'price',
                        'min_price': 'min_price',
                        'max_price': 'max_price',
                    }
                    
                    for old_col, new_col in rename_map.items():
                        if old_col in df.columns:
                            df.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Ensure date column
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    # Ensure price column
                    if 'price' in df.columns:
                        df['price'] = pd.to_numeric(df['price'], errors='coerce')
                        df = df.dropna(subset=['price'])
                    
                    if not df.empty:
                        logger.info(f"Fetched {len(df)} records from Agmarknet API")
                        return df
        except Exception as e:
            logger.debug(f"GET to dashboard-data failed: {e}")
        
        # Fallback: Try GET endpoints
        endpoints = [
            f"{self.base_url}/marketwise-price-arrival",
            f"{self.base_url}/price-arrival",
        ]
        
        params = {}
        if commodity_id:
            params['cmdt_id'] = commodity_id
        if state_id:
            params['state_id'] = state_id
        if district_id:
            params['district_id'] = district_id
        if market_id:
            params['market_id'] = market_id
        if start_date:
            params['date_from'] = start_date
        if end_date:
            params['date_to'] = end_date
        params['limit'] = limit
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=self.headers, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle different response structures
                    if isinstance(data, dict):
                        if 'data' in data and isinstance(data['data'], list):
                            records = data['data']
                        elif 'status' in data and data.get('status') and 'data' in data:
                            records = data['data'] if isinstance(data['data'], list) else []
                        else:
                            continue
                    elif isinstance(data, list):
                        records = data
                    else:
                        continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        
                        # Standardize column names
                        rename_map = {
                            'date': 'date',
                            'arrival_date': 'date',
                            'commodity': 'crop',
                            'commodity_name': 'crop',
                            'cmdt_name': 'crop',
                            'state': 'state',
                            'state_name': 'state',
                            'district': 'district',
                            'district_name': 'district',
                            'market': 'market',
                            'market_name': 'market',
                            'modal_price': 'price',
                            'price': 'price',
                            'min_price': 'min_price',
                            'max_price': 'max_price',
                        }
                        
                        for old_col, new_col in rename_map.items():
                            if old_col in df.columns:
                                df.rename(columns={old_col: new_col}, inplace=True)
                        
                        # Ensure date column
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        
                        # Ensure price column
                        if 'price' in df.columns:
                            df['price'] = pd.to_numeric(df['price'], errors='coerce')
                        
                        df = df.dropna(subset=['price'])
                        
                        if not df.empty:
                            logger.info(f"Fetched {len(df)} records from Agmarknet API")
                            return df
                            
            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue
        
        logger.warning("No price data found from Agmarknet API")
        return None

def fetch_agmarknet_api_data(limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch data from Agmarknet 2.0 API
    
    Args:
        limit: Maximum records to fetch
    
    Returns:
        DataFrame with price data or None
    """
    client = AgmarknetAPIClient()
    
    # Try to get recent data (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    return client.get_price_data(start_date=start_date, end_date=end_date, limit=limit)

