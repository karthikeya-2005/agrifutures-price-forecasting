"""
Enhanced Market Data Fetcher
Implements actual data fetching from Data.gov.in API and other sources
"""

import requests
import pandas as pd
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)

def fetch_data_gov_data(api_key: str = None, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch data from Data.gov.in API
    
    Args:
        api_key: API key for data.gov.in (if None, tries to load from config)
        limit: Maximum number of records to fetch
    
    Returns:
        DataFrame with market data or None on failure
    """
    # Try to load API key from config if not provided
    if api_key is None:
        try:
            import yaml
            import os
            config_path = os.path.join('ingestion_pipeline', 'configs', 'api_keys.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    api_key = config.get('data_gov_key', '')
        except:
            pass
    
    if not api_key:
        logger.warning("No API key provided for Data.gov.in")
        return None
    
    base_url = "https://api.data.gov.in/resource/"
    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    
    url = f"{base_url}{resource_id}"
    
    params = {
        'api-key': api_key,
        'format': 'json',
        'limit': str(limit),
        'offset': '0'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'records' in data and len(data['records']) > 0:
            df = pd.DataFrame(data['records'])
            
            # Standardize column names
            rename_map = {
                'state': 'state',
                'district': 'district',
                'market': 'market',
                'commodity': 'crop',
                'variety': 'variety',
                'modal_price': 'price',
                'min_price': 'min_price',
                'max_price': 'max_price',
                'arrival_date': 'date'
            }
            
            # Rename columns that exist
            for old_col, new_col in rename_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure date column exists and is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Ensure price column exists and is numeric
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            df = df.dropna(subset=['date', 'price'])
            
            logger.info(f"Fetched {len(df)} records from Data.gov.in")
            return df
        else:
            logger.warning("No records returned from Data.gov.in")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch Data.gov.in data: {e}")
        return None

def fetch_enam_data() -> Optional[pd.DataFrame]:
    """
    Fetch data from e-NAM (National Agriculture Market)
    Uses web scraping approach with multiple strategies
    """
    try:
        from bs4 import BeautifulSoup
        import json
        import re
        
        url = "https://enam.gov.in/web/commodity"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=20, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data_records = []
        
        # Strategy 1: Look for JSON data in script tags
        script_tags = soup.find_all('script', type='text/javascript')
        for script in script_tags:
            if script.string:
                # Try to find JSON objects with price/commodity data
                json_matches = re.findall(r'\{[^{}]*"(?:price|commodity|market|date|state|district)"[^{}]*\}', script.string)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        if any(key in data for key in ['price', 'commodity', 'market']):
                            data_records.append(data)
                    except:
                        pass
        
        # Strategy 2: Look for data tables
        tables = soup.find_all('table')
        for table in tables:
            try:
                # Try pandas read_html
                dfs = pd.read_html(str(table))
                if dfs:
                    df_table = dfs[0]
                    # Check if it looks like price data
                    if any(col.lower() in ['price', 'commodity', 'market', 'date'] for col in df_table.columns):
                        # Standardize column names
                        df_table.columns = [col.strip().lower() for col in df_table.columns]
                        if 'date' in df_table.columns or 'price' in df_table.columns:
                            data_records.append(df_table.to_dict('records'))
            except:
                pass
        
        # Strategy 3: Look for divs with data attributes
        data_divs = soup.find_all('div', {'data-commodity': True}) + soup.find_all('div', {'data-price': True})
        for div in data_divs:
            record = {}
            for attr in div.attrs:
                if 'data-' in attr:
                    key = attr.replace('data-', '')
                    record[key] = div.attrs[attr]
            if record:
                data_records.append(record)
        
        # Strategy 4: Look for specific e-NAM API endpoints (if available)
        # Try common API patterns
        api_patterns = [
            "https://enam.gov.in/api/commodity",
            "https://enam.gov.in/web/api/commodity",
            "https://www.enam.gov.in/api/commodity"
        ]
        
        for api_url in api_patterns:
            try:
                api_response = session.get(api_url, headers=headers, timeout=10, verify=False)
                if api_response.status_code == 200:
                    api_data = api_response.json()
                    if isinstance(api_data, list):
                        data_records.extend(api_data)
                    elif isinstance(api_data, dict) and 'data' in api_data:
                        data_records.extend(api_data['data'])
            except:
                pass
        
        if data_records:
            # Flatten nested lists
            flat_records = []
            for record in data_records:
                if isinstance(record, list):
                    flat_records.extend(record)
                else:
                    flat_records.append(record)
            
            df = pd.DataFrame(flat_records)
            
            # Standardize column names
            rename_map = {
                'commodity': 'crop',
                'modal_price': 'price',
                'min_price': 'min_price',
                'max_price': 'max_price',
                'arrival_date': 'date',
                'market_name': 'market'
            }
            
            for old_col, new_col in rename_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns exist
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            df = df.dropna(subset=['price'])
            
            if not df.empty:
                logger.info(f"Fetched {len(df)} records from e-NAM")
                return df
        
        logger.warning("e-NAM: No data found with current scraping strategies")
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch e-NAM data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def fetch_msamb_data() -> Optional[pd.DataFrame]:
    """
    Fetch data from MSAMB (Maharashtra State Agriculture Marketing Board)
    Uses web scraping with multiple strategies
    """
    try:
        from bs4 import BeautifulSoup
        import json
        import re
        
        # MSAMB website URLs to try
        urls = [
            "https://mahamandi.com",
            "https://www.mahamandi.com",
            "https://mahamandi.com/market-prices",
            "https://www.mahamandi.com/market-prices"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        session = requests.Session()
        data_records = []
        
        for url in urls:
            try:
                response = session.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Strategy 1: Look for tables with price data
                    tables = soup.find_all('table')
                    for table in tables:
                        try:
                            dfs = pd.read_html(str(table))
                            if dfs:
                                df_table = dfs[0]
                                # Check if it looks like price data
                                cols_lower = [col.lower() for col in df_table.columns]
                                if any(keyword in ' '.join(cols_lower) for keyword in ['price', 'commodity', 'market', 'date', 'modal']):
                                    # Standardize
                                    df_table.columns = [col.strip().lower() for col in df_table.columns]
                                    data_records.extend(df_table.to_dict('records'))
                        except:
                            pass
                    
                    # Strategy 2: Look for JSON data in script tags
                    script_tags = soup.find_all('script')
                    for script in script_tags:
                        if script.string:
                            # Look for price/commodity data
                            json_matches = re.findall(r'\{[^{}]*"(?:price|commodity|market|modal)"[^{}]*\}', script.string, re.IGNORECASE)
                            for match in json_matches:
                                try:
                                    data = json.loads(match)
                                    if any(key in str(data).lower() for key in ['price', 'commodity', 'market']):
                                        data_records.append(data)
                                except:
                                    pass
                    
                    # Strategy 3: Look for API endpoints
                    api_links = soup.find_all('a', href=re.compile(r'api|json|data', re.I))
                    for link in api_links[:5]:  # Limit to first 5
                        try:
                            api_url = link.get('href')
                            if not api_url.startswith('http'):
                                api_url = url + api_url
                            api_response = session.get(api_url, headers=headers, timeout=10)
                            if api_response.status_code == 200:
                                try:
                                    api_data = api_response.json()
                                    if isinstance(api_data, list):
                                        data_records.extend(api_data)
                                    elif isinstance(api_data, dict):
                                        data_records.append(api_data)
                                except:
                                    pass
                        except:
                            pass
                    
                    break  # If we got a successful response, stop trying other URLs
                    
            except requests.exceptions.RequestException:
                continue
        
        if data_records:
            df = pd.DataFrame(data_records)
            
            # Standardize column names
            rename_map = {
                'commodity': 'crop',
                'modal_price': 'price',
                'min_price': 'min_price',
                'max_price': 'max_price',
                'arrival_date': 'date',
                'market_name': 'market',
                'state': 'state',
                'district': 'district'
            }
            
            for old_col, new_col in rename_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Add state if missing (MSAMB is Maharashtra)
            if 'state' not in df.columns:
                df['state'] = 'Maharashtra'
            
            df = df.dropna(subset=['price'])
            
            if not df.empty:
                logger.info(f"Fetched {len(df)} records from MSAMB")
                return df
        
        # Check if website shows "Coming Soon"
        try:
            response = session.get("https://mahamandi.com", headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if 'coming soon' in soup.get_text().lower():
                    logger.warning("MSAMB: Website shows 'Coming Soon' - not functional")
                    return None
        except:
            pass
        
        logger.warning("MSAMB: No data found with current scraping strategies")
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch MSAMB data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

