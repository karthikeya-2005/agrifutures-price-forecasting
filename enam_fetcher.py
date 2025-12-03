"""
e-NAM (National Agriculture Market) Live Price Fetcher
Fetches current market prices from https://enam.gov.in/web/dashboard/live_price
"""
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnamFetcher:
    """Fetcher for e-NAM live price data"""
    
    def __init__(self):
        self.base_url = "https://enam.gov.in"
        self.live_price_url = "https://enam.gov.in/web/dashboard/live_price"
        self.trade_data_url = "https://enam.gov.in/web/dashboard/trade-data"
        self.agm_enam_url = "https://enam.gov.in/web/dashboard/Agm_Enam_ctrl"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://enam.gov.in/'
        })
    
    def fetch_live_prices(self, state: Optional[str] = None, commodity: Optional[str] = None, 
                          from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch live prices from e-NAM
        Tries multiple endpoints in priority order:
        1. trade-data (https://enam.gov.in/web/dashboard/trade-data) - Most comprehensive
        2. Agm_Enam_ctrl (https://enam.gov.in/web/dashboard/Agm_Enam_ctrl) - e-NAM vs AGMARKNET
        3. live_price (https://enam.gov.in/web/dashboard/live_price) - Live prices
        
        Args:
            state: Filter by state (optional)
            commodity: Filter by commodity (optional)
            from_date: Start date (YYYY-MM-DD format, optional)
            to_date: End date (YYYY-MM-DD format, optional)
        
        Returns:
            DataFrame with columns: date, state, district, crop, price, min_price, max_price
        """
        try:
            logger.info(f"Fetching e-NAM data for state={state}, commodity={commodity}")
            
            # Strategy 1: Try trade-data endpoint (most comprehensive)
            # URL: https://enam.gov.in/web/dashboard/trade-data
            logger.debug("Trying trade-data endpoint...")
            df = self._fetch_trade_data(state, commodity, from_date, to_date)
            if df is not None and not df.empty:
                logger.info(f"✅ Successfully fetched {len(df)} records from e-NAM trade-data")
                return df
            
            # Strategy 2: Try Agm_Enam_ctrl endpoint (e-NAM vs AGMARKNET comparison)
            # URL: https://enam.gov.in/web/dashboard/Agm_Enam_ctrl
            logger.debug("Trying Agm_Enam_ctrl endpoint...")
            df = self._fetch_agm_enam_data(state, commodity, from_date, to_date)
            if df is not None and not df.empty:
                logger.info(f"✅ Successfully fetched {len(df)} records from e-NAM Agm_Enam_ctrl")
                return df
            
            # Strategy 3: Try API endpoint (if available)
            logger.debug("Trying API endpoint...")
            api_url = f"{self.base_url}/api/live-price"
            params = {}
            if state:
                params['state'] = state
            if commodity:
                params['commodity'] = commodity
            if from_date:
                params['from_date'] = from_date
            if to_date:
                params['to_date'] = to_date
            
            try:
                response = self.session.get(api_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list):
                        df = pd.DataFrame(data)
                        if not df.empty:
                            df = self._standardize_dataframe(df)
                            logger.info(f"✅ Successfully fetched {len(df)} records from e-NAM API")
                            return df
            except Exception as e:
                logger.debug(f"e-NAM API approach failed: {e}")
            
            # Strategy 4: Fallback to scraping live_price page
            # URL: https://enam.gov.in/web/dashboard/live_price
            logger.debug("Trying live_price page scraping...")
            df = self._scrape_live_prices(state, commodity, from_date, to_date)
            if df is not None and not df.empty:
                logger.info(f"✅ Successfully scraped {len(df)} records from e-NAM live_price page")
                return df
            
            logger.warning("All e-NAM endpoints failed to return data")
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching e-NAM data: {e}")
            return None
    
    def _scrape_live_prices(self, state: Optional[str] = None, commodity: Optional[str] = None,
                            from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Scrape live prices from e-NAM live_price dashboard
        Endpoint: https://enam.gov.in/web/dashboard/live_price
        Note: This page loads data dynamically, so we scrape HTML tables as fallback
        """
        try:
            # Try to get data from the live price page
            # URL: https://enam.gov.in/web/dashboard/live_price
            response = self.session.get(self.live_price_url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"e-NAM page returned status {response.status_code}")
                return None
            
            # Look for JSON data in the page
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find script tags with JSON data
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and ('price' in script.string.lower() or 'commodity' in script.string.lower()):
                    try:
                        # Try to extract JSON data
                        text = script.string
                        # Look for JSON objects
                        import re
                        json_matches = re.findall(r'\{[^{}]*"price"[^{}]*\}', text)
                        if json_matches:
                            data = []
                            for match in json_matches:
                                try:
                                    obj = json.loads(match)
                                    data.append(obj)
                                except:
                                    continue
                            if data:
                                df = pd.DataFrame(data)
                                df = self._standardize_dataframe(df)
                                logger.info(f"Scraped {len(df)} records from e-NAM page")
                                return df
                    except Exception as e:
                        logger.debug(f"Error parsing script data: {e}")
            
            # Try to find table data
            from io import StringIO
            tables = soup.find_all('table')
            for table in tables:
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    if not df.empty and any(col in df.columns.str.lower() for col in ['price', 'commodity', 'apmc']):
                        df = self._standardize_dataframe(df)
                        logger.info(f"Scraped {len(df)} records from e-NAM table")
                        return df
                except Exception as e:
                    logger.debug(f"Error parsing table: {e}")
                    continue
            
            logger.warning("No data found on e-NAM page")
            return None
            
        except Exception as e:
            logger.warning(f"Error scraping e-NAM data: {e}")
            return None
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize e-NAM dataframe to common format"""
        if df.empty:
            return df
        
        # Standardize column names
        # IMPORTANT: Preserve APMC column separately - don't rename it to district yet
        rename_map = {
            'Commodity': 'crop', 'commodity': 'crop',
            'State': 'state', 'state': 'state',
            'Date': 'date', 'date': 'date',
            'Price in Rs.': 'price', 'Price': 'price', 'price': 'price',
            'Min Price': 'min_price', 'Min': 'min_price',
            'Max Price': 'max_price', 'Max': 'max_price',
            'Modal Price': 'price', 'Modal': 'price',
            'District': 'district', 'district': 'district'  # Keep district if it exists
        }
        
        # Convert column names to lowercase for matching
        df.columns = df.columns.str.strip()
        
        # Check if APMC column exists
        has_apmc = False
        apmc_col_name = None
        for col in df.columns:
            col_lower = col.lower()
            if 'apmc' in col_lower and col_lower not in ['district']:
                has_apmc = True
                apmc_col_name = col
                # Rename to 'apmc' for consistency
                df.rename(columns={col: 'apmc'}, inplace=True)
                break
        
        # Rename other columns
        for old_col in df.columns:
            if old_col == 'apmc':  # Skip APMC column
                continue
            for key, new_col in rename_map.items():
                if old_col.lower() == key.lower():
                    df.rename(columns={old_col: new_col}, inplace=True)
                    break
        
        # Ensure required columns exist
        if 'date' not in df.columns:
            df['date'] = datetime.now().date()
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].fillna(datetime.now().date())
        
        if 'price' not in df.columns:
            # Try to use modal price or calculate from min/max
            if 'min_price' in df.columns and 'max_price' in df.columns:
                df['price'] = (df['min_price'] + df['max_price']) / 2
            else:
                logger.warning("No price column found in e-NAM data")
                return pd.DataFrame()
        
        # Convert price to numeric
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', '').str.replace('Rs', '').str.replace('/', '').str.strip(), errors='coerce')
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]
        
        # Add today's date if missing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].fillna(pd.Timestamp.now())
        
        # Ensure state and district are strings (before APMC mapping)
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip()
            # Replace 'nan' string with actual NaN for mapping
            df.loc[df['state'].str.lower() == 'nan', 'state'] = None
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip()
            # Replace 'nan' string with actual NaN for mapping
            df.loc[df['district'].str.lower() == 'nan', 'district'] = None
        if 'crop' in df.columns:
            df['crop'] = df['crop'].astype(str).str.strip()
        if 'apmc' in df.columns:
            df['apmc'] = df['apmc'].astype(str).str.strip()
            # Remove rows where APMC is 'nan' or empty (invalid data)
            df = df[df['apmc'].notna() & (df['apmc'].str.lower() != 'nan') & (df['apmc'].str.strip() != '')]
        
        # Map APMCs to districts/states if APMC column exists
        # Always map APMCs to ensure correct district/state mapping
        # This ensures data from e-NAM at APMC level is correctly mapped for filtering
        if 'apmc' in df.columns:
            # Check if mapping is needed
            needs_mapping = False
            mapping_reason = []
            
            # Need mapping if:
            # 1. No district column exists
            # 2. District column exists but has any missing values
            # 3. No state column exists  
            # 4. State column exists but has any missing values
            
            if 'district' not in df.columns:
                needs_mapping = True
                mapping_reason.append("No district column")
            elif df['district'].isna().any():
                needs_mapping = True
                missing_count = df['district'].isna().sum()
                mapping_reason.append(f"{missing_count} missing districts")
            
            if 'state' not in df.columns:
                needs_mapping = True
                mapping_reason.append("No state column")
            elif df['state'].isna().any():
                needs_mapping = True
                missing_count = df['state'].isna().sum()
                mapping_reason.append(f"{missing_count} missing states")
            
            # Always map when APMC exists to ensure correct mapping even if district/state exist
            # This validates and enhances existing mappings, ensuring consistency
            if 'apmc' in df.columns and not df['apmc'].isna().all():
                needs_mapping = True
                if not mapping_reason:
                    mapping_reason.append("Validating/enhancing APMC mappings")
            
            if needs_mapping:
                try:
                    from apmc_mapper import get_apmc_mapper
                    mapper = get_apmc_mapper()
                    rows_before = len(df)
                    
                    # Count missing before mapping
                    missing_dist_before = df['district'].isna().sum() if 'district' in df.columns else 0
                    missing_state_before = df['state'].isna().sum() if 'state' in df.columns else 0
                    
                    # Map APMCs - this will fill missing districts/states and validate existing ones
                    df = mapper.map_apmc_dataframe(df)
                    
                    # Count missing after mapping
                    missing_dist_after = df['district'].isna().sum() if 'district' in df.columns else 0
                    missing_state_after = df['state'].isna().sum() if 'state' in df.columns else 0
                    
                    mapped_districts = missing_dist_before - missing_dist_after
                    mapped_states = missing_state_before - missing_state_after
                    
                    logger.info(f"Mapped APMCs to districts/states: {rows_before} records processed")
                    logger.debug(f"  Reasons: {', '.join(mapping_reason)}")
                    if mapped_districts > 0:
                        logger.debug(f"  Mapped {mapped_districts} missing districts")
                    if mapped_states > 0:
                        logger.debug(f"  Mapped {mapped_states} missing states")
                    
                    # Log mapping statistics
                    if 'district' in df.columns and 'state' in df.columns:
                        complete_mappings = df[df['district'].notna() & df['state'].notna()].shape[0]
                        logger.debug(f"  Complete mappings (district+state): {complete_mappings}/{len(df)}")
                except Exception as e:
                    logger.warning(f"Failed to map APMCs: {e}")
                    # Continue without mapping - data will still be usable but may have missing districts/states
        
        # Select output columns (include apmc if it exists)
        output_cols = ['date', 'state', 'district', 'crop', 'price']
        if 'min_price' in df.columns:
            output_cols.append('min_price')
        if 'max_price' in df.columns:
            output_cols.append('max_price')
        if 'apmc' in df.columns:
            output_cols.append('apmc')
        
        # Return only columns that exist
        available_cols = [col for col in output_cols if col in df.columns]
        if all(col in available_cols for col in ['date', 'crop', 'price']):
            return df[available_cols].copy()
        else:
            return df
    
    def _fetch_trade_data(self, state: Optional[str] = None, commodity: Optional[str] = None,
                          from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data from e-NAM trade-data endpoint
        Primary endpoint: https://enam.gov.in/web/dashboard/trade-data
        Uses AJAX API: /web/Ajax_ctrl/trade_data_list
        """
        try:
            # Try AJAX API endpoint first (data is loaded dynamically)
            # This is the backend API that powers the trade-data dashboard
            api_url = f"{self.base_url}/web/Ajax_ctrl/trade_data_list"
            
            # Prepare POST data
            post_data = {}
            if state and state != "-- All --":
                post_data['state'] = state
            if commodity and commodity != "-- Select Commodity --":
                post_data['commodity'] = commodity
            if from_date:
                post_data['from_date'] = from_date
            if to_date:
                post_data['to_date'] = to_date
            
            # Set proper headers for AJAX request
            headers = self.session.headers.copy()
            headers.update({
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': self.trade_data_url
            })
            
            try:
                response = self.session.post(api_url, data=post_data, headers=headers, timeout=15)
                if response.status_code == 200:
                    # Try to parse as JSON first
                    try:
                        data = response.json()
                        if data and isinstance(data, (list, dict)):
                            if isinstance(data, dict) and 'data' in data:
                                data = data['data']
                            if isinstance(data, list) and len(data) > 0:
                                df = pd.DataFrame(data)
                                if not df.empty:
                                    df = self._standardize_dataframe(df)
                                    logger.info(f"Fetched {len(df)} records from e-NAM trade-data API")
                                    return df
                    except (json.JSONDecodeError, ValueError):
                        # If not JSON, try parsing as HTML table
                        from bs4 import BeautifulSoup
                        from io import StringIO
                        soup = BeautifulSoup(response.text, 'html.parser')
                        tables = soup.find_all('table')
                        for table in tables:
                            try:
                                df = pd.read_html(StringIO(str(table)))[0]
                                if not df.empty and any(col in df.columns.str.lower() for col in ['price', 'commodity', 'apmc', 'state']):
                                    df = self._standardize_dataframe(df)
                                    if not df.empty:
                                        logger.info(f"Fetched {len(df)} records from e-NAM trade-data HTML")
                                        return df
                            except Exception as e:
                                logger.debug(f"Error parsing trade-data table: {e}")
                                continue
            except Exception as e:
                logger.debug(f"e-NAM trade-data API failed: {e}")
            
            # Fallback: Try scraping the page (may have empty tables if data loads via JS)
            response = self.session.get(self.trade_data_url, timeout=15)
            if response.status_code != 200:
                return None
            
            from bs4 import BeautifulSoup
            from io import StringIO
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find trade data table
            tables = soup.find_all('table')
            for table in tables:
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    if not df.empty and any(col in df.columns.str.lower() for col in ['price', 'commodity', 'apmc', 'state']):
                        df = self._standardize_dataframe(df)
                        
                        # Apply filters
                        if state:
                            df = df[df['state'].astype(str).str.contains(state, case=False, na=False)]
                        if commodity:
                            df = df[df['crop'].astype(str).str.contains(commodity, case=False, na=False)]
                        
                        if not df.empty:
                            return df
                except Exception as e:
                    logger.debug(f"Error parsing trade-data table: {e}")
                    continue
            
            return None
        except Exception as e:
            logger.debug(f"Error fetching trade-data: {e}")
            return None
    
    def _fetch_agm_enam_data(self, state: Optional[str] = None, commodity: Optional[str] = None,
                             from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data from e-NAM Agm_Enam_ctrl endpoint (e-NAM vs AGMARKNET comparison)
        Primary endpoint: https://enam.gov.in/web/dashboard/Agm_Enam_ctrl
        Uses AJAX API: /web/Ajax_ctrl/agm_enam_data
        """
        try:
            # Try AJAX API endpoint first
            # This is the backend API that powers the Agm_Enam_ctrl dashboard
            api_url = f"{self.base_url}/web/Ajax_ctrl/agm_enam_data"
            
            # Set proper headers for AJAX request
            headers = self.session.headers.copy()
            headers.update({
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': self.agm_enam_url
            })
            
            post_data = {}
            if state and state != "-- All --":
                post_data['state'] = state
            if commodity and commodity != "-- Select Commodity --":
                post_data['commodity'] = commodity
            if from_date:
                post_data['from_date'] = from_date
            if to_date:
                post_data['to_date'] = to_date
            
            try:
                response = self.session.post(api_url, data=post_data, headers=headers, timeout=15)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and isinstance(data, (list, dict)):
                            if isinstance(data, dict) and 'data' in data:
                                data = data['data']
                            if isinstance(data, list) and len(data) > 0:
                                df = pd.DataFrame(data)
                                if not df.empty:
                                    df = self._standardize_dataframe(df)
                                    logger.info(f"Fetched {len(df)} records from e-NAM Agm_Enam API")
                                    return df
                    except (json.JSONDecodeError, ValueError):
                        pass
            except Exception as e:
                logger.debug(f"e-NAM Agm_Enam API failed: {e}")
            
            # Fallback: Try scraping the page
            response = self.session.get(self.agm_enam_url, timeout=15)
            if response.status_code != 200:
                return None
            
            from bs4 import BeautifulSoup
            from io import StringIO
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find comparison data tables
            tables = soup.find_all('table')
            all_data = []
            
            for table in tables:
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    if not df.empty and any(col in df.columns.str.lower() for col in ['price', 'commodity', 'apmc', 'state', 'enam', 'agmarknet']):
                        df = self._standardize_dataframe(df)
                        
                        # Apply filters
                        if state:
                            df = df[df['state'].astype(str).str.contains(state, case=False, na=False)]
                        if commodity:
                            df = df[df['crop'].astype(str).str.contains(commodity, case=False, na=False)]
                        
                        if not df.empty:
                            all_data.append(df)
                except Exception as e:
                    logger.debug(f"Error parsing Agm_Enam table: {e}")
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True).drop_duplicates()
                return combined_df
            
            return None
        except Exception as e:
            logger.debug(f"Error fetching Agm_Enam data: {e}")
            return None


def fetch_enam_data(state: Optional[str] = None, commodity: Optional[str] = None, 
                    limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch data from e-NAM
    
    Args:
        state: Filter by state (optional)
        commodity: Filter by commodity (optional)
        limit: Maximum number of records to return
    
    Returns:
        DataFrame with market price data
    """
    try:
        fetcher = EnamFetcher()
        df = fetcher.fetch_live_prices(state=state, commodity=commodity)
        
        if df is not None and not df.empty:
            if limit and len(df) > limit:
                df = df.head(limit)
            return df
        
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch e-NAM data: {e}")
        return None

