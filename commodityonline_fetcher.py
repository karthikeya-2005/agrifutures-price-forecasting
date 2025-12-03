"""
Commodity Online Mandi Price Fetcher
Fetches current market prices from https://www.commodityonline.com/mandiprices
Uses multiple strategies: API endpoints, AJAX calls, HTML scraping with retry logic
"""
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
import logging
from bs4 import BeautifulSoup
import re
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommodityOnlineFetcher:
    """Fetcher for Commodity Online mandi prices with multiple strategies"""
    
    def __init__(self):
        self.base_url = "https://www.commodityonline.com"
        self.mandi_prices_url = "https://www.commodityonline.com/mandiprices"
        self.session = requests.Session()
        self._setup_session()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _setup_session(self):
        """Setup session with proper headers to avoid bot detection"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        })
    
    def fetch_mandi_prices(self, commodity: Optional[str] = None, 
                          state: Optional[str] = None, 
                          district: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch mandi prices from Commodity Online using multiple strategies
        
        Strategy 1: Try API/AJAX endpoints
        Strategy 2: Try direct URL access with retry logic
        Strategy 3: Fallback to HTML scraping
        
        Args:
            commodity: Filter by commodity name (optional)
            state: Filter by state (optional)
            district: Filter by district (optional)
        
        Returns:
            DataFrame with columns: date, state, district, crop, price, min_price, max_price
        """
        # Strategy 1: Try API/AJAX endpoints
        df = self._try_api_endpoints(commodity, state, district)
        if df is not None and not df.empty:
            return df
        
        # Strategy 2: Try direct URL access with retry logic
        df = self._try_direct_access_with_retry(commodity, state, district)
        if df is not None and not df.empty:
            return df
        
        # Strategy 3: Fallback to HTML scraping
        return self._scrape_html(commodity, state, district)
    
    def _try_api_endpoints(self, commodity: Optional[str] = None,
                          state: Optional[str] = None,
                          district: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Try to fetch data from API/AJAX endpoints"""
        try:
            # Try common API endpoint patterns
            api_endpoints = [
                f"{self.base_url}/api/mandi-prices",
                f"{self.base_url}/api/prices",
                f"{self.base_url}/ajax/mandi-prices",
            ]
            
            params = {}
            if commodity:
                params['commodity'] = commodity
            if state:
                params['state'] = state
            if district:
                params['district'] = district
            
            headers = self.session.headers.copy()
            headers.update({
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': self.mandi_prices_url
            })
            
            for endpoint in api_endpoints:
                try:
                    response = self.session.get(endpoint, params=params, headers=headers, timeout=10)
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
                                        logger.info(f"Fetched {len(df)} records from Commodity Online API")
                                        return df
                        except (json.JSONDecodeError, ValueError):
                            continue
                except Exception as e:
                    logger.debug(f"API endpoint {endpoint} failed: {e}")
                    continue
            
            return None
        except Exception as e:
            logger.debug(f"API endpoints strategy failed: {e}")
            return None
    
    def _try_direct_access_with_retry(self, commodity: Optional[str] = None,
                                      state: Optional[str] = None,
                                      district: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Try direct URL access with retry logic and better headers"""
        # Build URL with filters
        url = self.mandi_prices_url
        params = {}
        
        if commodity:
            commodity_slug = commodity.lower().replace(' ', '-').replace('/', '-')
            url = f"{self.base_url}/mandiprices/{commodity_slug}"
        
        if state:
            state_slug = state.lower().replace(' ', '-')
            url = f"{url}/{state_slug}"
        
        if district:
            district_slug = district.lower().replace(' ', '-')
            url = f"{url}/{district_slug}"
        
        # Try with retry logic
        for attempt in range(self.max_retries):
            try:
                # Update referer for each attempt
                headers = self.session.headers.copy()
                headers['Referer'] = self.base_url if attempt == 0 else url
                
                response = self.session.get(url, params=params, headers=headers, timeout=15)
                
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
                                    logger.info(f"Fetched {len(df)} records from Commodity Online (direct access)")
                                    return df
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, will try HTML parsing
                        pass
                    
                    # Try HTML parsing
                    df = self._parse_html_response(response.text, commodity, state, district)
                    if df is not None and not df.empty:
                        return df
                
                elif response.status_code == 403:
                    # 403 Forbidden - wait longer and try again
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        logger.debug(f"Got 403, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                        time.sleep(wait_time)
                        # Try rotating User-Agent
                        self._rotate_user_agent()
                        continue
                    else:
                        logger.warning(f"Commodity Online returned 403 after {self.max_retries} attempts")
                        return None
                else:
                    logger.warning(f"Commodity Online returned status {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return None
        
        return None
    
    def _rotate_user_agent(self):
        """Rotate User-Agent to avoid detection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        import random
        self.session.headers['User-Agent'] = random.choice(user_agents)
    
    def _scrape_html(self, commodity: Optional[str] = None,
                     state: Optional[str] = None,
                     district: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fallback HTML scraping method"""
        try:
            url = self.mandi_prices_url
            if commodity:
                commodity_slug = commodity.lower().replace(' ', '-').replace('/', '-')
                url = f"{self.base_url}/mandiprices/{commodity_slug}"
            if state:
                state_slug = state.lower().replace(' ', '-')
                url = f"{url}/{state_slug}"
            if district:
                district_slug = district.lower().replace(' ', '-')
                url = f"{url}/{district_slug}"
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Commodity Online returned status {response.status_code}")
                return None
            
            return self._parse_html_response(response.text, commodity, state, district)
            
        except Exception as e:
            logger.warning(f"Error fetching Commodity Online data: {e}")
            return None
    
    def _parse_html_response(self, html: str, commodity: Optional[str] = None,
                            state: Optional[str] = None,
                            district: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Parse HTML response to extract price data"""
        try:
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find price table
            tables = soup.find_all('table')
            data_rows = []
            
            for table in tables:
                try:
                    rows = table.find_all('tr')
                    headers = []
                    
                    # Get headers from first row
                    if rows:
                        header_row = rows[0]
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                    
                    # Skip if no price-related headers
                    if not any(keyword in ' '.join(headers).lower() for keyword in ['price', 'commodity', 'mandi', 'district']):
                        continue
                    
                    # Parse data rows
                    for row in rows[1:]:
                        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                        if len(cells) >= 3:  # At least commodity, location, price
                            row_data = {}
                            
                            # Try to extract data based on common patterns
                            for i, cell in enumerate(cells):
                                if i < len(headers):
                                    header = headers[i].lower()
                                    if 'commodity' in header or 'crop' in header:
                                        row_data['crop'] = cell
                                    elif 'state' in header:
                                        row_data['state'] = cell
                                    elif 'district' in header or 'mandi' in header or 'apmc' in header:
                                        row_data['district'] = cell
                                    elif 'price' in header and 'min' in header:
                                        row_data['min_price'] = self._extract_price(cell)
                                    elif 'price' in header and 'max' in header:
                                        row_data['max_price'] = self._extract_price(cell)
                                    elif 'price' in header or 'rs' in header:
                                        row_data['price'] = self._extract_price(cell)
                                    elif 'date' in header:
                                        row_data['date'] = self._parse_date(cell)
                            
                            # If we have essential data, add the row
                            if 'crop' in row_data and ('price' in row_data or 'min_price' in row_data):
                                # Fill missing fields
                                if 'date' not in row_data:
                                    row_data['date'] = datetime.now().date()
                                if 'state' not in row_data:
                                    row_data['state'] = state if state else ''
                                if 'district' not in row_data:
                                    row_data['district'] = district if district else ''
                                
                                # Calculate price if only min/max available
                                if 'price' not in row_data:
                                    if 'min_price' in row_data and 'max_price' in row_data:
                                        row_data['price'] = (row_data['min_price'] + row_data['max_price']) / 2
                                    elif 'min_price' in row_data:
                                        row_data['price'] = row_data['min_price']
                                    elif 'max_price' in row_data:
                                        row_data['price'] = row_data['max_price']
                                
                                data_rows.append(row_data)
                
                except Exception as e:
                    logger.debug(f"Error parsing table: {e}")
                    continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                df = self._standardize_dataframe(df)
                logger.info(f"Fetched {len(df)} records from Commodity Online")
                return df
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                df = self._standardize_dataframe(df)
                logger.info(f"Scraped {len(df)} records from Commodity Online HTML")
                return df
            
            logger.warning("No price data found on Commodity Online page")
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing HTML: {e}")
            return None
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price value from text"""
        if not text:
            return None
        
        # Remove currency symbols and extract numbers
        price_text = re.sub(r'[^\d.,]', '', str(text))
        price_text = price_text.replace(',', '')
        
        try:
            return float(price_text)
        except:
            return None
    
    def _parse_date(self, text: str) -> datetime:
        """Parse date from text"""
        if not text:
            return datetime.now().date()
        
        try:
            # Try common date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d %b %Y', '%d %B %Y']:
                try:
                    return datetime.strptime(text.strip(), fmt).date()
                except:
                    continue
        except:
            pass
        
        return datetime.now().date()
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Commodity Online dataframe to common format"""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_cols = ['date', 'state', 'district', 'crop', 'price']
        for col in required_cols:
            if col not in df.columns:
                if col == 'date':
                    df['date'] = datetime.now().date()
                else:
                    df[col] = ''
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date'] = df['date'].fillna(datetime.now())
        
        # Convert price to numeric
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]
        
        # Ensure string columns
        for col in ['state', 'district', 'crop']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Add min_price and max_price if missing
        if 'min_price' not in df.columns:
            df['min_price'] = None
        if 'max_price' not in df.columns:
            df['max_price'] = None
        
        return df[['date', 'state', 'district', 'crop', 'price', 'min_price', 'max_price']].copy()


def fetch_commodityonline_data(commodity: Optional[str] = None, 
                               state: Optional[str] = None,
                               district: Optional[str] = None,
                               limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch data from Commodity Online
    
    Args:
        commodity: Filter by commodity name (optional)
        state: Filter by state (optional)
        district: Filter by district (optional)
        limit: Maximum number of records to return
    
    Returns:
        DataFrame with market price data
    """
    try:
        fetcher = CommodityOnlineFetcher()
        df = fetcher.fetch_mandi_prices(commodity=commodity, state=state, district=district)
        
        if df is not None and not df.empty:
            if limit and len(df) > limit:
                df = df.head(limit)
            return df
        
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch Commodity Online data: {e}")
        return None

