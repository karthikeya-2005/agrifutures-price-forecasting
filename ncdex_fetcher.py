"""
NCDEX (National Commodity & Derivatives Exchange) Spot Price Fetcher
Fetches historical spot prices from https://www.ncdex.com/markets/spotprices
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from bs4 import BeautifulSoup
import re
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCDEXFetcher:
    """Fetcher for NCDEX spot price data"""
    
    def __init__(self):
        self.base_url = "https://www.ncdex.com"
        self.spot_prices_url = "https://www.ncdex.com/markets/spotprices"
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 2
        self._setup_session()
    
    def _setup_session(self):
        """Setup session with proper headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.ncdex.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'max-age=0'
        })
    
    def fetch_spot_prices(self, commodity: Optional[str] = None, 
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch spot prices from NCDEX using multiple strategies
        
        Strategy 1: Try API/AJAX endpoints
        Strategy 2: Try AJAX data endpoints discovered from page
        Strategy 3: Fallback to HTML scraping with retry
        
        Args:
            commodity: Filter by commodity name (optional)
            from_date: Start date (YYYY-MM-DD format, optional)
            to_date: End date (YYYY-MM-DD format, optional)
        
        Returns:
            DataFrame with columns: date, commodity, price, min_price, max_price
        """
        # Strategy 1: Try known API endpoints
        df = self._try_api_endpoints(commodity, from_date, to_date)
        if df is not None and not df.empty:
            return df
        
        # Strategy 2: Try to discover and use AJAX endpoints
        df = self._try_ajax_endpoints(commodity, from_date, to_date)
        if df is not None and not df.empty:
            return df
        
        # Strategy 3: Fallback to HTML scraping with retry
        return self._scrape_spot_prices_with_retry(commodity, from_date, to_date)
    
    def _try_api_endpoints(self, commodity: Optional[str] = None,
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Try known API endpoint patterns"""
        api_endpoints = [
            f"{self.base_url}/api/spot-prices",
            f"{self.base_url}/api/v1/spot-prices",
            f"{self.base_url}/markets/api/spot-prices",
        ]
        
        params = {}
        if commodity:
            params['commodity'] = commodity
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        
        for api_url in api_endpoints:
            try:
                response = self.session.get(api_url, params=params, timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and isinstance(data, (list, dict)):
                            if isinstance(data, dict):
                                if 'data' in data:
                                    data = data['data']
                                elif 'records' in data:
                                    data = data['records']
                                elif 'results' in data:
                                    data = data['results']
                            if isinstance(data, list) and len(data) > 0:
                                df = pd.DataFrame(data)
                                if not df.empty:
                                    df = self._standardize_dataframe(df)
                                    logger.info(f"Fetched {len(df)} records from NCDEX API: {api_url}")
                                    return df
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"Failed to parse JSON from {api_url}: {e}")
                        continue
            except Exception as e:
                logger.debug(f"API endpoint {api_url} failed: {e}")
                continue
        
        return None
    
    def _try_ajax_endpoints(self, commodity: Optional[str] = None,
                           from_date: Optional[str] = None,
                           to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Try to discover and use AJAX endpoints from the page"""
        try:
            # First, load the page to discover AJAX endpoints
            response = self.session.get(self.spot_prices_url, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for script tags that might contain API endpoints
            scripts = soup.find_all('script')
            api_patterns = []
            
            for script in scripts:
                if script.string:
                    # Look for common API endpoint patterns
                    import re
                    patterns = [
                        r'["\']([^"\']*api[^"\']*spot[^"\']*)["\']',
                        r'["\']([^"\']*ajax[^"\']*price[^"\']*)["\']',
                        r'url:\s*["\']([^"\']*)["\']',
                        r'endpoint:\s*["\']([^"\']*)["\']',
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, script.string, re.I)
                        api_patterns.extend(matches)
            
            # Try discovered endpoints
            for pattern in set(api_patterns[:5]):  # Limit to first 5 unique patterns
                if not pattern.startswith('http'):
                    pattern = f"{self.base_url}{pattern}" if pattern.startswith('/') else f"{self.base_url}/{pattern}"
                
                try:
                    headers = self.session.headers.copy()
                    headers.update({
                        'X-Requested-With': 'XMLHttpRequest',
                        'Content-Type': 'application/json',
                        'Referer': self.spot_prices_url
                    })
                    
                    params = {}
                    if commodity:
                        params['commodity'] = commodity
                    if from_date:
                        params['from_date'] = from_date
                    if to_date:
                        params['to_date'] = to_date
                    
                    response = self.session.get(pattern, params=params, headers=headers, timeout=10)
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
                                        logger.info(f"Fetched {len(df)} records from NCDEX AJAX endpoint")
                                        return df
                        except (json.JSONDecodeError, ValueError):
                            continue
                except Exception as e:
                    logger.debug(f"AJAX endpoint {pattern} failed: {e}")
                    continue
            
            return None
        except Exception as e:
            logger.debug(f"AJAX endpoint discovery failed: {e}")
            return None
    
    def _scrape_spot_prices_with_retry(self, commodity: Optional[str] = None,
                                       from_date: Optional[str] = None,
                                       to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Scrape spot prices with retry logic"""
        for attempt in range(self.max_retries):
            try:
                df = self._scrape_spot_prices(commodity, from_date, to_date)
                if df is not None and not df.empty:
                    return df
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logger.debug(f"Scraping attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
        
        return None
    
    def _scrape_spot_prices(self, commodity: Optional[str] = None,
                            from_date: Optional[str] = None,
                            to_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Scrape spot prices from NCDEX page"""
        try:
            response = self.session.get(self.spot_prices_url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"NCDEX page returned status {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data tables or JSON data
            tables = soup.find_all('table')
            data_rows = []
            
            for table in tables:
                try:
                    rows = table.find_all('tr')
                    if not rows:
                        continue
                    
                    headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
                    
                    # Check if this is a price table
                    if not any(keyword in ' '.join(headers).lower() for keyword in ['price', 'commodity', 'spot', 'date']):
                        continue
                    
                    # Parse data rows
                    for row in rows[1:]:
                        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                        if len(cells) < 2:
                            continue
                        
                        row_data = {}
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                header = headers[i].lower()
                                if 'commodity' in header or 'product' in header:
                                    row_data['crop'] = cell
                                elif 'date' in header:
                                    row_data['date'] = self._parse_date(cell)
                                elif 'price' in header and ('min' in header or 'low' in header):
                                    row_data['min_price'] = self._extract_price(cell)
                                elif 'price' in header and ('max' in header or 'high' in header):
                                    row_data['max_price'] = self._extract_price(cell)
                                elif 'price' in header or 'spot' in header:
                                    row_data['price'] = self._extract_price(cell)
                        
                        if 'crop' in row_data and ('price' in row_data or 'min_price' in row_data):
                            if 'date' not in row_data:
                                row_data['date'] = datetime.now().date()
                            if 'price' not in row_data:
                                if 'min_price' in row_data and 'max_price' in row_data:
                                    row_data['price'] = (row_data['min_price'] + row_data['max_price']) / 2
                                elif 'min_price' in row_data:
                                    row_data['price'] = row_data['min_price']
                                elif 'max_price' in row_data:
                                    row_data['price'] = row_data['max_price']
                            
                            # Filter by commodity if specified
                            if commodity and commodity.lower() not in row_data.get('crop', '').lower():
                                continue
                            
                            data_rows.append(row_data)
                
                except Exception as e:
                    logger.debug(f"Error parsing table: {e}")
                    continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                df = self._standardize_dataframe(df)
                logger.info(f"Scraped {len(df)} records from NCDEX page")
                return df
            
            logger.warning("No spot price data found on NCDEX page")
            return None
            
        except Exception as e:
            logger.warning(f"Error scraping NCDEX data: {e}")
            return None
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price value from text"""
        if not text:
            return None
        
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
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d %b %Y', '%d %B %Y']:
                try:
                    return datetime.strptime(text.strip(), fmt).date()
                except:
                    continue
        except:
            pass
        
        return datetime.now().date()
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize NCDEX dataframe to common format"""
        if df.empty:
            return df
        
        # Ensure required columns
        if 'date' not in df.columns:
            df['date'] = datetime.now().date()
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = df['date'].fillna(datetime.now().date())
        
        if 'price' not in df.columns:
            if 'min_price' in df.columns and 'max_price' in df.columns:
                df['price'] = (df['min_price'] + df['max_price']) / 2
            else:
                logger.warning("No price column found in NCDEX data")
                return pd.DataFrame()
        
        # Convert price to numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]
        
        # Add state/district as empty (NCDEX is exchange-level, not location-specific)
        if 'state' not in df.columns:
            df['state'] = ''
        if 'district' not in df.columns:
            df['district'] = ''
        
        # Ensure crop column exists
        if 'crop' not in df.columns:
            df['crop'] = ''
        
        return df[['date', 'state', 'district', 'crop', 'price', 'min_price', 'max_price']].copy() if all(col in df.columns for col in ['date', 'crop', 'price']) else df


def fetch_ncdex_data(commodity: Optional[str] = None, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch spot price data from NCDEX
    
    Args:
        commodity: Filter by commodity name (optional)
        limit: Maximum number of records to return
    
    Returns:
        DataFrame with market price data
    """
    try:
        fetcher = NCDEXFetcher()
        df = fetcher.fetch_spot_prices(commodity=commodity)
        
        if df is not None and not df.empty:
            if limit and len(df) > limit:
                df = df.head(limit)
            return df
        
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch NCDEX data: {e}")
        return None

