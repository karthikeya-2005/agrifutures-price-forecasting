import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import logging
import io
import time
import re
import os

def fetch_agmarknet_data(use_api=True, use_selenium=False, limit_commodities=None):
    """
    Fetch market prices data from https://agmarknet.gov.in
    
    Uses the new Agmarknet 2.0 API (recommended) or falls back to scraping.
    
    Args:
        use_api: If True, use the new API (recommended)
        use_selenium: If True and use_api=False, use Selenium for scraping
        limit_commodities: Limit number of commodities (for Selenium only)
    
    Returns:
        DataFrame or None on failure
    """
    # Try new API approach first (recommended)
    if use_api:
        try:
            from agmarknet_api_fetcher import fetch_agmarknet_api_data
            return fetch_agmarknet_api_data(limit=1000)
        except ImportError:
            logging.warning("agmarknet_api_fetcher not available, trying other methods")
        except Exception as e:
            logging.warning(f"API approach failed: {e}, trying other methods")
    
    # Try Selenium approach (old method, may not work with new website)
    if use_selenium:
        try:
            return _fetch_agmarknet_selenium(limit_commodities)
        except Exception as e:
            logging.warning(f"Selenium approach failed: {e}, trying requests-based approach")
    
    # Fallback to requests-based approach
    URLs = [
        "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx",
        "https://www.agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx",
        "https://agmarknet.gov.in/home",
        "https://www.agmarknet.gov.in/home"
    ]
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        for URL in URLs:
            try:
                r = session.get(URL, timeout=20)
                if r.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(r.text, 'html.parser')
                
                # Strategy 1: Search for table dynamically by regex on id attribute
                table = None
                id_pattern = re.compile(r'GridView\d+|gvPrice|gvCommodity|DataGrid|DataTable', re.I)
                for t in soup.find_all('table'):
                    if t.has_attr('id') and id_pattern.match(t['id']):
                        table = t
                        break
                
                # Strategy 2: Try to find a table that has headers like expected ones
                if not table:
                    for t in soup.find_all('table'):
                        headers = [th.text.strip().lower() for th in t.find_all(['th', 'td'])]
                        expected_keywords = {'date', 'state', 'commodity', 'price', 'modal', 'market', 'district'}
                        if any(keyword in ' '.join(headers[:10]) for keyword in expected_keywords):
                            # Check if it has multiple rows (data table)
                            rows = t.find_all('tr')
                            if len(rows) > 1:
                                table = t
                                break
                
                # Strategy 3: Try pandas read_html (works for well-formed tables)
                if not table:
                    try:
                        from io import StringIO
                        dfs = pd.read_html(StringIO(r.text))
                        for df_temp in dfs:
                            cols_lower = [col.lower() for col in df_temp.columns]
                            if any(keyword in ' '.join(cols_lower) for keyword in ['price', 'commodity', 'date', 'state']):
                                if len(df_temp) > 0:
                                    # Found a data table
                                    df = df_temp.copy()
                                    
                                    # Standardize column names
                                    rename_map = {
                                        'Date': 'date', 'date': 'date',
                                        'State': 'state', 'state': 'state',
                                        'District': 'district', 'district': 'district',
                                        'Market': 'market', 'market': 'market',
                                        'Commodity': 'crop', 'commodity': 'crop', 'crop': 'crop',
                                        'Variety': 'variety', 'variety': 'variety',
                                        'Grade': 'grade', 'grade': 'grade',
                                        'Min Price': 'min_price', 'min price': 'min_price',
                                        'Max Price': 'max_price', 'max price': 'max_price',
                                        'Modal Price': 'price', 'modal price': 'price', 'price': 'price'
                                    }
                                    
                                    for old_col, new_col in rename_map.items():
                                        if old_col in df.columns:
                                            df.rename(columns={old_col: new_col}, inplace=True)
                                    
                                    if 'date' in df.columns:
                                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                                    if 'price' in df.columns:
                                        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
                                    
                                    df = df.dropna(subset=['price'])
                                    
                                    if not df.empty:
                                        logging.info(f"Fetched {len(df)} records from Agmarknet using pandas read_html")
                                        return df
                    except:
                        pass
                
                # Strategy 4: Manual table parsing
                if table:
                    rows = table.find_all('tr')
                    if len(rows) > 1:
                        headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
                        if not headers:
                            # Try first row as headers
                            headers = [td.text.strip() for td in rows[0].find_all('td')]
                        
                        data = []
                        for row in rows[1:]:
                            cols = [td.text.strip() for td in row.find_all('td')]
                            if len(cols) == len(headers) and len(cols) > 0:
                                data.append(cols)
                        
                        if data:
                            df = pd.DataFrame(data, columns=headers)
                            
                            rename_map = {
                                'Date': 'date', 'date': 'date',
                                'State': 'state', 'state': 'state',
                                'District': 'district', 'district': 'district',
                                'Market': 'market', 'market': 'market',
                                'Commodity': 'crop', 'commodity': 'crop', 'crop': 'crop',
                                'Variety': 'variety', 'variety': 'variety',
                                'Grade': 'grade', 'grade': 'grade',
                                'Min Price': 'min_price', 'min price': 'min_price',
                                'Max Price': 'max_price', 'max price': 'max_price',
                                'Modal Price': 'price', 'modal price': 'price', 'price': 'price'
                            }
                            
                            for old_col, new_col in rename_map.items():
                                if old_col in df.columns:
                                    df.rename(columns={old_col: new_col}, inplace=True)
                            
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            if 'price' in df.columns:
                                df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
                            
                            df = df.dropna(subset=['price'])
                            
                            if not df.empty:
                                logging.info(f"Fetched {len(df)} records from Agmarknet using manual parsing")
                                return df
                
                # Strategy 5: Look for JSON data in script tags
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string:
                        # Look for price/commodity data in JavaScript
                        import json
                        json_pattern = re.compile(r'\{[^{}]*"(?:price|commodity|date|state)"[^{}]*\}', re.IGNORECASE)
                        matches = json_pattern.findall(script.string)
                        if matches:
                            records = []
                            for match in matches[:100]:  # Limit to avoid too many
                                try:
                                    data = json.loads(match)
                                    if any(key in str(data).lower() for key in ['price', 'commodity']):
                                        records.append(data)
                                except:
                                    pass
                            if records:
                                df = pd.DataFrame(records)
                                if 'price' in df.columns or 'modal_price' in df.columns:
                                    logging.info(f"Fetched {len(df)} records from Agmarknet using JSON extraction")
                                    return df
                
            except requests.exceptions.RequestException:
                continue
        
        logging.warning("Agmarknet: No data found with any scraping strategy")
        return None

    except Exception as e:
        logging.error(f"Failed to fetch agmarknet data: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None

def _fetch_agmarknet_selenium(limit_commodities=None):
    """
    Selenium-based scraper for Agmarknet (based on GitHub repo approach)
    Uses SearchCmmMkt.aspx page and handles pagination properly
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import NoSuchElementException, TimeoutException
    except ImportError:
        logging.error("Selenium not installed. Install with: pip install selenium")
        return None
    
    # Check for commodity CSV file
    commodities_csv = 'cloned_external_repo/CommodityAndCommodityHeads.csv'
    if not os.path.exists(commodities_csv):
        commodities_csv = 'CommodityAndCommodityHeads.csv'
        if not os.path.exists(commodities_csv):
            logging.warning("Commodity CSV not found. Using simple date-based approach.")
            return _fetch_agmarknet_simple()
    
    try:
        df_commodities = pd.read_csv(commodities_csv)
    except Exception as e:
        logging.error(f"Failed to read commodity CSV: {e}")
        return _fetch_agmarknet_simple()
    
    # Limit commodities if specified
    if limit_commodities:
        df_commodities = df_commodities.head(limit_commodities)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = None
    all_data = []
    
    try:
        # Try to use webdriver_manager for automatic driver management
        try:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except ImportError:
            # Fallback to system chromedriver
            driver = webdriver.Chrome(options=chrome_options)
        
        base_url = 'https://agmarknet.gov.in/SearchCmmMkt.aspx'
        
        for idx, row in df_commodities.iterrows():
            try:
                commodity = row.get('Commodity', '')
                commodity_head = row.get('CommodityHead', '')
                
                if not commodity or not commodity_head:
                    continue
                
                # Build URL with parameters (based on GitHub repo approach)
                # Use URL encoding for proper formatting
                from urllib.parse import urlencode
                
                params = {
                    "Tx_Commodity": str(commodity),
                    "Tx_CommodityHead": str(commodity_head),
                    "Tx_State": "0",
                    "Tx_District": "0",
                    "Tx_Market": "0",
                    "DateFrom": "01-Jan-2024",  # Recent data
                    "DateTo": datetime.datetime.now().strftime("%d-%b-%Y"),
                    "Fr_Date": "01-Jan-2024",
                    "To_Date": datetime.datetime.now().strftime("%d-%b-%Y"),
                    "Tx_Trend": "0",
                    "Tx_StateHead": "--Select--",
                    "Tx_DistrictHead": "--Select--",
                    "Tx_MarketHead": "--Select--",
                }
                
                query_string = urlencode(params)
                
                url = f"{base_url}?{query_string}"
                
                logging.info(f"  URL: {url[:100]}...")  # Log first 100 chars
                
                logging.info(f"Fetching data for {commodity} ({idx+1}/{len(df_commodities)})")
                driver.get(url)
                
                # Wait for page to load (based on repo: 10 seconds)
                time.sleep(10)
                
                # Extract data from all pages
                page_count = 0
                max_wait_attempts = 3
                
                while True:
                    try:
                        # Wait for table to load (try multiple times)
                        table = None
                        for attempt in range(max_wait_attempts):
                            soup = BeautifulSoup(driver.page_source, 'html.parser')
                            
                            # Try multiple table detection strategies
                            # Strategy 1: GridView pattern
                            table = soup.find('table', attrs={'id': re.compile('GridView.*', re.I)})
                            
                            # Strategy 2: Any table with price-related headers
                            if not table:
                                all_tables = soup.find_all('table')
                                for t in all_tables:
                                    headers_text = ' '.join([th.text.strip().lower() for th in t.find_all(['th', 'td'])])
                                    if any(keyword in headers_text for keyword in ['price', 'modal', 'commodity', 'date', 'state']):
                                        rows = t.find_all('tr')
                                        if len(rows) > 1:  # Has data rows
                                            table = t
                                            break
                            
                            # Strategy 3: Try pandas read_html
                            if not table:
                                try:
                                    from io import StringIO
                                    dfs = pd.read_html(StringIO(driver.page_source))
                                    for df_temp in dfs:
                                        if len(df_temp) > 0:
                                            cols_lower = [str(col).lower() for col in df_temp.columns]
                                            if any(keyword in ' '.join(cols_lower) for keyword in ['price', 'commodity', 'date']):
                                                # Found a data table - convert back to HTML for consistency
                                                table = soup.find('table')
                                                if table:
                                                    break
                                except:
                                    pass
                            
                            if table:
                                break
                            
                            # Wait a bit more if table not found
                            if attempt < max_wait_attempts - 1:
                                time.sleep(2)
                        
                        if not table:
                            logging.warning(f"No table found for {commodity} after {max_wait_attempts} attempts")
                            # Save page source for debugging
                            try:
                                with open(f'debug_agmarknet_{commodity}.html', 'w', encoding='utf-8') as f:
                                    f.write(driver.page_source)
                                logging.info(f"Saved page source to debug_agmarknet_{commodity}.html for inspection")
                            except:
                                pass
                            break
                        
                        # Extract headers
                        headers = [th.text.strip() for th in table.find_all('th')]
                        if not headers:
                            # Try first row
                            first_row = table.find('tr')
                            if first_row:
                                headers = [td.text.strip() for td in first_row.find_all(['th', 'td'])]
                        
                        # Extract data rows
                        rows = table.find_all('tr')[1:]  # Skip header
                        data_rows = []
                        for tr in rows:
                            cols = [td.text.strip() for td in tr.find_all('td')]
                            if len(cols) == len(headers) and len(cols) > 0:
                                data_rows.append(cols)
                        
                        if data_rows:
                            df_page = pd.DataFrame(data_rows, columns=headers)
                            all_data.append(df_page)
                            page_count += 1
                            logging.info(f"  Page {page_count}: {len(data_rows)} records")
                        
                        # Try to click next button (based on GitHub repo approach)
                        try:
                            # Wait for next button to be clickable
                            next_button = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, "//input[contains(@alt, '>')]"))
                            )
                            
                            # Check if button is enabled
                            if not next_button.is_enabled():
                                break
                            
                            next_button.click()
                            time.sleep(5)  # Wait for next page to load (repo uses 5 seconds)
                            
                        except (NoSuchElementException, TimeoutException):
                            # No more pages
                            break
                            
                    except Exception as e:
                        logging.error(f"Error processing page for {commodity}: {e}")
                        break
                
            except Exception as e:
                logging.error(f"Error processing commodity {commodity}: {e}")
                continue
        
        driver.quit()
        
        if not all_data:
            logging.warning("No data scraped from Agmarknet using Selenium")
            return None
        
        # Combine all pages
        df_all = pd.concat(all_data, ignore_index=True)
        
        # Standardize column names
        rename_map = {
            'Date': 'date',
            'State': 'state',
            'District': 'district',
            'Market': 'market',
            'Commodity': 'crop',
            'Variety': 'variety',
            'Grade': 'grade',
            'Min Price': 'min_price',
            'Max Price': 'max_price',
            'Modal Price': 'price'
        }
        
        for old_col, new_col in rename_map.items():
            if old_col in df_all.columns:
                df_all.rename(columns={old_col: new_col}, inplace=True)
        
        # Clean data
        if 'date' in df_all.columns:
            df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')
        if 'price' in df_all.columns:
            df_all['price'] = pd.to_numeric(df_all['price'].astype(str).str.replace(',', ''), errors='coerce')
        
        df_all = df_all.dropna(subset=['date', 'price'])
        
        logging.info(f"Successfully scraped {len(df_all)} records from Agmarknet using Selenium")
        return df_all
        
    except Exception as e:
        logging.error(f"Selenium scraping failed: {e}")
        if driver:
            driver.quit()
        return None

def _fetch_agmarknet_simple():
    """Simple fallback approach without commodity CSV"""
    # Use the existing requests-based approach as fallback
    URLs = [
        "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx",
    ]
    
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        
        for URL in URLs:
            try:
                r = session.get(URL, timeout=20)
                if r.status_code == 200:
                    from io import StringIO
                    dfs = pd.read_html(StringIO(r.text))
                    for df in dfs:
                        if len(df) > 0 and any('price' in str(col).lower() for col in df.columns):
                            return df
            except:
                continue
    except:
        pass
    
    return None

def fetch_data_gov_data():
    """
    Fetch data from Data.gov.in API
    Tries to use enhanced fetcher if available, otherwise returns empty DataFrame
    """
    try:
        from enhanced_market_data_fetcher import fetch_data_gov_data as enhanced_fetch
        return enhanced_fetch()
    except ImportError:
        # Fallback to stub
        import pandas as pd
        return pd.DataFrame()

def fetch_enam_data():
    """
    Fetch data from e-NAM
    Tries to use enhanced fetcher if available, otherwise returns empty DataFrame
    """
    try:
        from enhanced_market_data_fetcher import fetch_enam_data as enhanced_fetch
        return enhanced_fetch()
    except ImportError:
        import pandas as pd
        return pd.DataFrame()

def fetch_msamb_data():
    """
    Fetch data from MSAMB
    Tries to use enhanced fetcher if available, otherwise returns empty DataFrame
    """
    try:
        from enhanced_market_data_fetcher import fetch_msamb_data as enhanced_fetch
        return enhanced_fetch()
    except ImportError:
        import pandas as pd
        return pd.DataFrame()

def fetch_all_market_data():
    data_sources = [
        ("Agmarknet", fetch_agmarknet_data),
        ("DataGov", fetch_data_gov_data),
        ("e-NAM", fetch_enam_data),
        ("MSAMB", fetch_msamb_data),
    ]

    dfs = []
    for name, func in data_sources:
        logging.info(f"Fetching data from {name}")
        try:
            df = func()
            if df is not None and not df.empty:
                dfs.append(df)
            else:
                logging.warning(f"No data from {name}")
        except Exception as e:
            logging.error(f"Error fetching from {name}: {e}")

    if not dfs:
        logging.error("No market data fetched from any source.")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['date', 'state', 'district', 'crop'])
    combined_df = combined_df.sort_values(by=['date'])

    return combined_df
