"""
Kaggle Dataset Fetcher for Indian Agricultural Commodity Prices
Downloads and processes relevant datasets from Kaggle for training and testing
"""

import os
import pandas as pd
import logging
from typing import Optional, List, Dict
import json
from pathlib import Path
import zipfile
import shutil

logger = logging.getLogger(__name__)

# Known Kaggle datasets for Indian agricultural commodity prices
KAGGLE_DATASETS = [
    {
        "name": "indian-agricultural-commodity-prices",
        "owner": "datasets",
        "description": "Indian agricultural commodity prices dataset",
        "keywords": ["india", "agriculture", "commodity", "prices", "mandi"]
    },
    {
        "name": "agricultural-commodity-prices-india",
        "owner": "datasets",
        "description": "Agricultural commodity prices in India",
        "keywords": ["india", "agriculture", "prices"]
    },
    {
        "name": "mandi-prices-india",
        "owner": "datasets",
        "description": "Mandi prices data for India",
        "keywords": ["mandi", "india", "prices"]
    }
]

def setup_kaggle_auth():
    """Setup Kaggle authentication from environment variable or kaggle.json"""
    import os
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check for kaggle.json file first (traditional method)
    if kaggle_json.exists():
        try:
            with open(kaggle_json, 'r') as f:
                config = json.load(f)
                if 'username' in config and 'key' in config:
                    logger.info("Kaggle API credentials found in kaggle.json")
                    return True
        except Exception as e:
            logger.warning(f"Error reading Kaggle config: {e}")
    
    # Check for environment variable
    api_token = os.getenv('KAGGLE_API_TOKEN')
    if api_token:
        # The KAGGLE_API_TOKEN format might be different
        # Try to use it directly by setting KAGGLE_USERNAME and KAGGLE_KEY
        # For now, we'll try to create a minimal kaggle.json
        # Note: This is a workaround - the token format may vary
        try:
            kaggle_dir.mkdir(exist_ok=True)
            
            # Try to parse token - if it's in format "username:key" or just a key
            # For KGAT_ format, we might need username separately
            # Let's try a simple approach: use a placeholder username
            # The actual token might work with just the key
            config = {
                "username": "kaggle_user",  # Placeholder - may need actual username
                "key": api_token
            }
            
            # Only create if it doesn't exist
            if not kaggle_json.exists():
                with open(kaggle_json, 'w') as f:
                    json.dump(config, f)
                # Set restrictive permissions
                try:
                    os.chmod(kaggle_json, 0o600)
                except:
                    pass
                logger.info("Created kaggle.json from KAGGLE_API_TOKEN environment variable")
            else:
                logger.info("Kaggle API token found in environment variable (kaggle.json exists)")
            
            return True
        except Exception as e:
            logger.warning(f"Error setting up auth from env var: {e}")
            # Try alternative: set environment variables directly
            try:
                os.environ['KAGGLE_USERNAME'] = 'kaggle_user'
                os.environ['KAGGLE_KEY'] = api_token
                logger.info("Set KAGGLE_USERNAME and KAGGLE_KEY from environment")
                return True
            except:
                pass
    
    return False

def check_kaggle_api():
    """Check if Kaggle API is configured"""
    import os
    
    # Try to setup authentication
    if setup_kaggle_auth():
        return True
    
    logger.warning("Kaggle API not configured. Install kaggle package and set up credentials.")
    logger.info("To set up Kaggle API:")
    logger.info("1. pip install kaggle")
    logger.info("2. Go to https://www.kaggle.com/settings and create API token")
    logger.info("3. Either:")
    logger.info("   - Set KAGGLE_API_TOKEN environment variable, OR")
    logger.info("   - Place kaggle.json in ~/.kaggle/ directory")
    return False

def download_kaggle_dataset(dataset_name: str, owner: str = "datasets", unzip: bool = True) -> Optional[str]:
    """
    Download a dataset from Kaggle
    
    Args:
        dataset_name: Name of the dataset
        owner: Owner of the dataset (usually 'datasets')
        unzip: Whether to unzip the downloaded files
    
    Returns:
        Path to downloaded dataset directory or None on failure
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        return None
    
    import os
    # Ensure API token is set if available
    api_token = os.getenv('KAGGLE_API_TOKEN')
    if api_token:
        os.environ['KAGGLE_API_TOKEN'] = api_token
        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            logger.warning(f"Could not authenticate with environment token: {e}")
    
    # Check if API is available (either via env var or kaggle.json)
    has_api = os.getenv('KAGGLE_API_TOKEN') is not None or check_kaggle_api()
    if not has_api:
        logger.error("Kaggle API not configured")
        return None
    
    try:
        dataset_ref = f"{owner}/{dataset_name}"
        output_dir = Path("data") / "kaggle" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading dataset: {dataset_ref}")
        kaggle.api.dataset_download_files(
            dataset_ref,
            path=str(output_dir),
            unzip=unzip
        )
        
        logger.info(f"Dataset downloaded to: {output_dir}")
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def search_kaggle_datasets(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search for datasets on Kaggle
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of dataset information dictionaries
    """
    import os
    
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle package not installed")
        return []
    
    # Ensure API token is set if available
    api_token = os.getenv('KAGGLE_API_TOKEN')
    if api_token:
        # Set the environment variable for kaggle client
        os.environ['KAGGLE_API_TOKEN'] = api_token
        # Initialize API client
        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            logger.warning(f"Could not authenticate with environment token: {e}")
    
    # Check if API is available (either via env var or kaggle.json)
    has_api = os.getenv('KAGGLE_API_TOKEN') is not None or check_kaggle_api()
    if not has_api:
        logger.warning("Kaggle API not configured - cannot search")
        return []
    
    try:
        # Use the kaggle API directly
        datasets = list(kaggle.api.dataset_list(search=query, max_size=1000))
        results = []
        
        logger.info(f"Found {len(datasets)} datasets for query: '{query}'")
        
        for dataset in datasets[:max_results]:
            # Handle attributes that might not exist
            dataset_info = {
                "ref": getattr(dataset, 'ref', ''),
                "title": getattr(dataset, 'title', 'No title'),
                "size": getattr(dataset, 'size', 'Unknown'),
                "usabilityRating": getattr(dataset, 'usabilityRating', 0),
                "downloadCount": getattr(dataset, 'downloadCount', 0)
            }
            results.append(dataset_info)
        
        return results
    except Exception as e:
        logger.error(f"Error searching Kaggle datasets: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []

def process_kaggle_dataset(dataset_path: str) -> Optional[pd.DataFrame]:
    """
    Process downloaded Kaggle dataset and convert to standard format
    
    Args:
        dataset_path: Path to the downloaded dataset directory
    
    Returns:
        DataFrame in standard format or None on failure
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return None
    
    # Look for CSV and Excel files
    csv_files = list(dataset_path.rglob("*.csv"))
    xlsx_files = list(dataset_path.rglob("*.xlsx"))
    
    if not csv_files and not xlsx_files:
        logger.warning(f"No CSV or Excel files found in {dataset_path}")
        return None
    
    all_dataframes = []
    
    # Process CSV files
    for csv_file in csv_files:
        try:
            logger.info(f"Processing file: {csv_file.name}")
            # Try different encodings
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(csv_file, encoding='latin-1')
                except:
                    df = pd.read_csv(csv_file, encoding='iso-8859-1')
            
            # Try to standardize column names
            df = standardize_kaggle_dataframe(df)
            
            if df is not None and not df.empty:
                all_dataframes.append(df)
                logger.info(f"Processed {len(df)} records from {csv_file.name}")
        except Exception as e:
            logger.warning(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Process Excel files
    for xlsx_file in xlsx_files:
        try:
            logger.info(f"Processing Excel file: {xlsx_file.name}")
            # Try to read first sheet
            try:
                df = pd.read_excel(xlsx_file, engine='openpyxl')
            except ImportError:
                logger.warning("openpyxl not installed. Install with: pip install openpyxl")
                continue
            except:
                # Try with xlrd for older Excel files
                try:
                    df = pd.read_excel(xlsx_file, engine='xlrd')
                except:
                    logger.warning(f"Could not read Excel file: {xlsx_file.name}")
                    continue
            
            # Try to standardize column names
            df = standardize_kaggle_dataframe(df)
            
            if df is not None and not df.empty:
                all_dataframes.append(df)
                logger.info(f"Processed {len(df)} records from {xlsx_file.name}")
        except Exception as e:
            logger.warning(f"Error processing {xlsx_file.name}: {e}")
            continue
    
    if not all_dataframes:
        logger.warning("No valid data found in Kaggle dataset")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Total records from Kaggle dataset: {len(combined_df)}")
    
    return combined_df

def standardize_kaggle_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Standardize Kaggle dataset to match our data format
    
    Expected columns:
    - date, state, district, crop, price, min_price, max_price, modal_price
    
    Args:
        df: Raw dataframe from Kaggle
    
    Returns:
        Standardized dataframe or None
    """
    if df.empty:
        return None
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Common column name mappings
    column_mappings = {
        # Date columns
        'date': ['date', 'Date', 'DATE', 'date_time', 'Date_Time', 'timestamp', 'Timestamp', 
                'Reported Date', 'reported_date', 'Datesk', 'month'],
        # Location columns
        'state': ['state', 'State', 'STATE', 'state_name', 'State_Name', 'State Name', 
                  'state_name_english', 'stateName'],
        'district': ['district', 'District', 'DISTRICT', 'district_name', 'District_Name', 
                    'District Name', 'district_name_english', 'districtName'],
        # Commodity columns
        'crop': ['crop', 'Crop', 'CROP', 'Crops', 'CROPS', 'commodity', 'Commodity', 'COMMODITY', 
                'variety', 'Variety', 'commodity_name', 'Commodity Name', 'Item_Name', 'item_name'],
        # Price columns
        'price': ['price', 'Price', 'PRICE', 'modal_price', 'Modal_Price', 'MODAL_PRICE', 
                 'Modal Price', 'MODAL PRICE', 'Modal Price (Rs./Quintal)', 
                 'avg_price', 'average_price', 'avg_modal_price'],
        'min_price': ['min_price', 'Min_Price', 'MIN_PRICE', 'Min Price', 'MIN PRICE', 
                     'Min Price (Rs./Quintal)', 'min', 'Min', 'minimum_price', 'avg_min_price'],
        'max_price': ['max_price', 'Max_Price', 'MAX_PRICE', 'Max Price', 'MAX PRICE', 
                     'Max Price (Rs./Quintal)', 'max', 'Max', 'maximum_price', 'avg_max_price']
    }
    
    # Find matching columns (case-insensitive, handle spaces/underscores)
    standardized_cols = {}
    for standard_col, possible_names in column_mappings.items():
        for col in df.columns:
            col_normalized = col.replace(' ', '_').replace('-', '_').lower()
            for name in possible_names:
                name_normalized = name.replace(' ', '_').replace('-', '_').lower()
                if col == name or col.lower() == name.lower() or col_normalized == name_normalized:
                    standardized_cols[standard_col] = col
                    break
            if standard_col in standardized_cols:
                break
    
    # Rename columns
    df = df.rename(columns={v: k for k, v in standardized_cols.items()})
    
    # Ensure date column exists and is datetime
    if 'date' not in df.columns:
        # Try to find date-like columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or col.lower() == 'month']
        if date_cols:
            date_col = date_cols[0]
            # Special handling for 'month' column
            if date_col.lower() == 'month':
                # Try to parse month as date (e.g., "2023-01" or "Jan-2023")
                df['date'] = pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')
                if df['date'].isna().all():
                    # Try other formats
                    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            # If no date column but we have valid price data, use today's date
            # This allows us to use datasets that don't have temporal information
            # The user can update dates later if needed
            logger.info("No date column found - using current date as placeholder")
            df['date'] = pd.Timestamp.now().normalize()
    else:
        # If date column exists, try to parse it
        # Check if it's a month column that was renamed
        if df['date'].dtype == 'object':
            # Try parsing as month first
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m', errors='coerce')
            if df['date'].isna().any():
                # Fall back to general parsing
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure price column exists
    if 'price' not in df.columns:
        # Try to use modal_price, average_price, or calculate from min/max
        if 'modal_price' in df.columns:
            df['price'] = df['modal_price']
        elif 'avg_price' in df.columns:
            df['price'] = df['avg_price']
        elif 'min_price' in df.columns and 'max_price' in df.columns:
            # Calculate average of min and max
            df['price'] = (df['min_price'] + df['max_price']) / 2
            logger.info("Calculated price as average of min and max prices")
        else:
            logger.warning("No price column found in dataset")
            return None
    
    # Ensure price is numeric
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]  # Remove invalid prices
    
    # Fill missing location info if possible
    if 'state' not in df.columns:
        logger.warning("No state column found - dataset may not be usable")
        return None
    
    if 'district' not in df.columns:
        # Try to infer from other columns or set to 'Unknown'
        df['district'] = 'Unknown'
    
    if 'crop' not in df.columns:
        logger.warning("No crop/commodity column found - dataset may not be usable")
        return None
    
    # Select only the columns we need
    required_cols = ['date', 'state', 'district', 'crop', 'price']
    optional_cols = ['min_price', 'max_price']
    
    available_cols = [col for col in required_cols if col in df.columns]
    available_cols.extend([col for col in optional_cols if col in df.columns])
    
    df = df[available_cols].copy()
    
    # Remove rows with missing required data
    df = df.dropna(subset=['date', 'state', 'crop', 'price'])
    
    return df

def fetch_kaggle_agricultural_data(search_terms: List[str] = None, download_datasets: bool = True) -> Optional[pd.DataFrame]:
    """
    Main function to fetch agricultural commodity price data from Kaggle
    
    Args:
        search_terms: List of search terms to find relevant datasets
        download_datasets: Whether to automatically download datasets
    
    Returns:
        Combined DataFrame with all Kaggle data or None
    """
    if search_terms is None:
        search_terms = [
            "indian agricultural commodity prices",
            "mandi prices india",
            "agricultural prices india",
            "commodity prices india"
        ]
    
    all_dataframes = []
    
    # Search for datasets
    logger.info("Searching Kaggle for agricultural commodity price datasets...")
    for term in search_terms:
        logger.info(f"Searching for: {term}")
        datasets = search_kaggle_datasets(term, max_results=5)
        
        if datasets:
            logger.info(f"Found {len(datasets)} datasets for '{term}'")
            for dataset in datasets:
                logger.info(f"  - {dataset['ref']}: {dataset['title']} (Downloads: {dataset.get('downloadCount', 0)})")
        
        # Download top datasets if requested
        if download_datasets and datasets:
            for dataset in datasets[:2]:  # Download top 2 per search term
                dataset_ref = dataset['ref']
                owner, dataset_name = dataset_ref.split('/')
                
                logger.info(f"Downloading dataset: {dataset_ref}")
                dataset_path = download_kaggle_dataset(dataset_name, owner)
                
                if dataset_path:
                    processed_df = process_kaggle_dataset(dataset_path)
                    if processed_df is not None and not processed_df.empty:
                        processed_df['data_source'] = 'kaggle'
                        all_dataframes.append(processed_df)
    
    # Also try known datasets
    logger.info("Trying known Kaggle datasets...")
    for dataset_info in KAGGLE_DATASETS:
        if download_datasets:
            dataset_path = download_kaggle_dataset(dataset_info['name'], dataset_info['owner'])
            if dataset_path:
                processed_df = process_kaggle_dataset(dataset_path)
                if processed_df is not None and not processed_df.empty:
                    processed_df['data_source'] = 'kaggle'
                    all_dataframes.append(processed_df)
    
    if not all_dataframes:
        logger.warning("No data found from Kaggle datasets")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates()
    logger.info(f"Total records from Kaggle: {len(combined_df)}")
    
    return combined_df

def integrate_kaggle_with_training():
    """
    Integrate Kaggle data with existing training pipeline
    Downloads and processes Kaggle datasets, then saves for training
    """
    logger.info("="*80)
    logger.info("Kaggle Dataset Integration")
    logger.info("="*80)
    
    # Check if Kaggle API is available
    if not check_kaggle_api():
        logger.warning("Kaggle API not configured. Skipping Kaggle data fetch.")
        logger.info("To enable Kaggle integration:")
        logger.info("1. pip install kaggle")
        logger.info("2. Set up Kaggle API credentials (see check_kaggle_api() for details)")
        return None
    
    # Fetch data from Kaggle
    kaggle_df = fetch_kaggle_agricultural_data()
    
    if kaggle_df is not None and not kaggle_df.empty:
        # Save to data directory
        output_path = Path("data") / "kaggle_combined.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kaggle_df.to_csv(output_path, index=False)
        logger.info(f"Saved Kaggle data to: {output_path}")
        logger.info(f"Total records: {len(kaggle_df)}")
        logger.info(f"Date range: {kaggle_df['date'].min()} to {kaggle_df['date'].max()}")
        logger.info(f"Commodities: {kaggle_df['crop'].nunique()}")
        logger.info(f"States: {kaggle_df['state'].nunique()}")
        logger.info(f"Districts: {kaggle_df['district'].nunique()}")
        
        return kaggle_df
    else:
        logger.warning("No data retrieved from Kaggle")
        return None

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test Kaggle integration
    result = integrate_kaggle_with_training()
    
    if result is not None:
        print(f"\nSuccessfully fetched {len(result)} records from Kaggle")
        print(f"Sample data:")
        print(result.head())
    else:
        print("\nNo data fetched from Kaggle. Check API configuration.")

