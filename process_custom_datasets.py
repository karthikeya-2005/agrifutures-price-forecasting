"""Process datasets that need custom column mappings"""
from pathlib import Path
import pandas as pd
from datetime import datetime

print("="*80)
print("PROCESSING CUSTOM DATASETS")
print("="*80 + "\n")

all_processed = []

# 1. Agri Market Dataset 1
print("1. Processing: agri-market-dataset-1")
print("-" * 80)
try:
    csv_file = Path("data/kaggle/agri-market-dataset-1/Agri Market Dataset.csv")
    df = pd.read_csv(csv_file)
    print(f"  Loaded: {len(df)} records")
    
    # Map columns
    df_processed = pd.DataFrame()
    df_processed['date'] = pd.to_datetime(df['Reported Date'], format='%m/%d/%Y', errors='coerce')
    df_processed['state'] = df['State Name']
    df_processed['district'] = df['District Name']
    df_processed['crop'] = df['Commodity']
    df_processed['price'] = pd.to_numeric(df['Modal Price (Rs./Quintal)'], errors='coerce')
    df_processed['min_price'] = pd.to_numeric(df['Min Price (Rs./Quintal)'], errors='coerce')
    df_processed['max_price'] = pd.to_numeric(df['Max Price (Rs./Quintal)'], errors='coerce')
    
    # Clean
    df_processed = df_processed.dropna(subset=['date', 'state', 'crop', 'price'])
    df_processed = df_processed[df_processed['price'] > 0]
    
    df_processed['data_source'] = 'kaggle'
    df_processed['kaggle_dataset'] = 'agri-market-dataset-1'
    
    all_processed.append(df_processed)
    print(f"  [SUCCESS] Processed {len(df_processed):,} records")
except Exception as e:
    print(f"  [ERROR] {e}")

# 2. Daily Wholesale Commodity Prices India Mandis
print("\n2. Processing: daily-wholesale-commodity-prices-india-mandis")
print("-" * 80)
try:
    csv_file = Path("data/kaggle/daily-wholesale-commodity-prices-india-mandis/commodity_price.csv")
    df = pd.read_csv(csv_file)
    print(f"  Loaded: {len(df)} records")
    
    # Map columns
    df_processed = pd.DataFrame()
    df_processed['date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    df_processed['state'] = df['State']
    df_processed['district'] = df['District']
    df_processed['crop'] = df['Commodity']
    df_processed['price'] = pd.to_numeric(df['Modal_x0020_Price'], errors='coerce')
    df_processed['min_price'] = pd.to_numeric(df['Min_x0020_Price'], errors='coerce')
    df_processed['max_price'] = pd.to_numeric(df['Max_x0020_Price'], errors='coerce')
    
    # Clean
    df_processed = df_processed.dropna(subset=['date', 'state', 'crop', 'price'])
    df_processed = df_processed[df_processed['price'] > 0]
    
    df_processed['data_source'] = 'kaggle'
    df_processed['kaggle_dataset'] = 'daily-wholesale-commodity-prices-india-mandis'
    
    all_processed.append(df_processed)
    print(f"  [SUCCESS] Processed {len(df_processed):,} records")
except Exception as e:
    print(f"  [ERROR] {e}")

# 3. Wholesale Crop Price Dataset 2022-2023 (Excel with weather data!)
print("\n3. Processing: wholesale-crop-price-dataset-20222023")
print("-" * 80)
try:
    xlsx_file = Path("data/kaggle/wholesale-crop-price-dataset-20222023/Wholesale Crop Prices with Weather Data  India (20222023).xlsx")
    df = pd.read_excel(xlsx_file, engine='openpyxl')
    print(f"  Loaded: {len(df)} records")
    
    # Map columns
    df_processed = pd.DataFrame()
    # Create date from year and month
    df_processed['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df_processed['state'] = df['State']
    df_processed['district'] = 'Unknown'  # Not available
    df_processed['crop'] = df['Crop '].str.strip()
    df_processed['price'] = pd.to_numeric(df['Wholesale_Price [Rs. Per Quintal]'], errors='coerce')
    # Weather data as additional features (can be used later)
    df_processed['temperature'] = df['Temperature (Celsis)']
    df_processed['rainfall'] = df['Rainfall in mm']
    
    # Clean
    df_processed = df_processed.dropna(subset=['date', 'state', 'crop', 'price'])
    df_processed = df_processed[df_processed['price'] > 0]
    
    df_processed['data_source'] = 'kaggle'
    df_processed['kaggle_dataset'] = 'wholesale-crop-price-dataset-20222023'
    
    all_processed.append(df_processed)
    print(f"  [SUCCESS] Processed {len(df_processed):,} records (includes weather data!)")
except Exception as e:
    print(f"  [ERROR] {e}")

# 4. Vegetable and Fruits Prices in India (2010-2018)
print("\n4. Processing: vegetable-and-fruits-prices-in-india-2010-2018")
print("-" * 80)
try:
    csv_file = Path("data/kaggle/vegetable-and-fruits-prices-in-india-2010-2018/Vegetable and Fruits Prices in India.csv")
    df = pd.read_csv(csv_file)
    print(f"  Loaded: {len(df)} records")
    
    # Map columns
    df_processed = pd.DataFrame()
    df_processed['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df_processed['state'] = 'Unknown'  # Not available
    df_processed['district'] = 'Unknown'  # Not available
    df_processed['crop'] = df['Item_Name']
    df_processed['price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Clean
    df_processed = df_processed.dropna(subset=['date', 'crop', 'price'])
    df_processed = df_processed[df_processed['price'] > 0]
    
    df_processed['data_source'] = 'kaggle'
    df_processed['kaggle_dataset'] = 'vegetable-and-fruits-prices-in-india-2010-2018'
    
    all_processed.append(df_processed)
    print(f"  [SUCCESS] Processed {len(df_processed):,} records")
except Exception as e:
    print(f"  [ERROR] {e}")

# Combine all
if all_processed:
    print("\n" + "="*80)
    print("COMBINING ALL CUSTOM PROCESSED DATASETS")
    print("="*80 + "\n")
    
    combined = pd.concat(all_processed, ignore_index=True)
    print(f"Total records: {len(combined):,}")
    
    # Remove duplicates
    before = len(combined)
    if all(c in combined.columns for c in ['date', 'state', 'crop', 'price']):
        subset_cols = ['date', 'crop', 'price']
        if 'district' in combined.columns:
            subset_cols.insert(1, 'district')
        if 'state' in combined.columns:
            subset_cols.insert(0, 'state') if 'state' not in subset_cols else None
        
        combined = combined.drop_duplicates(subset=subset_cols, keep='first')
    after = len(combined)
    
    print(f"Removed {before - after:,} duplicates")
    print(f"Final records: {len(combined):,}")
    
    if not combined.empty:
        if 'date' in combined.columns:
            print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Commodities: {combined['crop'].nunique()}")
        if 'state' in combined.columns:
            print(f"States: {combined['state'].nunique()}")
    
    # Save
    output_file = Path('data/kaggle_combined/custom_processed_datasets.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Saved to: {output_file}")
    
    # Now combine with existing comprehensive dataset
    print("\n" + "="*80)
    print("MERGING WITH EXISTING COMPREHENSIVE DATASET")
    print("="*80 + "\n")
    
    existing_file = Path('data/kaggle_combined/all_kaggle_comprehensive_final.csv')
    if existing_file.exists():
        existing_df = pd.read_csv(existing_file)
        existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
        print(f"Loaded existing: {len(existing_df):,} records")
        
        # Merge
        final_df = pd.concat([existing_df, combined], ignore_index=True)
        
        # Remove duplicates
        before_merge = len(final_df)
        subset_cols = ['date', 'crop', 'price']
        if 'state' in final_df.columns:
            subset_cols.insert(0, 'state')
        if 'district' in final_df.columns:
            subset_cols.insert(1, 'district') if 'district' not in subset_cols else None
        
        final_df = final_df.drop_duplicates(subset=subset_cols, keep='first')
        after_merge = len(final_df)
        
        print(f"Total after merge: {len(final_df):,}")
        print(f"Removed {before_merge - after_merge:,} duplicates")
        
        # Save final
        final_file = Path('data/kaggle_combined/all_kaggle_final_complete.csv')
        final_df.to_csv(final_file, index=False)
        print(f"\n[SUCCESS] Final complete dataset saved to: {final_file}")
        print(f"  Total records: {len(final_df):,}")
        print(f"  Commodities: {final_df['crop'].nunique()}")
        if 'state' in final_df.columns:
            print(f"  States: {final_df['state'].nunique()}")
        if 'date' in final_df.columns:
            print(f"  Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    else:
        print("[WARNING] Existing comprehensive dataset not found")

print("\n[COMPLETE] Custom datasets processed!")

