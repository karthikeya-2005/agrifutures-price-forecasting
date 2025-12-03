import pandas as pd

def preprocess_market_data(market_df):
    """
    Preprocess raw market price data.
    :param market_df: pandas DataFrame with market price data
    :return: cleaned and preprocessed DataFrame
    """
    market_df['date'] = pd.to_datetime(market_df['date'], errors='coerce')
    market_df = market_df.dropna(subset=['date'])
    market_df['price'] = pd.to_numeric(market_df['price'], errors='coerce')
    market_df = market_df.dropna(subset=['price'])
    return market_df

def merge_datasets(market_df, weather_df):
    """
    Merge market and weather data on date and location.
    :param market_df: DataFrame with market price data
    :param weather_df: DataFrame with weather data
    :return: merged DataFrame
    """
    merged_df = pd.merge(market_df, weather_df, on=['date', 'state', 'district'], how='left')
    # Use forward fill and backward fill (updated syntax)
    merged_df = merged_df.ffill().bfill()
    return merged_df

def create_features(df):
    """
    Create features dataframe for model input.
    :param df: merged DataFrame
    :return: DataFrame with features and target variable
    """
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df
