"""
Data loading utilities for analyst ratings and stock price datasets.

This module provides class-based interfaces for loading and preprocessing
financial news data and stock price data, including robust date parsing
for mixed timezone formats.

Example:
    >>> loader = DataLoader()
    >>> df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
    >>> info = loader.get_data_info(df)
    >>> print(f"Loaded {info['total_records']:,} records")
"""
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Optional, List


class DataLoader:
    """
    A class for loading and preprocessing financial datasets.
    
    Provides methods for loading analyst ratings data and stock price data,
    with robust date parsing and data normalization capabilities.
    
    Attributes:
        default_analyst_ratings_path: Default path for analyst ratings data
        loaded_data: Dictionary to cache loaded datasets
        
    Example:
        >>> loader = DataLoader()
        >>> news_df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
        >>> stock_df = loader.load_stock_price_data('data/AAPL.csv')
        >>> info = loader.get_data_info(news_df)
    """
    
    def __init__(self, default_analyst_ratings_path: str = "data/raw_analyst_ratings.csv"):
        """
        Initialize the DataLoader.
        
        Args:
            default_analyst_ratings_path: Default path for analyst ratings CSV
        """
        self.default_analyst_ratings_path = default_analyst_ratings_path
        self.loaded_data: Dict[str, pd.DataFrame] = {}
    
    def load_analyst_ratings(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the raw analyst ratings dataset.
        
        Handles mixed date formats (timezone-aware and naive), extracts temporal
        features, and normalizes all dates to UTC-naive format.
        
        Args:
            data_path: Path to the CSV file containing analyst ratings.
                      If None, uses default path from initialization.
        
        Returns:
            DataFrame with preprocessed data including:
            - Original columns: headline, url, publisher, date, stock
            - Temporal features: year, month, day, day_of_week, hour, date_only
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
            >>> print(df.columns.tolist())
            ['headline', 'url', 'publisher', 'date', 'stock', 'year', 'month', ...]
            >>> print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        """
        if data_path is None:
            data_path = self.default_analyst_ratings_path
        
        # Check cache
        if data_path in self.loaded_data:
            return self.loaded_data[data_path].copy()
        
        # Load CSV file
        df = pd.read_csv(data_path)
        
        # Parse date column - handle mixed formats (some with timezone, some without)
        # Some dates have format "2020-06-05 10:30:54-04:00" (with timezone)
        # Others have format "2020-05-22 00:00:00" (without timezone)
        # Strategy: Parse timezone-aware and naive dates separately to avoid conflicts
        
        # Identify dates with timezone (pattern: ends with +/-HH:MM)
        tz_pattern = r'[-+]\d{2}:\d{2}$'
        has_tz_mask = df['date'].str.contains(tz_pattern, na=False, regex=True)
        
        # Parse dates separately for each group to avoid format conflicts
        dates_parsed = pd.Series(index=df.index, dtype='object')
        
        # Parse timezone-aware dates
        if has_tz_mask.any():
            dates_parsed[has_tz_mask] = pd.to_datetime(
                df.loc[has_tz_mask, 'date'], errors='coerce'
            )
        
        # Parse timezone-naive dates
        if (~has_tz_mask).any():
            dates_parsed[~has_tz_mask] = pd.to_datetime(
                df.loc[~has_tz_mask, 'date'], errors='coerce'
            )
        
        # Convert to datetime64[ns] and normalize timezones
        # Convert all timezone-aware dates to UTC, then remove timezone
        def normalize_datetime(dt):
            if pd.isna(dt):
                return pd.NaT
            if hasattr(dt, 'tz') and dt.tz is not None:
                # Convert to UTC and remove timezone
                return dt.tz_convert('UTC').tz_localize(None)
            return dt
        
        # Apply normalization
        dates_normalized = dates_parsed.apply(normalize_datetime)
        
        # Assign back to dataframe
        df['date'] = pd.to_datetime(dates_normalized, errors='coerce')
        
        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['date_only'] = df['date'].dt.date
        
        # Cache the loaded data
        self.loaded_data[data_path] = df.copy()
        
        return df
    
    def load_stock_price_data(self, data_path: str, normalize_columns: bool = True) -> pd.DataFrame:
        """
        Load and preprocess stock price data from CSV file.
        
        Handles various column name formats (Date/date, Close/close, etc.)
        and normalizes them to lowercase. Ensures required OHLCV columns exist.
        
        Args:
            data_path: Path to the CSV file containing stock price data
            normalize_columns: Whether to normalize column names to lowercase (default: True)
        
        Returns:
            DataFrame with normalized column names:
            - date: Date column (datetime)
            - open: Opening price
            - high: High price
            - low: Low price
            - close: Closing price
            - volume: Trading volume
        
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If required columns are missing
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_stock_price_data('data/AAPL.csv')
            >>> print(df.columns.tolist())
            ['date', 'open', 'high', 'low', 'close', 'volume']
            >>> print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        """
        # Check cache
        cache_key = f"stock_{data_path}"
        if cache_key in self.loaded_data:
            return self.loaded_data[cache_key].copy()
        
        # Load CSV file
        df = pd.read_csv(data_path)
        
        # Normalize column names to lowercase if requested
        if normalize_columns:
            df.columns = df.columns.str.lower()
        
        # Map common date column names
        date_cols = ['date', 'datetime', 'time', 'timestamp']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found. Expected one of: {date_cols}")
        
        # Rename date column to 'date'
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
        
        # Parse date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Verify required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove rows with missing dates
        df = df.dropna(subset=['date'])
        
        # Cache the loaded data
        self.loaded_data[cache_key] = df.copy()
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with dataset information including:
            - total_records: Number of records
            - date_range: Tuple of (min_date, max_date) if date column exists
            - unique_publishers: Number of unique publishers (if column exists)
            - unique_stocks: Number of unique stocks (if column exists)
            - columns: List of column names
            - missing_values: Dictionary of missing value counts per column
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_analyst_ratings()
            >>> info = loader.get_data_info(df)
            >>> print(f"Total records: {info['total_records']}")
        """
        info = {
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Add date range if date column exists
        if 'date' in df.columns:
            info['date_range'] = (df['date'].min(), df['date'].max())
        
        # Add publisher info if available
        if 'publisher' in df.columns:
            info['unique_publishers'] = df['publisher'].nunique()
        
        # Add stock info if available
        if 'stock' in df.columns:
            info['unique_stocks'] = df['stock'].nunique()
        
        return info
    
    def load_multiple_stocks(self, data_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple stock price data files.
        
        Args:
            data_paths: List of paths to stock price CSV files
        
        Returns:
            Dictionary mapping stock ticker (from filename) to DataFrame
        
        Example:
            >>> loader = DataLoader()
            >>> stocks = loader.load_multiple_stocks(['data/AAPL.csv', 'data/MSFT.csv'])
            >>> print(stocks.keys())
            dict_keys(['AAPL', 'MSFT'])
        """
        stocks = {}
        for path in data_paths:
            # Extract ticker from filename
            ticker = Path(path).stem.upper()
            stocks[ticker] = self.load_stock_price_data(path)
        return stocks
    
    def clear_cache(self):
        """Clear the loaded data cache."""
        self.loaded_data.clear()


# Backward compatibility functions
def load_analyst_ratings(data_path: str = "data/raw_analyst_ratings.csv") -> pd.DataFrame:
    """
    Backward compatibility function for loading analyst ratings.
    
    Args:
        data_path: Path to the CSV file containing analyst ratings
        
    Returns:
        DataFrame with preprocessed data
    """
    loader = DataLoader()
    return loader.load_analyst_ratings(data_path)


def load_stock_price_data(data_path: str) -> pd.DataFrame:
    """
    Backward compatibility function for loading stock price data.
    
    Args:
        data_path: Path to the CSV file containing stock price data
        
    Returns:
        DataFrame with normalized column names
    """
    loader = DataLoader()
    return loader.load_stock_price_data(data_path)


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Backward compatibility function for getting data info.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    loader = DataLoader()
    return loader.get_data_info(df)

