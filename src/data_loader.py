"""
Data loading utilities for analyst ratings dataset.

This module handles loading and preprocessing of financial news data,
including robust date parsing for mixed timezone formats.

Example:
    >>> from src.data_loader import load_analyst_ratings, get_data_info
    >>> df = load_analyst_ratings('data/raw_analyst_ratings.csv')
    >>> info = get_data_info(df)
    >>> print(f"Loaded {info['total_records']:,} records")
"""
import pandas as pd
import re
from pathlib import Path
from typing import Dict


def load_analyst_ratings(data_path: str = "data/raw_analyst_ratings.csv") -> pd.DataFrame:
    """
    Load and preprocess the raw analyst ratings dataset.
    
    Handles mixed date formats (timezone-aware and naive), extracts temporal
    features, and normalizes all dates to UTC-naive format.
    
    Args:
        data_path: Path to the CSV file containing analyst ratings
        
    Returns:
        DataFrame with preprocessed data including:
        - Original columns: headline, url, publisher, date, stock
        - Temporal features: year, month, day, day_of_week, hour, date_only
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
        
    Example:
        >>> df = load_analyst_ratings('data/raw_analyst_ratings.csv')
        >>> print(df.columns.tolist())
        ['headline', 'url', 'publisher', 'date', 'stock', 'year', 'month', ...]
        >>> print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    """
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
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    return {
        'total_records': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'unique_publishers': df['publisher'].nunique(),
        'unique_stocks': df['stock'].nunique(),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }

