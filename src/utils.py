"""
Utility functions for common operations across modules.

This module provides small, reusable utilities to increase cohesion
and reduce code duplication.
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all columns exist, False otherwise
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        >>> df = pd.DataFrame({'date': [1, 2], 'headline': ['a', 'b']})
        >>> validate_dataframe(df, ['date', 'headline'])
        True
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def ensure_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Ensure a column is datetime type, converting if necessary.
    
    Args:
        df: DataFrame to process
        date_column: Name of date column
        
    Returns:
        DataFrame with converted date column
        
    Example:
        >>> df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02']})
        >>> df = ensure_datetime(df, 'date')
        >>> df['date'].dtype
        datetime64[ns]
    """
    df = df.copy()
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df


def align_dates(df1: pd.DataFrame, df2: pd.DataFrame,
                date_col1: str = 'date', date_col2: str = 'date',
                how: str = 'inner', suffixes: Tuple[str, str] = ('_sentiment', '_stock')) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by date, extracting date-only for matching.
    
    This is a shared utility to reduce duplication of date alignment logic
    across different modules.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        date_col1: Date column name in first DataFrame
        date_col2: Date column name in second DataFrame
        how: Merge type ('inner', 'left', 'right', 'outer')
        suffixes: Tuple of suffixes for overlapping columns
        
    Returns:
        Tuple of aligned DataFrames (merged, merged) - both are the same merged DataFrame
        
    Example:
        >>> df1_aligned, df2_aligned = align_dates(sentiment_df, stock_df)
    """
    from typing import Tuple
    
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Ensure dates are datetime
    df1[date_col1] = pd.to_datetime(df1[date_col1], errors='coerce')
    df2[date_col2] = pd.to_datetime(df2[date_col2], errors='coerce')
    
    # Extract date only (without time)
    df1['_date_only'] = df1[date_col1].dt.date
    df2['_date_only'] = df2[date_col2].dt.date
    
    # Merge on date
    merged = pd.merge(
        df1,
        df2,
        on='_date_only',
        how=how,
        suffixes=suffixes
    )
    
    # Drop temporary column
    if '_date_only' in merged.columns:
        merged = merged.drop(columns=['_date_only'])
    
    return merged, merged


def safe_divide(numerator: Union[pd.Series, float], 
                denominator: Union[pd.Series, float],
                default: float = 0.0) -> Union[pd.Series, float]:
    """
    Safely divide two values, handling division by zero.
    
    Args:
        numerator: Numerator value or Series
        denominator: Denominator value or Series
        default: Value to return when denominator is zero
        
    Returns:
        Result of division or default value
        
    Example:
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(pd.Series([10, 20]), pd.Series([2, 0]))
        0    5.0
        1    0.0
        dtype: float64
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)
    else:
        return numerator / denominator if denominator != 0 else default


def create_output_dir(path: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
        
    Example:
        >>> output_dir = create_output_dir('output/results')
        >>> output_dir.exists()
        True
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
        
    Example:
        >>> format_number(1234.5678, decimals=2)
        '1,234.57'
    """
    return f"{value:,.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
        
    Example:
        >>> calculate_percentage_change(100, 110)
        10.0
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

