"""
Publisher analysis utilities.
"""
import pandas as pd
import re
from typing import Dict, Tuple


def extract_email_domains(df: pd.DataFrame, publisher_col: str = 'publisher') -> pd.DataFrame:
    """
    Extract email domains from publisher column.
    
    Args:
        df: DataFrame with publisher information
        publisher_col: Name of publisher column
        
    Returns:
        DataFrame with added 'domain' column
    """
    df = df.copy()
    
    # Identify email-based publishers
    email_mask = df[publisher_col].str.contains('@', na=False)
    
    # Extract domain
    df['is_email'] = email_mask
    df['domain'] = None
    df.loc[email_mask, 'domain'] = df.loc[email_mask, publisher_col].str.split('@').str[1]
    
    return df


def analyze_publisher_types(df: pd.DataFrame, publisher_col: str = 'publisher') -> Dict[str, int]:
    """
    Analyze different types of publishers.
    
    Args:
        df: DataFrame with publisher information
        publisher_col: Name of publisher column
        
    Returns:
        Dictionary with publisher type counts
    """
    email_count = df[publisher_col].str.contains('@', na=False).sum()
    newsdesk_count = df[publisher_col].str.contains('newsdesk', case=False, na=False).sum()
    insights_count = df[publisher_col].str.contains('insights', case=False, na=False).sum()
    
    return {
        'email_publishers': email_count,
        'newsdesk_publishers': newsdesk_count,
        'insights_publishers': insights_count,
        'other_publishers': len(df) - email_count - newsdesk_count - insights_count
    }


def get_publisher_stats(df: pd.DataFrame, publisher_col: str = 'publisher') -> pd.DataFrame:
    """
    Get statistics for each publisher.
    
    Args:
        df: DataFrame with publisher information
        publisher_col: Name of publisher column
        
    Returns:
        DataFrame with publisher statistics
    """
    stats = df.groupby(publisher_col).agg({
        'headline': 'count',
        'stock': 'nunique',
        'date': ['min', 'max']
    }).reset_index()
    
    stats.columns = ['publisher', 'article_count', 'unique_stocks', 'first_article', 'last_article']
    stats = stats.sort_values('article_count', ascending=False)
    
    return stats

