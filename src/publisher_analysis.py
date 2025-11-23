"""
Publisher analysis utilities for analyzing news source patterns.

This module provides functions for extracting email domains, classifying
publisher types, and generating publisher statistics.

Example:
    >>> from src.publisher_analysis import get_publisher_stats, extract_email_domains
    >>> stats = get_publisher_stats(df)
    >>> domains = extract_email_domains(df)
"""
import pandas as pd
import re
from typing import Dict, Tuple


def extract_email_domains(df: pd.DataFrame, publisher_col: str = 'publisher') -> pd.DataFrame:
    """
    Extract email domains from publisher column for organizational analysis.
    
    Identifies publishers that are email addresses (e.g., 'author@benzinga.com')
    and extracts the domain portion for grouping and analysis.
    
    Args:
        df: DataFrame with publisher information
        publisher_col: Name of the publisher column (default: 'publisher')
        
    Returns:
        DataFrame with added columns:
        - is_email: Boolean indicating if publisher is an email address
        - domain: Extracted email domain (None for non-email publishers)
        
    Example:
        >>> df = pd.DataFrame({'publisher': ['author@benzinga.com', 'News Desk']})
        >>> result = extract_email_domains(df)
        >>> print(result[result['is_email']]['domain'].iloc[0])
        'benzinga.com'
    """
    df = df.copy()
    
    # Identify email-based publishers (contain '@' symbol)
    email_mask = df[publisher_col].str.contains('@', na=False)
    
    # Extract domain by splitting on '@' and taking the second part
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

