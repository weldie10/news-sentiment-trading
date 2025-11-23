"""
News Sentiment Trading - Source Package

This package provides modules for analyzing financial news sentiment
and its correlation with stock price movements.

Main Modules:
    - data_loader: Data loading and preprocessing utilities
    - text_analysis: NLP and text processing for headlines
    - publisher_analysis: Publisher pattern analysis
    - technical_indicators: TA-Lib integration for technical analysis
    - sentiment_analysis: Sentiment scoring using TextBlob and VADER
    - correlation_analysis: Correlation between sentiment and stock returns

Example:
    >>> from src import DataLoader, SentimentAnalyzer, CorrelationAnalyzer
    >>> loader = DataLoader()
    >>> df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
    >>> analyzer = SentimentAnalyzer()
    >>> df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
"""
__version__ = "0.1.0"

# Core data loading
from .data_loader import load_analyst_ratings, get_data_info

# Text analysis
from .text_analysis import (
    preprocess_text,
    extract_ngrams,
    get_top_keywords,
    extract_financial_keywords
)

# Publisher analysis
from .publisher_analysis import (
    extract_email_domains,
    analyze_publisher_types,
    get_publisher_stats
)

# Technical indicators (optional - requires TA-Lib)
try:
    from .technical_indicators import TechnicalAnalyzer
    __all__ = [
        'load_analyst_ratings',
        'get_data_info',
        'preprocess_text',
        'extract_ngrams',
        'get_top_keywords',
        'extract_financial_keywords',
        'extract_email_domains',
        'analyze_publisher_types',
        'get_publisher_stats',
        'TechnicalAnalyzer',
        'SentimentAnalyzer',
        'CorrelationAnalyzer',
    ]
except ImportError:
    __all__ = [
        'load_analyst_ratings',
        'get_data_info',
        'preprocess_text',
        'extract_ngrams',
        'get_top_keywords',
        'extract_financial_keywords',
        'extract_email_domains',
        'analyze_publisher_types',
        'get_publisher_stats',
        'SentimentAnalyzer',
        'CorrelationAnalyzer',
    ]

# Sentiment analysis
from .sentiment_analysis import SentimentAnalyzer

# Correlation analysis
from .correlation_analysis import CorrelationAnalyzer
