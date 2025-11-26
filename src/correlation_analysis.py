"""
Correlation analysis between news sentiment and stock price movements.

This module provides a class-based interface for analyzing correlations
between sentiment scores and stock returns, including statistical testing.

Example:
    >>> analyzer = CorrelationAnalyzer()
    >>> correlation = analyzer.calculate_correlation(sentiment_df, stock_df)
    >>> results = analyzer.analyze_lagged_correlations(sentiment_df, stock_df)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
try:
    from .sentiment_analysis import SentimentAnalyzer
    from .utils import align_dates
except ImportError:
    from sentiment_analysis import SentimentAnalyzer
    from utils import align_dates


class CorrelationAnalyzer:
    """
    A class for analyzing correlations between sentiment and stock returns.
    
    Provides methods for calculating Pearson correlation, statistical
    significance testing, and lagged correlation analysis.
    
    Attributes:
        sentiment_analyzer: SentimentAnalyzer instance
        
    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> correlation = analyzer.calculate_correlation(sentiment_df, stock_df)
        >>> print(f"Correlation: {correlation['pearson_r']:.3f}")
    """
    
    def __init__(self):
        """Initialize the CorrelationAnalyzer."""
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def calculate_daily_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Calculate daily percentage returns from closing prices.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column
            
        Returns:
            Series of daily returns (as percentages)
            
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> returns = analyzer.calculate_daily_returns(stock_df, 'close')
        """
        return df[price_col].pct_change() * 100
    
    def align_data(self, sentiment_df: pd.DataFrame, stock_df: pd.DataFrame,
                  sentiment_date_col: str = 'date', stock_date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align sentiment and stock data by date.
        
        Matches sentiment scores to corresponding trading days, handling
        after-hours and pre-market news appropriately. Uses shared utility
        to reduce code duplication.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            stock_df: DataFrame with stock price data
            sentiment_date_col: Name of date column in sentiment DataFrame
            stock_date_col: Name of date column in stock DataFrame
            
        Returns:
            Tuple of aligned DataFrames (sentiment, stock)
            
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> sent_aligned, stock_aligned = analyzer.align_data(sentiment_df, stock_df)
        """
        # Use shared alignment utility
        merged, _ = align_dates(
            sentiment_df, stock_df,
            date_col1=sentiment_date_col,
            date_col2=stock_date_col,
            how='inner'
        )
        
        return merged, merged
    
    def calculate_correlation(self, sentiment_series: pd.Series,
                            returns_series: pd.Series) -> Dict:
        """
        Calculate Pearson correlation between sentiment and returns.
        
        Args:
            sentiment_series: Series of sentiment scores
            returns_series: Series of daily returns
            
        Returns:
            Dictionary with correlation statistics:
            - pearson_r: Pearson correlation coefficient
            - p_value: P-value for significance test
            - is_significant: Boolean indicating if correlation is significant (p < 0.05)
            - n_observations: Number of observations used
            
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> result = analyzer.calculate_correlation(sentiment_scores, returns)
            >>> print(f"Correlation: {result['pearson_r']:.3f}, p-value: {result['p_value']:.4f}")
        """
        # Align series (remove NaN values)
        aligned = pd.DataFrame({
            'sentiment': sentiment_series,
            'returns': returns_series
        }).dropna()
        
        if len(aligned) < 2:
            return {
                'pearson_r': np.nan,
                'p_value': np.nan,
                'is_significant': False,
                'n_observations': len(aligned)
            }
        
        # Calculate correlation
        r, p_value = stats.pearsonr(aligned['sentiment'], aligned['returns'])
        
        return {
            'pearson_r': r,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'n_observations': len(aligned)
        }
    
    def analyze_lagged_correlations(self, sentiment_df: pd.DataFrame,
                                   stock_df: pd.DataFrame,
                                   max_lag: int = 5) -> pd.DataFrame:
        """
        Analyze correlations with different lag periods.
        
        Tests correlation between sentiment and returns at various lags
        (same-day, next-day, etc.) to identify optimal time windows.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            stock_df: DataFrame with stock price data
            max_lag: Maximum number of days to lag (default: 5)
            
        Returns:
            DataFrame with correlation results for each lag:
            - lag: Number of days lagged
            - pearson_r: Correlation coefficient
            - p_value: P-value
            - is_significant: Whether correlation is significant
            
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> lagged = analyzer.analyze_lagged_correlations(sentiment_df, stock_df)
            >>> best_lag = lagged[lagged['is_significant']].iloc[0]
        """
        # Align data
        aligned_sent, aligned_stock = self.align_data(sentiment_df, stock_df)
        
        # Calculate returns
        returns = self.calculate_daily_returns(aligned_stock)
        
        # Get sentiment scores (assuming mean_sentiment column exists)
        if 'mean_sentiment' in aligned_sent.columns:
            sentiment = aligned_sent['mean_sentiment']
        elif 'sentiment' in aligned_sent.columns:
            sentiment = aligned_sent['sentiment']
        else:
            raise ValueError("Sentiment column not found. Expected 'mean_sentiment' or 'sentiment'")
        
        results = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # Same-day correlation
                sent_aligned = sentiment
                ret_aligned = returns
            else:
                # Lagged correlation: sentiment today, returns N days later
                sent_aligned = sentiment.shift(-lag)
                ret_aligned = returns
            
            # Calculate correlation
            corr_result = self.calculate_correlation(sent_aligned, ret_aligned)
            corr_result['lag'] = lag
            results.append(corr_result)
        
        return pd.DataFrame(results)
    
    def calculate_rolling_correlation(self, sentiment_series: pd.Series,
                                     returns_series: pd.Series,
                                     window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation over a specified window.
        
        Args:
            sentiment_series: Series of sentiment scores
            returns_series: Series of daily returns
            window: Rolling window size in days (default: 30)
            
        Returns:
            Series of rolling correlation coefficients
            
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> rolling_corr = analyzer.calculate_rolling_correlation(sentiment, returns, window=30)
        """
        aligned = pd.DataFrame({
            'sentiment': sentiment_series,
            'returns': returns_series
        }).dropna()
        
        rolling_corr = aligned['sentiment'].rolling(window=window).corr(aligned['returns'])
        
        return rolling_corr

