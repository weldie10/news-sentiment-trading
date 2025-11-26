"""
Financial metrics calculation using yfinance and pandas.

This module provides a class-based interface for calculating additional
financial metrics beyond technical indicators, including volatility,
returns, and risk-adjusted metrics.

Example:
    >>> metrics = FinancialMetrics()
    >>> df_with_metrics = metrics.calculate_all_metrics(price_df)
    >>> volatility = metrics.calculate_volatility(price_df, window=30)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Install with: pip install yfinance")


class FinancialMetrics:
    """
    A class for calculating financial metrics and performance indicators.
    
    Provides methods for computing volatility, returns, Sharpe ratio,
    and other risk-adjusted performance metrics.
    
    Attributes:
        risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
        
    Example:
        >>> metrics = FinancialMetrics(risk_free_rate=0.02)
        >>> df = metrics.calculate_all_metrics(price_df)
        >>> sharpe = metrics.calculate_sharpe_ratio(returns_series)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the FinancialMetrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_daily_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Calculate daily percentage returns.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column (default: 'close')
            
        Returns:
            Series of daily returns (as percentages)
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> returns = metrics.calculate_daily_returns(df, price_col='close')
        """
        return df[price_col].pct_change() * 100
    
    def calculate_log_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Calculate logarithmic returns.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column (default: 'close')
            
        Returns:
            Series of log returns
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> log_returns = metrics.calculate_log_returns(df)
        """
        return np.log(df[price_col] / df[price_col].shift(1)) * 100
    
    def calculate_volatility(self, df: pd.DataFrame, price_col: str = 'close',
                           window: int = 30, annualized: bool = True) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column (default: 'close')
            window: Rolling window size in days (default: 30)
            annualized: Whether to annualize volatility (default: True)
            
        Returns:
            Series of volatility values
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> volatility = metrics.calculate_volatility(df, window=30)
        """
        returns = self.calculate_daily_returns(df, price_col)
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            # Annualize by multiplying by sqrt(252) trading days
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                               window: Optional[int] = None) -> pd.Series:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Args:
            returns: Series of daily returns
            window: Rolling window size. If None, calculates overall Sharpe ratio
            
        Returns:
            Series of Sharpe ratios (if window specified) or single value
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> returns = metrics.calculate_daily_returns(df)
            >>> sharpe = metrics.calculate_sharpe_ratio(returns, window=252)
        """
        if window is None:
            # Overall Sharpe ratio
            excess_returns = returns.mean() - (self.risk_free_rate / 252)
            std_returns = returns.std()
            if std_returns == 0:
                return pd.Series([np.nan])
            sharpe = (excess_returns / std_returns) * np.sqrt(252)
            return pd.Series([sharpe])
        else:
            # Rolling Sharpe ratio
            excess_returns = returns - (self.risk_free_rate / 252)
            rolling_sharpe = (excess_returns.rolling(window).mean() / 
                            excess_returns.rolling(window).std()) * np.sqrt(252)
            return rolling_sharpe
    
    def calculate_beta(self, stock_returns: pd.Series, 
                      market_returns: pd.Series) -> float:
        """
        Calculate beta (sensitivity to market movements).
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns (e.g., S&P 500)
            
        Returns:
            Beta value
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> beta = metrics.calculate_beta(stock_returns, market_returns)
        """
        # Align series
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 2:
            return np.nan
        
        # Calculate covariance and variance
        covariance = aligned['stock'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        if market_variance == 0:
            return np.nan
        
        beta = covariance / market_variance
        return beta
    
    def calculate_max_drawdown(self, df: pd.DataFrame, price_col: str = 'close') -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column (default: 'close')
            
        Returns:
            Maximum drawdown as a percentage
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> max_dd = metrics.calculate_max_drawdown(df)
        """
        prices = df[price_col]
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        return max_drawdown
    
    def calculate_rolling_max_drawdown(self, df: pd.DataFrame, 
                                      price_col: str = 'close',
                                      window: int = 252) -> pd.Series:
        """
        Calculate rolling maximum drawdown.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the closing price column (default: 'close')
            window: Rolling window size in days (default: 252)
            
        Returns:
            Series of rolling maximum drawdowns
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> rolling_dd = metrics.calculate_rolling_max_drawdown(df, window=252)
        """
        prices = df[price_col]
        cumulative = (1 + prices.pct_change()).cumprod()
        
        rolling_dd = []
        for i in range(len(cumulative)):
            if i < window:
                window_data = cumulative.iloc[:i+1]
            else:
                window_data = cumulative.iloc[i-window+1:i+1]
            
            if len(window_data) > 0:
                running_max = window_data.expanding().max()
                drawdown = (window_data - running_max) / running_max
                rolling_dd.append(drawdown.min() * 100)
            else:
                rolling_dd.append(np.nan)
        
        return pd.Series(rolling_dd, index=df.index)
    
    def calculate_all_metrics(self, df: pd.DataFrame, 
                            price_col: str = 'close') -> pd.DataFrame:
        """
        Calculate all financial metrics and add to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of the closing price column (default: 'close')
            
        Returns:
            DataFrame with added metric columns:
            - daily_return: Daily percentage returns
            - log_return: Logarithmic returns
            - volatility_30d: 30-day rolling volatility (annualized)
            - volatility_252d: 252-day rolling volatility (annualized)
            - sharpe_252d: 252-day rolling Sharpe ratio
            - max_drawdown_252d: 252-day rolling maximum drawdown
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> df_with_metrics = metrics.calculate_all_metrics(price_df)
        """
        df = df.copy()
        
        # Returns
        df['daily_return'] = self.calculate_daily_returns(df, price_col)
        df['log_return'] = self.calculate_log_returns(df, price_col)
        
        # Volatility
        df['volatility_30d'] = self.calculate_volatility(df, price_col, window=30)
        df['volatility_252d'] = self.calculate_volatility(df, price_col, window=252)
        
        # Sharpe ratio
        df['sharpe_252d'] = self.calculate_sharpe_ratio(df['daily_return'], window=252)
        
        # Maximum drawdown
        df['max_drawdown_252d'] = self.calculate_rolling_max_drawdown(df, price_col, window=252)
        
        return df
    
    def download_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
            
        Example:
            >>> metrics = FinancialMetrics()
            >>> df = metrics.download_stock_data('AAPL', '2020-01-01', '2020-12-31')
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Normalize column names to lowercase
        data.columns = [col.lower() if isinstance(col, str) else col[0].lower() 
                       for col in data.columns]
        
        # Rename date column
        if 'date' not in data.columns:
            date_col = [col for col in data.columns if 'date' in str(col).lower()]
            if date_col:
                data = data.rename(columns={date_col[0]: 'date'})
        
        return data

