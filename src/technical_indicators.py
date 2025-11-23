"""
Technical analysis indicators using TA-Lib.

This module provides a class-based interface for calculating technical
indicators on stock price data, including moving averages, RSI, MACD,
and Bollinger Bands.

Example:
    >>> ta = TechnicalAnalyzer()
    >>> df_with_indicators = ta.calculate_all_indicators(price_df)
    >>> rsi = ta.calculate_rsi(price_df, period=14)
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Install with: pip install TA-Lib")


class TechnicalAnalyzer:
    """
    A class for calculating technical analysis indicators.
    
    Provides methods for computing common technical indicators including
    moving averages, momentum indicators, and volatility measures.
    
    Attributes:
        close_col (str): Name of the close price column
        high_col (str): Name of the high price column
        low_col (str): Name of the low price column
        volume_col (str): Name of the volume column
        
    Example:
        >>> ta = TechnicalAnalyzer()
        >>> df = pd.DataFrame({
        ...     'close': [100, 102, 101, 103, 105],
        ...     'high': [101, 103, 102, 104, 106],
        ...     'low': [99, 101, 100, 102, 104],
        ...     'volume': [1000, 1200, 1100, 1300, 1400]
        ... })
        >>> indicators = ta.calculate_all_indicators(df)
    """
    
    def __init__(self, close_col: str = 'close', high_col: str = 'high',
                 low_col: str = 'low', volume_col: str = 'volume'):
        """
        Initialize the TechnicalAnalyzer.
        
        Args:
            close_col: Name of the closing price column
            high_col: Name of the high price column
            low_col: Name of the low price column
            volume_col: Name of the volume column
        """
        self.close_col = close_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required. Install with: pip install TA-Lib\n"
                "Note: You may need to install the C library first:\n"
                "  Ubuntu/Debian: sudo apt-get install ta-lib\n"
                "  macOS: brew install ta-lib"
            )
    
    def calculate_sma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for moving average (default: 20)
            
        Returns:
            Series with SMA values
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> sma = ta.calculate_sma(df, period=20)
        """
        return talib.SMA(df[self.close_col].values, timeperiod=period)
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            df: DataFrame with price data
            period: Number of periods for moving average (default: 20)
            
        Returns:
            Series with EMA values
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> ema = ta.calculate_ema(df, period=20)
        """
        return talib.EMA(df[self.close_col].values, timeperiod=period)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI ranges from 0 to 100:
        - Above 70: Overbought
        - Below 30: Oversold
        
        Args:
            df: DataFrame with price data
            period: Number of periods for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> rsi = ta.calculate_rsi(df, period=14)
            >>> overbought = df[rsi > 70]
        """
        return talib.RSI(df[self.close_col].values, timeperiod=period)
    
    def calculate_macd(self, df: pd.DataFrame, fastperiod: int = 12,
                      slowperiod: int = 26, signalperiod: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with price data
            fastperiod: Fast EMA period (default: 12)
            slowperiod: Slow EMA period (default: 26)
            signalperiod: Signal line period (default: 9)
            
        Returns:
            Dictionary with keys:
            - macd: MACD line
            - signal: Signal line
            - histogram: MACD histogram
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> macd_data = ta.calculate_macd(df)
            >>> df['macd'] = macd_data['macd']
            >>> df['macd_signal'] = macd_data['signal']
        """
        macd, signal, histogram = talib.MACD(
            df[self.close_col].values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return {
            'macd': pd.Series(macd, index=df.index),
            'signal': pd.Series(signal, index=df.index),
            'histogram': pd.Series(histogram, index=df.index)
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20,
                                nbdevup: float = 2, nbdevdn: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            period: Number of periods for moving average (default: 20)
            nbdevup: Number of standard deviations for upper band (default: 2)
            nbdevdn: Number of standard deviations for lower band (default: 2)
            
        Returns:
            Dictionary with keys:
            - upper: Upper Bollinger Band
            - middle: Middle band (SMA)
            - lower: Lower Bollinger Band
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> bb = ta.calculate_bollinger_bands(df)
            >>> df['bb_upper'] = bb['upper']
            >>> df['bb_lower'] = bb['lower']
        """
        upper, middle, lower = talib.BBANDS(
            df[self.close_col].values,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
        return {
            'upper': pd.Series(upper, index=df.index),
            'middle': pd.Series(middle, index=df.index),
            'lower': pd.Series(lower, index=df.index)
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame,
                                sma_period: int = 20, ema_period: int = 20,
                                rsi_period: int = 14) -> pd.DataFrame:
        """
        Calculate all common technical indicators and add to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            sma_period: Period for SMA (default: 20)
            ema_period: Period for EMA (default: 20)
            rsi_period: Period for RSI (default: 14)
            
        Returns:
            DataFrame with added indicator columns:
            - sma: Simple Moving Average
            - ema: Exponential Moving Average
            - rsi: Relative Strength Index
            - macd: MACD line
            - macd_signal: MACD signal line
            - macd_histogram: MACD histogram
            - bb_upper: Upper Bollinger Band
            - bb_middle: Middle Bollinger Band
            - bb_lower: Lower Bollinger Band
            
        Example:
            >>> ta = TechnicalAnalyzer()
            >>> df_with_indicators = ta.calculate_all_indicators(price_df)
        """
        df = df.copy()
        
        # Moving averages
        df['sma'] = self.calculate_sma(df, period=sma_period)
        df['ema'] = self.calculate_ema(df, period=ema_period)
        
        # RSI
        df['rsi'] = self.calculate_rsi(df, period=rsi_period)
        
        # MACD
        macd_data = self.calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        return df

