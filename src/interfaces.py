"""
Interface definitions for consistent class contracts.

This module defines abstract base classes and protocols to ensure
consistent interfaces across all analyzer classes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import pandas as pd


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzer classes.
    
    Ensures consistent interface across DataLoader, TechnicalAnalyzer,
    SentimentAnalyzer, CorrelationAnalyzer, FinancialMetrics, etc.
    """
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the analyzer."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the analyzer instance.
        
        Returns:
            Dictionary with analyzer information
        """
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__
        }
    
    def validate_input(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate input DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If required columns are missing
        """
        from .utils import validate_dataframe
        return validate_dataframe(df, required_columns)


class DataProcessor(BaseAnalyzer):
    """
    Interface for classes that process DataFrames.
    
    Ensures consistent processing methods across data loaders and processors.
    """
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        pass


class MetricsCalculator(BaseAnalyzer):
    """
    Interface for classes that calculate metrics.
    
    Ensures consistent metric calculation methods.
    """
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics on a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with calculated metrics
        """
        pass
    
    def get_metrics_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of calculated metrics.
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            Dictionary with summary statistics
        """
        return {}


class Visualizer(BaseAnalyzer):
    """
    Interface for visualization classes.
    
    Ensures consistent visualization methods.
    """
    
    @abstractmethod
    def plot(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Create a plot from DataFrame.
        
        Args:
            df: DataFrame to plot
            **kwargs: Additional plot parameters
            
        Returns:
            Plot object (matplotlib figure, etc.)
        """
        pass
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the current plot to file.
        
        Args:
            filepath: Path to save plot
            **kwargs: Additional save parameters
        """
        pass

