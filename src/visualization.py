"""
Visualization utilities for financial data and technical indicators.

This module provides a class-based interface for creating visualizations
of stock prices, technical indicators, and financial metrics.

Example:
    >>> viz = StockVisualizer()
    >>> viz.plot_price_with_indicators(df)
    >>> viz.plot_technical_indicators(df)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StockVisualizer:
    """
    A class for visualizing stock data and technical indicators.
    
    Provides methods for creating various financial charts including
    price charts, technical indicator plots, and correlation visualizations.
    
    Attributes:
        figsize: Default figure size tuple (default: (12, 6))
        style: Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        
    Example:
        >>> viz = StockVisualizer(figsize=(14, 8))
        >>> viz.plot_price_with_indicators(df)
        >>> viz.save_plot('output/chart.png')
    """
    
    def __init__(self, figsize: tuple = (12, 6), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the StockVisualizer.
        
        Args:
            figsize: Default figure size tuple (default: (12, 6))
            style: Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        self.current_figure = None
        self.current_axes = None
    
    def plot_price_data(self, df: pd.DataFrame, price_col: str = 'close',
                       date_col: str = 'date', title: Optional[str] = None,
                       show_volume: bool = True) -> plt.Figure:
        """
        Plot stock price data with optional volume.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the price column (default: 'close')
            date_col: Name of the date column (default: 'date')
            title: Plot title (default: 'Stock Price')
            show_volume: Whether to show volume subplot (default: True)
            
        Returns:
            Matplotlib Figure object
            
        Example:
            >>> viz = StockVisualizer()
            >>> fig = viz.plot_price_data(df, title='AAPL Stock Price')
        """
        if title is None:
            title = f'Stock Price - {price_col.upper()}'
        
        if show_volume and 'volume' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
            ax2 = None
        
        # Plot price
        ax1.plot(df[date_col], df[price_col], linewidth=1.5, label=price_col.upper())
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot volume if available
        if ax2 is not None and 'volume' in df.columns:
            ax2.bar(df[date_col], df['volume'], alpha=0.6, color='steelblue')
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        self.current_figure = fig
        self.current_axes = (ax1, ax2) if ax2 else (ax1,)
        return fig
    
    def plot_price_with_indicators(self, df: pd.DataFrame, 
                                   indicators: Optional[List[str]] = None,
                                   price_col: str = 'close',
                                   date_col: str = 'date',
                                   title: Optional[str] = None) -> plt.Figure:
        """
        Plot stock price with moving averages and other indicators.
        
        Args:
            df: DataFrame with price and indicator data
            indicators: List of indicator columns to plot (default: ['sma', 'ema'])
            price_col: Name of the price column (default: 'close')
            date_col: Name of the date column (default: 'date')
            title: Plot title
            
        Returns:
            Matplotlib Figure object
            
        Example:
            >>> viz = StockVisualizer()
            >>> fig = viz.plot_price_with_indicators(df, indicators=['sma', 'ema', 'bb_upper', 'bb_lower'])
        """
        if indicators is None:
            indicators = ['sma', 'ema']
        
        if title is None:
            title = 'Stock Price with Technical Indicators'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot price
        ax.plot(df[date_col], df[price_col], linewidth=2, label='Close Price', color='black')
        
        # Plot indicators
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, indicator in enumerate(indicators):
            if indicator in df.columns:
                color = colors[i % len(colors)]
                ax.plot(df[date_col], df[indicator], linewidth=1.5, 
                       label=indicator.upper(), alpha=0.7, color=color)
        
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_figure = fig
        self.current_axes = (ax,)
        return fig
    
    def plot_technical_indicators(self, df: pd.DataFrame, 
                                 date_col: str = 'date',
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Create subplots for multiple technical indicators.
        
        Args:
            df: DataFrame with indicator data
            date_col: Name of the date column (default: 'date')
            title: Overall plot title
            
        Returns:
            Matplotlib Figure object
            
        Example:
            >>> viz = StockVisualizer()
            >>> fig = viz.plot_technical_indicators(df)
        """
        if title is None:
            title = 'Technical Indicators Analysis'
        
        # Determine which indicators to plot
        indicators_to_plot = []
        if 'rsi' in df.columns:
            indicators_to_plot.append(('rsi', 'RSI', (0, 100), [30, 70]))
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            indicators_to_plot.append(('macd', 'MACD', None, None))
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            indicators_to_plot.append(('bollinger', 'Bollinger Bands', None, None))
        
        n_plots = len(indicators_to_plot)
        if n_plots == 0:
            raise ValueError("No technical indicators found in DataFrame")
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], 4 * n_plots),
                                sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        for idx, (ind_type, label, ylim, ref_lines) in enumerate(indicators_to_plot):
            ax = axes[idx]
            
            if ind_type == 'rsi':
                ax.plot(df[date_col], df['rsi'], linewidth=1.5, label='RSI', color='purple')
                if ref_lines:
                    ax.axhline(y=ref_lines[0], color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
                    ax.axhline(y=ref_lines[1], color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
                if ylim:
                    ax.set_ylim(ylim)
                ax.set_ylabel('RSI', fontsize=12)
                ax.legend(loc='best')
                
            elif ind_type == 'macd':
                ax.plot(df[date_col], df['macd'], linewidth=1.5, label='MACD', color='blue')
                ax.plot(df[date_col], df['macd_signal'], linewidth=1.5, label='Signal', color='red')
                if 'macd_histogram' in df.columns:
                    ax.bar(df[date_col], df['macd_histogram'], alpha=0.3, label='Histogram', color='gray')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_ylabel('MACD', fontsize=12)
                ax.legend(loc='best')
                
            elif ind_type == 'bollinger':
                ax.plot(df[date_col], df['close'], linewidth=1.5, label='Close', color='black')
                ax.plot(df[date_col], df['bb_upper'], linewidth=1, label='Upper Band', color='red', alpha=0.7)
                ax.plot(df[date_col], df['bb_lower'], linewidth=1, label='Lower Band', color='green', alpha=0.7)
                if 'bb_middle' in df.columns:
                    ax.plot(df[date_col], df['bb_middle'], linewidth=1, label='Middle (SMA)', color='blue', alpha=0.7)
                ax.fill_between(df[date_col], df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
                ax.set_ylabel('Price ($)', fontsize=12)
                ax.legend(loc='best')
            
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        self.current_figure = fig
        self.current_axes = axes
        return fig
    
    def plot_correlation(self, x: pd.Series, y: pd.Series,
                        title: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None) -> plt.Figure:
        """
        Create a scatter plot with correlation line.
        
        Args:
            x: X-axis data series
            y: Y-axis data series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Matplotlib Figure object
            
        Example:
            >>> viz = StockVisualizer()
            >>> fig = viz.plot_correlation(sentiment, returns, 
            ...                           title='Sentiment vs Returns',
            ...                           xlabel='Sentiment Score',
            ...                           ylabel='Daily Returns (%)')
        """
        if title is None:
            title = 'Correlation Analysis'
        
        # Align data
        aligned = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot
        ax.scatter(aligned['x'], aligned['y'], alpha=0.6, s=50)
        
        # Correlation line
        z = np.polyfit(aligned['x'], aligned['y'], 1)
        p = np.poly1d(z)
        ax.plot(aligned['x'], p(aligned['x']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        # Calculate correlation
        corr = aligned['x'].corr(aligned['y'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(xlabel or 'X', fontsize=12)
        ax.set_ylabel(ylabel or 'Y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_figure = fig
        self.current_axes = (ax,)
        return fig
    
    def plot_returns_distribution(self, returns: pd.Series,
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of returns with statistics.
        
        Args:
            returns: Series of returns
            title: Plot title
            
        Returns:
            Matplotlib Figure object
            
        Example:
            >>> viz = StockVisualizer()
            >>> fig = viz.plot_returns_distribution(df['daily_return'])
        """
        if title is None:
            title = 'Returns Distribution'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Histogram
        ax.hist(returns.dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add statistics
        mean_return = returns.mean()
        std_return = returns.std()
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
        ax.axvline(mean_return + std_return, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'+1 Std: {mean_return + std_return:.2f}%')
        ax.axvline(mean_return - std_return, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'-1 Std: {mean_return - std_return:.2f}%')
        
        ax.set_xlabel('Returns (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.current_figure = fig
        self.current_axes = (ax,)
        return fig
    
    def save_plot(self, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """
        Save the current figure to file.
        
        Args:
            filepath: Path to save the figure
            dpi: Resolution in dots per inch (default: 300)
            bbox_inches: Bounding box setting (default: 'tight')
            
        Example:
            >>> viz = StockVisualizer()
            >>> viz.plot_price_data(df)
            >>> viz.save_plot('output/price_chart.png')
        """
        if self.current_figure is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.current_figure.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to {filepath}")
    
    def close(self):
        """Close the current figure."""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            self.current_axes = None

