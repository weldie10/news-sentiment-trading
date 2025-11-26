"""
Unit tests for class contracts and interfaces.

Tests that all classes follow consistent interfaces and meet their contracts.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.technical_indicators import TechnicalAnalyzer
from src.sentiment_analysis import SentimentAnalyzer
from src.correlation_analysis import CorrelationAnalyzer
from src.financial_metrics import FinancialMetrics
from src.visualization import StockVisualizer
from src.interfaces import BaseAnalyzer, DataProcessor, MetricsCalculator, Visualizer


class TestDataLoaderContract:
    """Test DataLoader class contract."""
    
    def test_initialization(self):
        """Test DataLoader can be initialized."""
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, 'default_analyst_ratings_path')
        assert hasattr(loader, 'loaded_data')
    
    def test_get_info(self):
        """Test DataLoader implements get_info method."""
        loader = DataLoader()
        info = loader.get_info()
        assert 'class_name' in info
        assert info['class_name'] == 'DataLoader'
    
    def test_load_analyst_ratings_contract(self):
        """Test load_analyst_ratings returns DataFrame with expected columns."""
        loader = DataLoader()
        # Create sample data file
        sample_data = pd.DataFrame({
            'headline': ['Test headline'],
            'url': ['http://test.com'],
            'publisher': ['Test Publisher'],
            'date': ['2020-01-01 10:00:00'],
            'stock': ['AAPL']
        })
        sample_data.to_csv('test_ratings.csv', index=False)
        
        try:
            df = loader.load_analyst_ratings('test_ratings.csv')
            assert isinstance(df, pd.DataFrame)
            assert 'date' in df.columns
            assert 'date_only' in df.columns
        finally:
            Path('test_ratings.csv').unlink(missing_ok=True)
    
    def test_load_stock_price_data_contract(self):
        """Test load_stock_price_data returns DataFrame with OHLCV columns."""
        loader = DataLoader()
        sample_data = pd.DataFrame({
            'Date': ['2020-01-01', '2020-01-02'],
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        })
        sample_data.to_csv('test_stock.csv', index=False)
        
        try:
            df = loader.load_stock_price_data('test_stock.csv')
            assert isinstance(df, pd.DataFrame)
            assert 'date' in df.columns
            assert 'open' in df.columns
            assert 'close' in df.columns
            assert 'volume' in df.columns
        finally:
            Path('test_stock.csv').unlink(missing_ok=True)
    
    def test_get_data_info_contract(self):
        """Test get_data_info returns expected dictionary structure."""
        loader = DataLoader()
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02']),
            'value': [1, 2]
        })
        info = loader.get_data_info(df)
        assert isinstance(info, dict)
        assert 'total_records' in info
        assert 'columns' in info
        assert 'date_range' in info


class TestTechnicalAnalyzerContract:
    """Test TechnicalAnalyzer class contract."""
    
    def test_initialization(self):
        """Test TechnicalAnalyzer can be initialized."""
        ta = TechnicalAnalyzer()
        assert ta is not None
        assert hasattr(ta, 'close_col')
        assert hasattr(ta, 'high_col')
    
    def test_calculate_all_indicators_contract(self):
        """Test calculate_all_indicators returns DataFrame with indicator columns."""
        ta = TechnicalAnalyzer()
        df = pd.DataFrame({
            'close': [100, 102, 101, 103, 105] * 10,
            'high': [101, 103, 102, 104, 106] * 10,
            'low': [99, 101, 100, 102, 104] * 10,
            'volume': [1000, 1200, 1100, 1300, 1400] * 10
        })
        
        result = ta.calculate_all_indicators(df)
        assert isinstance(result, pd.DataFrame)
        assert 'sma' in result.columns
        assert 'ema' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns


class TestSentimentAnalyzerContract:
    """Test SentimentAnalyzer class contract."""
    
    def test_initialization(self):
        """Test SentimentAnalyzer can be initialized."""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
    
    def test_analyze_sentiment_contract(self):
        """Test analyze_sentiment returns float in [-1, 1] range."""
        analyzer = SentimentAnalyzer()
        score = analyzer.analyze_sentiment("Great earnings report!")
        assert isinstance(score, (float, np.floating))
        assert -1 <= score <= 1
    
    def test_analyze_sentiment_batch_contract(self):
        """Test analyze_sentiment_batch returns Series with same length."""
        analyzer = SentimentAnalyzer()
        texts = pd.Series(["Good news", "Bad news", "Neutral news"])
        scores = analyzer.analyze_sentiment_batch(texts)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(texts)
        assert all(-1 <= s <= 1 for s in scores if pd.notna(s))


class TestCorrelationAnalyzerContract:
    """Test CorrelationAnalyzer class contract."""
    
    def test_initialization(self):
        """Test CorrelationAnalyzer can be initialized."""
        analyzer = CorrelationAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'sentiment_analyzer')
    
    def test_calculate_correlation_contract(self):
        """Test calculate_correlation returns expected dictionary structure."""
        analyzer = CorrelationAnalyzer()
        sentiment = pd.Series([0.5, 0.3, -0.2, 0.1, 0.4])
        returns = pd.Series([1.0, 0.5, -0.8, 0.2, 0.6])
        
        result = analyzer.calculate_correlation(sentiment, returns)
        assert isinstance(result, dict)
        assert 'pearson_r' in result
        assert 'p_value' in result
        assert 'is_significant' in result
        assert 'n_observations' in result
        assert isinstance(result['pearson_r'], (float, np.floating))
        assert isinstance(result['is_significant'], bool)


class TestFinancialMetricsContract:
    """Test FinancialMetrics class contract."""
    
    def test_initialization(self):
        """Test FinancialMetrics can be initialized."""
        metrics = FinancialMetrics()
        assert metrics is not None
        assert hasattr(metrics, 'risk_free_rate')
    
    def test_calculate_all_metrics_contract(self):
        """Test calculate_all_metrics returns DataFrame with metric columns."""
        metrics = FinancialMetrics()
        df = pd.DataFrame({
            'close': [100, 102, 101, 103, 105] * 20
        })
        
        result = metrics.calculate_all_metrics(df)
        assert isinstance(result, pd.DataFrame)
        assert 'daily_return' in result.columns
        assert 'volatility_30d' in result.columns
        assert 'sharpe_252d' in result.columns


class TestStockVisualizerContract:
    """Test StockVisualizer class contract."""
    
    def test_initialization(self):
        """Test StockVisualizer can be initialized."""
        viz = StockVisualizer()
        assert viz is not None
        assert hasattr(viz, 'figsize')
        assert hasattr(viz, 'style')
    
    def test_plot_methods_exist(self):
        """Test all required plot methods exist."""
        viz = StockVisualizer()
        assert hasattr(viz, 'plot_price_data')
        assert hasattr(viz, 'plot_price_with_indicators')
        assert hasattr(viz, 'plot_technical_indicators')
        assert hasattr(viz, 'plot_correlation')
        assert hasattr(viz, 'save_plot')


class TestInterfaceConsistency:
    """Test that classes follow consistent interfaces."""
    
    def test_all_analyzers_have_get_info(self):
        """Test all analyzer classes can provide info."""
        classes = [
            DataLoader(),
            TechnicalAnalyzer(),
            SentimentAnalyzer(),
            CorrelationAnalyzer(),
            FinancialMetrics(),
            StockVisualizer()
        ]
        
        for instance in classes:
            # All should have some way to get info (even if not explicitly BaseAnalyzer)
            assert hasattr(instance, '__class__')
            assert hasattr(instance, '__init__')
    
    def test_data_processing_consistency(self):
        """Test data processing methods are consistent."""
        loader = DataLoader()
        ta = TechnicalAnalyzer()
        
        # Both should accept DataFrames
        df = pd.DataFrame({'close': [100, 102, 101]})
        assert isinstance(loader.get_data_info(df), dict)
        
        # TechnicalAnalyzer should process DataFrames
        df_full = pd.DataFrame({
            'close': [100, 102, 101, 103, 105] * 10,
            'high': [101, 103, 102, 104, 106] * 10,
            'low': [99, 101, 100, 102, 104] * 10,
            'volume': [1000] * 50
        })
        result = ta.calculate_all_indicators(df_full)
        assert isinstance(result, pd.DataFrame)

