"""
Smoke tests for functional readiness.

Quick tests to verify basic functionality without requiring full datasets.
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


class TestSmokeTests:
    """Smoke tests to verify basic functionality."""
    
    def test_data_loader_smoke(self):
        """Smoke test: DataLoader can be instantiated and used."""
        loader = DataLoader()
        assert loader is not None
        
        # Create minimal test data
        test_df = pd.DataFrame({
            'headline': ['Test'],
            'url': ['http://test.com'],
            'publisher': ['Test'],
            'date': ['2020-01-01 10:00:00'],
            'stock': ['AAPL']
        })
        test_df.to_csv('smoke_test.csv', index=False)
        
        try:
            df = loader.load_analyst_ratings('smoke_test.csv')
            assert len(df) > 0
        finally:
            Path('smoke_test.csv').unlink(missing_ok=True)
    
    def test_technical_indicators_smoke(self):
        """Smoke test: Technical indicators can be calculated."""
        ta = TechnicalAnalyzer()
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        result = ta.calculate_all_indicators(df)
        assert len(result) > 0
        assert 'sma' in result.columns
    
    def test_sentiment_analysis_smoke(self):
        """Smoke test: Sentiment analysis works on sample text."""
        analyzer = SentimentAnalyzer()
        test_text = "Apple stock rises 5% after strong earnings"
        score = analyzer.analyze_sentiment(test_text)
        assert -1 <= score <= 1
    
    def test_correlation_smoke(self):
        """Smoke test: Correlation calculation works."""
        analyzer = CorrelationAnalyzer()
        sentiment = pd.Series([0.5, 0.3, -0.2, 0.1, 0.4])
        returns = pd.Series([1.0, 0.5, -0.8, 0.2, 0.6])
        
        result = analyzer.calculate_correlation(sentiment, returns)
        assert 'pearson_r' in result
        assert isinstance(result['pearson_r'], (float, np.floating))
    
    def test_financial_metrics_smoke(self):
        """Smoke test: Financial metrics can be calculated."""
        metrics = FinancialMetrics()
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        })
        
        result = metrics.calculate_all_metrics(df)
        assert 'daily_return' in result.columns
        assert len(result) > 0
    
    def test_visualization_smoke(self):
        """Smoke test: Visualization can be created."""
        viz = StockVisualizer()
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50),
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        # Should not raise exception
        fig = viz.plot_price_data(df)
        assert fig is not None
        viz.close()
    
    def test_end_to_end_smoke(self):
        """Smoke test: Complete pipeline works with sample data."""
        # Create sample news data
        news_df = pd.DataFrame({
            'headline': [
                'Apple stock rises 5%',
                'Microsoft shares fall',
                'Tesla receives approval'
            ],
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'stock': ['AAPL', 'MSFT', 'TSLA']
        })
        
        # Create sample stock data
        stock_df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'close': [100.0, 102.0, 98.0],
            'high': [101.0, 103.0, 99.0],
            'low': [99.0, 101.0, 97.0],
            'open': [100.0, 101.0, 98.0],
            'volume': [1000000, 1100000, 900000]
        })
        
        # Test sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        news_df['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(news_df['headline'])
        assert len(news_df) > 0
        
        # Test correlation
        correlation_analyzer = CorrelationAnalyzer()
        returns = correlation_analyzer.calculate_daily_returns(stock_df)
        assert len(returns) > 0
        
        # Test alignment
        aligned_sent, aligned_stock = correlation_analyzer.align_data(news_df, stock_df)
        assert len(aligned_sent) > 0

