"""
Integration test demonstrating end-to-end workflow.

Tests the complete pipeline from data loading through sentiment analysis
to correlation calculation.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_analyst_ratings, get_data_info
from src.text_analysis import get_top_keywords, extract_financial_keywords
from src.sentiment_analysis import SentimentAnalyzer
from src.correlation_analysis import CorrelationAnalyzer


class TestEndToEndWorkflow:
    """Integration test for complete analysis pipeline."""
    
    def setup_method(self):
        """Set up test fixtures with sample data."""
        # Create sample news data
        self.sample_news = pd.DataFrame({
            'headline': [
                'Apple stock rises 5% after strong earnings',
                'Microsoft shares fall on weak guidance',
                'Tesla receives FDA approval for new product',
                'Amazon price target raised to $200',
                'Google maintains buy rating'
            ],
            'date': pd.to_datetime([
                '2020-06-05 10:00:00',
                '2020-06-06 11:00:00',
                '2020-06-07 09:00:00',
                '2020-06-08 14:00:00',
                '2020-06-09 15:00:00'
            ]),
            'publisher': ['Publisher A', 'Publisher B', 'Publisher A', 'Publisher C', 'Publisher B'],
            'stock': ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']
        })
        
        # Create sample stock data
        self.sample_stock = pd.DataFrame({
            'date': pd.to_datetime([
                '2020-06-05', '2020-06-06', '2020-06-07', '2020-06-08', '2020-06-09'
            ]),
            'close': [100.0, 102.0, 98.0, 105.0, 103.0],
            'high': [101.0, 103.0, 99.0, 106.0, 104.0],
            'low': [99.0, 101.0, 97.0, 104.0, 102.0],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000]
        })
    
    def test_data_loading_pipeline(self):
        """Test data loading and preprocessing."""
        # Test that we can process the sample data
        df = self.sample_news.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        info = get_data_info(df)
        
        assert info['total_records'] == 5
        assert 'date_range' in info
        assert info['unique_stocks'] == 5
    
    def test_text_analysis_pipeline(self):
        """Test text analysis on sample headlines."""
        # Extract keywords
        keywords = get_top_keywords(self.sample_news, 'headline', top_n=5)
        
        assert len(keywords) > 0
        assert isinstance(keywords, pd.Series)
        
        # Extract financial keywords
        financial_terms = extract_financial_keywords(self.sample_news)
        
        assert isinstance(financial_terms, dict)
        assert 'price target' in financial_terms or 'fda approval' in financial_terms
    
    def test_sentiment_analysis_pipeline(self):
        """Test sentiment analysis on sample headlines."""
        analyzer = SentimentAnalyzer()
        
        # Analyze sentiment
        sentiments = analyzer.analyze_sentiment_batch(self.sample_news['headline'])
        
        assert len(sentiments) == len(self.sample_news)
        assert all(-1 <= s <= 1 for s in sentiments)
        
        # Aggregate daily sentiment
        self.sample_news['sentiment'] = sentiments
        daily = analyzer.aggregate_daily_sentiment(self.sample_news)
        
        assert 'mean_sentiment' in daily.columns
        assert 'article_count' in daily.columns
    
    def test_correlation_analysis_pipeline(self):
        """Test correlation analysis between sentiment and returns."""
        # Calculate sentiment
        sentiment_analyzer = SentimentAnalyzer()
        self.sample_news['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(
            self.sample_news['headline']
        )
        
        # Aggregate daily sentiment
        daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(self.sample_news)
        
        # Calculate returns
        corr_analyzer = CorrelationAnalyzer()
        returns = corr_analyzer.calculate_daily_returns(self.sample_stock)
        
        # Align data
        aligned_sent, aligned_stock = corr_analyzer.align_data(
            daily_sentiment, self.sample_stock
        )
        
        # Calculate correlation
        if len(aligned_sent) > 0 and 'mean_sentiment' in aligned_sent.columns:
            correlation = corr_analyzer.calculate_correlation(
                aligned_sent['mean_sentiment'], returns
            )
            
            assert 'pearson_r' in correlation
            assert 'p_value' in correlation
            assert 'n_observations' in correlation
    
    def test_complete_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Load and preprocess data
        df = self.sample_news.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Step 2: Text analysis
        keywords = get_top_keywords(df, 'headline', top_n=3)
        assert len(keywords) > 0
        
        # Step 3: Sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        df['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(df['headline'])
        daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(df)
        
        # Step 4: Stock data preparation
        stock_df = self.sample_stock.copy()
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        # Step 5: Correlation analysis
        corr_analyzer = CorrelationAnalyzer()
        returns = corr_analyzer.calculate_daily_returns(stock_df)
        
        # Step 6: Align and correlate
        if len(daily_sentiment) > 0:
            aligned_sent, aligned_stock = corr_analyzer.align_data(
                daily_sentiment, stock_df
            )
            
            if 'mean_sentiment' in aligned_sent.columns and len(aligned_sent) > 1:
                correlation = corr_analyzer.calculate_correlation(
                    aligned_sent['mean_sentiment'], returns
                )
                
                # Verify we got valid results
                assert isinstance(correlation, dict)
                assert 'pearson_r' in correlation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

