"""
Unit tests for SentimentAnalyzer class.

Tests sentiment analysis methods and aggregation.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sentiment_analysis import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_initialization(self):
        """Test SentimentAnalyzer initialization."""
        analyzer = SentimentAnalyzer()
        assert analyzer.text_analyzer is not None
    
    def test_analyze_sentiment_textblob(self):
        """Test TextBlob sentiment analysis."""
        positive_text = "Great earnings report! Stock rises 10%"
        negative_text = "Poor earnings report. Stock falls 10%"
        
        positive_score = self.analyzer.analyze_sentiment_textblob(positive_text)
        negative_score = self.analyzer.analyze_sentiment_textblob(negative_text)
        
        assert -1 <= positive_score <= 1
        assert -1 <= negative_score <= 1
        assert positive_score > negative_score
    
    def test_classify_sentiment(self):
        """Test sentiment classification."""
        assert self.analyzer.classify_sentiment(0.5) == 'positive'
        assert self.analyzer.classify_sentiment(-0.5) == 'negative'
        assert self.analyzer.classify_sentiment(0.05) == 'neutral'
    
    def test_analyze_sentiment_batch(self):
        """Test batch sentiment analysis."""
        texts = pd.Series([
            "Stock rises 5%",
            "Stock falls 3%",
            "Stock unchanged"
        ])
        
        scores = self.analyzer.analyze_sentiment_batch(texts)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(texts)
        assert all(-1 <= score <= 1 for score in scores)
    
    def test_aggregate_daily_sentiment(self):
        """Test daily sentiment aggregation."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-06-05', '2020-06-05', '2020-06-06']),
            'sentiment': [0.5, -0.3, 0.2]
        })
        
        daily = self.analyzer.aggregate_daily_sentiment(df)
        
        assert 'mean_sentiment' in daily.columns
        assert 'article_count' in daily.columns
        assert 'positive_count' in daily.columns
        assert len(daily) == 2  # Two unique dates


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

