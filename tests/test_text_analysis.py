"""
Unit tests for TextAnalyzer class.

Tests text preprocessing, n-gram extraction, and keyword analysis.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_analysis import TextAnalyzer


class TestTextAnalyzer:
    """Test suite for TextAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
    
    def test_initialization(self):
        """Test TextAnalyzer initialization."""
        analyzer = TextAnalyzer(min_token_length=3)
        assert analyzer.min_token_length == 3
        assert len(analyzer.stop_words) > 0
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Apple's stock rises 5%!"
        tokens = self.analyzer.preprocess_text(text)
        
        assert isinstance(tokens, list)
        assert 'apple' in tokens
        assert 'stock' in tokens
        assert 'rises' in tokens
        assert '%' not in tokens  # Special characters removed
    
    def test_preprocess_text_with_stopwords(self):
        """Test text preprocessing with stopword removal."""
        text = "The stock is rising"
        tokens_with_stopwords = self.analyzer.preprocess_text(text, remove_stopwords=False)
        tokens_without_stopwords = self.analyzer.preprocess_text(text, remove_stopwords=True)
        
        assert len(tokens_without_stopwords) < len(tokens_with_stopwords)
        assert 'the' not in tokens_without_stopwords
        assert 'is' not in tokens_without_stopwords
    
    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        text = "price target raised"
        bigrams = self.analyzer.extract_ngrams(text, n=2)
        
        assert isinstance(bigrams, list)
        assert len(bigrams) > 0
        assert any('price target' in bg for bg in bigrams)
    
    def test_get_top_keywords(self):
        """Test top keyword extraction."""
        df = pd.DataFrame({
            'headline': [
                'Apple stock rises',
                'Apple stock falls',
                'Microsoft stock rises'
            ]
        })
        
        top_keywords = self.analyzer.get_top_keywords(df, 'headline', top_n=5)
        
        assert isinstance(top_keywords, pd.Series)
        assert len(top_keywords) <= 5
        assert 'stock' in top_keywords.index
        assert 'rises' in top_keywords.index
    
    def test_extract_financial_keywords(self):
        """Test financial keyword extraction."""
        df = pd.DataFrame({
            'headline': [
                'Price target raised to $100',
                'FDA approval granted',
                'Earnings beat expectations'
            ]
        })
        
        financial_keywords = self.analyzer.extract_financial_keywords(df)
        
        assert isinstance(financial_keywords, dict)
        assert 'price target' in financial_keywords
        assert 'fda approval' in financial_keywords
        assert 'earnings' in financial_keywords


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

