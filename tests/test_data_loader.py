"""
Unit tests for DataLoader class.

Tests data loading, date parsing, and temporal feature extraction.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(date_column='custom_date')
        assert loader.date_column == 'custom_date'
        assert loader.tz_pattern == r'[-+]\d{2}:\d{2}$'
    
    def test_normalize_datetime_timezone_aware(self):
        """Test datetime normalization with timezone-aware dates."""
        from datetime import datetime
        import pytz
        
        # Create timezone-aware datetime
        tz_aware = datetime(2020, 6, 5, 10, 30, 54, tzinfo=pytz.timezone('US/Eastern'))
        normalized = self.loader._normalize_datetime(tz_aware)
        
        assert normalized.tz is None  # Should be timezone-naive
        assert normalized.year == 2020
    
    def test_normalize_datetime_naive(self):
        """Test datetime normalization with naive dates."""
        from datetime import datetime
        
        naive = datetime(2020, 6, 5, 10, 30, 54)
        normalized = self.loader._normalize_datetime(naive)
        
        assert normalized == naive
        assert normalized.tz is None
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-06-05 10:30:00', '2020-06-06 14:45:00'])
        })
        
        result = self.loader._extract_temporal_features(df)
        
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'hour' in result.columns
        assert result['year'].iloc[0] == 2020
        assert result['hour'].iloc[0] == 10
    
    def test_get_data_info(self):
        """Test data info extraction."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-06-05', '2020-06-06']),
            'publisher': ['Publisher A', 'Publisher B'],
            'stock': ['AAPL', 'MSFT']
        })
        
        info = self.loader.get_data_info(df)
        
        assert info['total_records'] == 2
        assert info['unique_publishers'] == 2
        assert info['unique_stocks'] == 2
        assert 'date_range' in info
        assert 'columns' in info
        assert 'missing_values' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

