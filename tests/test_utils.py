"""
Unit tests for utility functions.

Tests reusable utility functions for data validation and formatting.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    validate_dataframe,
    ensure_datetime,
    safe_divide,
    create_output_dir,
    format_number,
    calculate_percentage_change
)


class TestUtils:
    """Test suite for utility functions."""
    
    def test_validate_dataframe_success(self):
        """Test successful dataframe validation."""
        df = pd.DataFrame({'date': [1, 2], 'headline': ['a', 'b']})
        result = validate_dataframe(df, ['date', 'headline'])
        assert result is True
    
    def test_validate_dataframe_failure(self):
        """Test dataframe validation with missing columns."""
        df = pd.DataFrame({'date': [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, ['date', 'headline'])
    
    def test_ensure_datetime(self):
        """Test datetime conversion."""
        df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02']})
        result = ensure_datetime(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_safe_divide_scalar(self):
        """Test safe division with scalars."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=99) == 99.0
    
    def test_safe_divide_series(self):
        """Test safe division with Series."""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 5])
        result = safe_divide(numerator, denominator)
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 6.0
    
    def test_create_output_dir(self, tmp_path):
        """Test directory creation."""
        output_path = tmp_path / "test_output"
        result = create_output_dir(output_path)
        assert result.exists()
        assert result.is_dir()
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1234.5678, decimals=2) == "1,234.57"
        assert format_number(1000, decimals=0) == "1,000"
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        assert calculate_percentage_change(100, 110) == 10.0
        assert calculate_percentage_change(100, 90) == -10.0
        assert calculate_percentage_change(0, 100) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

