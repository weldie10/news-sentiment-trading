# Workflow Documentation

## End-to-End Analysis Pipeline

This document describes the complete workflow for analyzing news sentiment and its correlation with stock prices.

## Pipeline Steps

### 1. Data Loading
```python
from src.data_loader import load_analyst_ratings

df = load_analyst_ratings('data/raw_analyst_ratings.csv')
```

### 2. Text Analysis
```python
from src.text_analysis import get_top_keywords, extract_financial_keywords

keywords = get_top_keywords(df, 'headline', top_n=20)
financial_terms = extract_financial_keywords(df)
```

### 3. Sentiment Analysis
```python
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
daily_sentiment = analyzer.aggregate_daily_sentiment(df)
```

### 4. Stock Data & Technical Indicators
```python
from src.technical_indicators import TechnicalAnalyzer

ta = TechnicalAnalyzer()
df_with_indicators = ta.calculate_all_indicators(stock_df)
```

### 5. Correlation Analysis
```python
from src.correlation_analysis import CorrelationAnalyzer

corr_analyzer = CorrelationAnalyzer()
correlation = corr_analyzer.calculate_correlation(sentiment_scores, returns)
```

## Running the Complete Pipeline

### Quick Start
```bash
python scripts/end_to_end_example.py
```

### Step-by-Step
1. **Load news data**: `python -c "from src.data_loader import load_analyst_ratings; df = load_analyst_ratings('data/raw_analyst_ratings.csv')"`
2. **Download stock data**: `python scripts/ingest_stock_data.py --tickers AAPL --start 2020-01-01 --end 2020-12-31`
3. **Calculate indicators**: `python scripts/calculate_indicators.py --input data/stock_data.csv --output data/stock_with_indicators.csv`
4. **Analyze correlation**: `python scripts/analyze_correlation.py --sentiment data/daily_sentiment.csv --stock data/stock_data.csv`

## Testing

Run integration tests:
```bash
pytest tests/test_integration.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

