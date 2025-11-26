# News Sentiment Trading

A quantitative finance project analyzing the correlation between financial news sentiment and stock price movements using natural language processing and technical analysis.

## Overview

This project performs comprehensive analysis of analyst ratings and financial news to:
- Conduct exploratory data analysis on news publications
- Calculate technical indicators using TA-Lib
- Analyze sentiment correlation with stock price movements

## Tasks

### Task 1: Exploratory Data Analysis (EDA)
Comprehensive analysis of financial news data including:
- **Data Loading**: Robust parsing of mixed timezone formats with temporal feature extraction
- **Text Analysis**: NLP-based keyword extraction and financial term identification
- **Publisher Analysis**: Publisher statistics, activity patterns, and domain analysis
- **Temporal Analysis**: Time series patterns, publication frequency, and trend identification

**Deliverables**: `notebooks/task_1_eda.ipynb`

### Task 2: Quantitative Analysis
Technical analysis and financial metrics calculation:
- **Data Preparation**: Stock price data loading with OHLCV normalization
- **Technical Indicators**: TA-Lib integration for SMA, EMA, RSI, MACD, and Bollinger Bands
- **Financial Metrics**: Volatility, Sharpe ratio, maximum drawdown, and risk-adjusted returns
- **Visualization**: Comprehensive charts for price trends, indicators, and returns distribution

**Deliverables**: 
- `scripts/task2_quantitative_analysis.py`
- `notebooks/task_2_quantitative_analysis.ipynb`

### Task 3: Correlation Analysis
Sentiment-driven stock movement correlation:
- **Date Alignment**: Normalization and alignment of news and stock datasets by trading dates
- **Sentiment Analysis**: Multi-method sentiment scoring using TextBlob and VADER
- **Daily Aggregation**: Average sentiment computation for days with multiple articles
- **Correlation Analysis**: Pearson correlation with statistical significance testing and lagged correlation analysis

**Deliverables**:
- `scripts/task3_correlation_analysis.py`
- `notebooks/task_3_correlation_analysis.ipynb`

## Project Structure

```
├── .vscode/              # VS Code settings
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Raw datasets
├── src/                  # Source code modules (OO design)
│   ├── data_loader.py          # DataLoader class
│   ├── text_analysis.py        # TextAnalyzer class
│   ├── publisher_analysis.py   # PublisherAnalyzer class
│   ├── technical_indicators.py # TechnicalAnalyzer class
│   ├── financial_metrics.py   # FinancialMetrics class
│   ├── visualization.py       # StockVisualizer class
│   ├── sentiment_analysis.py   # SentimentAnalyzer class
│   └── correlation_analysis.py # CorrelationAnalyzer class
├── notebooks/            # Jupyter notebooks for analysis
│   ├── task_1_eda.ipynb
│   ├── task_2_quantitative_analysis.ipynb
│   └── task_3_correlation_analysis.ipynb
├── tests/                # Unit tests
└── scripts/              # Utility scripts
    ├── task2_quantitative_analysis.py
    └── task3_correlation_analysis.py
```

## Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** TA-Lib requires the C library. Install it first:
- Ubuntu/Debian: `sudo apt-get install ta-lib`
- macOS: `brew install ta-lib`

## Quick Start

1. **Task 1 - Exploratory Data Analysis:**
```bash
jupyter notebook notebooks/task_1_eda.ipynb
```

2. **Task 2 - Quantitative Analysis:**
```bash
python scripts/task2_quantitative_analysis.py --stock data/AAPL.csv --output output/task2
# Or use the notebook:
jupyter notebook notebooks/task_2_quantitative_analysis.ipynb
```

3. **Task 3 - Correlation Analysis:**
```bash
python scripts/task3_correlation_analysis.py --news data/raw_analyst_ratings.csv --stock data/AAPL.csv --output output/task3
# Or use the notebook:
jupyter notebook notebooks/task_3_correlation_analysis.ipynb
```

4. **Run tests:**
```bash
pytest tests/
pytest tests/test_integration.py -v  # End-to-end integration test
```

## Key Features

- **Object-Oriented Design**: Consistent class-based APIs for all modules
- **Exploratory Data Analysis**: Comprehensive analysis of news publication patterns, publisher activity, and temporal trends
- **Text Analysis**: NLP-based topic modeling and keyword extraction from financial headlines
- **Technical Analysis**: TA-Lib integration for calculating technical indicators (RSI, MACD, moving averages)
- **Sentiment Analysis**: Multi-method sentiment analysis (TextBlob, VADER) with aggregation
- **Correlation Analysis**: Statistical correlation between news sentiment and stock returns

## Current Progress

### ✅ Completed Features
- **Task 1 - EDA**: Data loading, text analysis, publisher analysis, temporal patterns
- **Task 2 - Quantitative Analysis**: Technical indicators (TA-Lib), financial metrics, comprehensive visualizations
- **Task 3 - Correlation Analysis**: Date alignment, sentiment analysis, correlation with lagged analysis
- **Object-Oriented Design**: Class-based APIs for all modules (DataLoader, FinancialMetrics, StockVisualizer)
- **Integration Tests**: End-to-end workflow validation
- **Production Scripts**: Runnable scripts for all three tasks

### 📊 Expected Inputs/Outputs

**Data Loading:**
- Input: CSV file with columns: `headline`, `url`, `publisher`, `date`, `stock`
- Output: DataFrame with temporal features (`year`, `month`, `day_of_week`, `hour`, etc.)

**Sentiment Analysis:**
- Input: Series of headline strings
- Output: Series of sentiment scores (-1 to 1) and daily aggregated sentiment

**Correlation Analysis:**
- Input: Sentiment scores and stock returns (aligned by date)
- Output: Dictionary with `pearson_r`, `p_value`, `is_significant`, `n_observations`

## API Examples

### Data Loading
```python
from src.data_loader import DataLoader

# Initialize DataLoader
loader = DataLoader()

# Load analyst ratings data
df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
# Output: DataFrame with ~1.4M records, temporal features added

# Load stock price data
stock_df = loader.load_stock_price_data('data/AAPL.csv')
# Output: DataFrame with normalized OHLCV columns

# Get data information
info = loader.get_data_info(df)
# Output: {'total_records': 1407328, 'date_range': (...), ...}
```

### Text Analysis
```python
from src.text_analysis import get_top_keywords, extract_financial_keywords

# Extract top keywords
keywords = get_top_keywords(df, 'headline', top_n=20)
# Output: Series with top 20 keywords and their counts

# Extract financial terms
financial_terms = extract_financial_keywords(df)
# Output: Dict with financial keywords and frequencies
```

### Technical Indicators & Financial Metrics
```python
from src.technical_indicators import TechnicalAnalyzer
from src.financial_metrics import FinancialMetrics

# Calculate technical indicators
ta = TechnicalAnalyzer(close_col='close', high_col='high', low_col='low', volume_col='volume')
df_with_indicators = ta.calculate_all_indicators(price_df)
rsi = ta.calculate_rsi(price_df, period=14)
macd = ta.calculate_macd(price_df)

# Calculate financial metrics
metrics = FinancialMetrics(risk_free_rate=0.02)
df_with_metrics = metrics.calculate_all_metrics(df_with_indicators, price_col='close')
volatility = metrics.calculate_volatility(price_df, window=30)
sharpe = metrics.calculate_sharpe_ratio(returns, window=252)
```

### Sentiment Analysis
```python
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
# Analyze sentiment for headlines
df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
# Output: Series of sentiment scores (-1 to 1)

# Aggregate daily sentiment
daily_sentiment = analyzer.aggregate_daily_sentiment(df)
# Output: DataFrame with mean_sentiment, article_count, positive_count, etc.
```

### Visualization
```python
from src.visualization import StockVisualizer

viz = StockVisualizer(figsize=(14, 8))
viz.plot_price_with_indicators(df, indicators=['sma', 'ema', 'bb_upper', 'bb_lower'])
viz.plot_technical_indicators(df)
viz.plot_correlation(sentiment_scores, returns)
viz.save_plot('output/chart.png')
```

### Correlation Analysis
```python
from src.correlation_analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
# Calculate correlation
correlation = analyzer.calculate_correlation(sentiment_scores, returns)
# Output: {'pearson_r': 0.15, 'p_value': 0.02, 'is_significant': True, ...}

# Analyze lagged correlations
lagged = analyzer.analyze_lagged_correlations(sentiment_df, stock_df, max_lag=5)
# Output: DataFrame with correlation results for each lag period
```

## Data

The project uses analyst ratings data containing:
- Financial news headlines
- Publication timestamps
- Publisher information
- Stock ticker symbols

## CI/CD

This project uses GitHub Actions for continuous integration and deployment.

### Workflow

The CI/CD pipeline (`.github/workflows/unittests.yml`) automatically:

- **Triggers**: Runs on push and pull requests to `main` and `develop` branches
- **Environment**: Ubuntu latest with Python 3.12
- **Steps**:
  1. Checkout code
  2. Set up Python 3.12
  3. Install project dependencies
  4. Run unit tests with pytest

### Status

View CI/CD status in the GitHub repository's "Actions" tab. All tests must pass before merging pull requests.

## Development

- **Source code**: `src/` - Reusable analysis modules
- **Notebooks**: `notebooks/` - Interactive analysis and research
- **Tests**: `tests/` - Unit tests for modules
- **Scripts**: `scripts/` - Utility and automation scripts

