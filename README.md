# News Sentiment Trading

A quantitative finance project analyzing the correlation between financial news sentiment and stock price movements using natural language processing and technical analysis.

## Overview

This project performs comprehensive analysis of analyst ratings and financial news to:
- Conduct exploratory data analysis on news publications
- Calculate technical indicators using TA-Lib
- Analyze sentiment correlation with stock price movements

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
│   ├── sentiment_analysis.py   # SentimentAnalyzer class
│   └── correlation_analysis.py # CorrelationAnalyzer class
├── notebooks/            # Jupyter notebooks for analysis
│   └── task_1_eda.ipynb
├── tests/                # Unit tests
└── scripts/              # Utility scripts
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

1. **Run end-to-end example:**
```bash
python scripts/end_to_end_example.py
```

2. **Run EDA analysis:**
```bash
jupyter notebook notebooks/task_1_eda.ipynb
```

3. **Run tests:**
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

## API Examples

### Data Loading
```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load_analyst_ratings('data/raw_analyst_ratings.csv')
info = loader.get_data_info(df)
```

### Text Analysis
```python
from src.text_analysis import TextAnalyzer

analyzer = TextAnalyzer()
tokens = analyzer.preprocess_text("Apple stock rises 5%")
keywords = analyzer.get_top_keywords(df, 'headline', top_n=20)
financial_terms = analyzer.extract_financial_keywords(df)
```

### Technical Indicators
```python
from src.technical_indicators import TechnicalAnalyzer

ta = TechnicalAnalyzer()
df_with_indicators = ta.calculate_all_indicators(price_df)
rsi = ta.calculate_rsi(price_df, period=14)
macd = ta.calculate_macd(price_df)
```

### Sentiment Analysis
```python
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
df['sentiment'] = analyzer.analyze_sentiment_batch(df['headline'])
daily_sentiment = analyzer.aggregate_daily_sentiment(df)
```

### Correlation Analysis
```python
from src.correlation_analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
correlation = analyzer.calculate_correlation(sentiment_scores, returns)
lagged = analyzer.analyze_lagged_correlations(sentiment_df, stock_df)
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

