# Pull Request: Task 3 - Correlation Analysis between News and Stock Movement

## Description
This PR implements Task 3: Correlation Analysis between news sentiment and stock price movements. The implementation includes comprehensive date alignment, sentiment analysis, daily returns calculation, and statistical correlation analysis with lagged correlation support.

## Type of Change
- [x] New feature
- [x] Documentation update

## Related Task
- [x] Task 3: Correlation Analysis

## Changes Made

### Core Functionality
- **Date Alignment**:
  - Normalization of timestamps between news and stock datasets
  - Trading day matching with proper date handling
  - Support for after-hours and pre-market news alignment

- **Sentiment Analysis**:
  - Multi-method sentiment scoring (TextBlob + VADER combined)
  - Sentiment classification (positive, negative, neutral)
  - Daily sentiment aggregation for multiple articles per day
  - Mean and median sentiment calculation

- **Stock Returns Calculation**:
  - Daily percentage returns from closing prices
  - Proper handling of missing data and edge cases
  - Returns statistics and distribution analysis

- **Correlation Analysis**:
  - Pearson correlation coefficient calculation
  - Statistical significance testing (p-values)
  - Lagged correlation analysis (0-5 days)
  - Rolling correlation support

### Scripts and Notebooks
- **task3_correlation_analysis.py**: Complete end-to-end correlation analysis script
- **task_3_correlation_analysis.ipynb**: Comprehensive Jupyter notebook with step-by-step analysis

### Integration
- Uses existing `SentimentAnalyzer` class (already OOP)
- Uses existing `CorrelationAnalyzer` class (already OOP)
- Integrates with `DataLoader` for data loading
- Uses `StockVisualizer` for correlation visualizations

### Visualizations
- Correlation scatter plots with trend lines
- Time series plots for sentiment and returns
- Statistical correlation metrics display

## Testing
- [x] Date alignment tested with various date formats
- [x] Sentiment analysis validated with sample headlines
- [x] Correlation calculations verified with statistical tests
- [x] Lagged correlation analysis tested

## Checklist
- [x] Code follows project style guidelines
- [x] Documentation updated (README.md)
- [x] Commit messages are descriptive
- [x] No breaking changes

## Key Features
1. **Robust Date Alignment**: Handles mixed timezone formats and trading day matching
2. **Multi-Method Sentiment**: Combines TextBlob and VADER for accurate sentiment scoring
3. **Statistical Rigor**: Pearson correlation with p-value significance testing
4. **Lagged Analysis**: Identifies optimal time windows for correlation
5. **Comprehensive Output**: CSV files, JSON results, and visualizations

## Files Changed
- `scripts/task3_correlation_analysis.py` - New file
- `notebooks/task_3_correlation_analysis.ipynb` - New notebook
- `README.md` - Updated with Task 3 documentation

## Usage Example
```bash
python scripts/task3_correlation_analysis.py \
  --news data/raw_analyst_ratings.csv \
  --stock data/AAPL.csv \
  --stock-ticker AAPL \
  --output output/task3
```

## Output Files
- `aligned_data.csv`: Aligned sentiment and returns data
- `correlation_results.json`: Correlation statistics
- `lagged_correlations.csv`: Lagged correlation analysis
- `sentiment_vs_returns.png`: Correlation scatter plot
- `sentiment_returns_timeseries.png`: Time series visualization

## Statistical Results
The analysis provides:
- Pearson correlation coefficient (r)
- P-value for significance testing
- Number of observations
- Lagged correlations (0-5 days)
- Best lag identification

