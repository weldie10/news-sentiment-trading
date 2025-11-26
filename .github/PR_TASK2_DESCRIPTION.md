# Pull Request: Task 2 - Quantitative Analysis with OOP Refactoring

## Description
This PR implements Task 2: Quantitative Analysis using TA-Lib and Financial Metrics. The implementation includes a comprehensive refactoring to Object-Oriented Programming (OOP) design with new classes for data loading, financial metrics calculation, and visualization.

## Type of Change
- [x] New feature
- [x] Refactoring
- [x] Documentation update

## Related Task
- [x] Task 2: Quantitative Analysis

## Changes Made

### Core Classes (OOP Refactoring)
- **DataLoader Class** (`src/data_loader.py`):
  - Refactored from functions to class-based design
  - Added data caching for performance
  - Methods: `load_analyst_ratings()`, `load_stock_price_data()`, `load_multiple_stocks()`, `get_data_info()`, `clear_cache()`
  - Backward compatibility functions maintained

- **FinancialMetrics Class** (`src/financial_metrics.py`):
  - New class for calculating financial metrics
  - Methods: `calculate_daily_returns()`, `calculate_volatility()`, `calculate_sharpe_ratio()`, `calculate_max_drawdown()`, `calculate_all_metrics()`
  - Integration with yfinance for additional data sources
  - Risk-adjusted performance metrics

- **StockVisualizer Class** (`src/visualization.py`):
  - New class for comprehensive financial visualizations
  - Methods: `plot_price_data()`, `plot_price_with_indicators()`, `plot_technical_indicators()`, `plot_correlation()`, `plot_returns_distribution()`
  - Professional chart styling and customization

### Scripts and Notebooks
- **task2_quantitative_analysis.py**: End-to-end script for quantitative analysis
- **task_2_quantitative_analysis.ipynb**: Comprehensive Jupyter notebook
- Updated `calculate_indicators.py` to use new class-based approach

### Technical Indicators
- TA-Lib integration: SMA, EMA, RSI, MACD, Bollinger Bands
- All indicators calculated using TechnicalAnalyzer class

### Financial Metrics
- Daily returns and log returns
- Volatility (30-day and 252-day rolling, annualized)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown (rolling and overall)

## Testing
- [x] Code follows OOP principles
- [x] Backward compatibility maintained
- [x] Documentation updated
- [x] Examples provided in scripts and notebooks

## Checklist
- [x] Code follows project style guidelines
- [x] Documentation updated (README.md)
- [x] Commit messages are descriptive
- [x] No breaking changes (backward compatibility maintained)

## Key Features
1. **Object-Oriented Design**: All modules use consistent class-based APIs
2. **Data Caching**: Improved performance with intelligent caching
3. **Comprehensive Metrics**: Volatility, Sharpe ratio, drawdown analysis
4. **Professional Visualizations**: Publication-ready charts and graphs
5. **Extensibility**: Easy to add new indicators and metrics

## Files Changed
- `src/data_loader.py` - Refactored to DataLoader class
- `src/financial_metrics.py` - New file
- `src/visualization.py` - New file
- `scripts/task2_quantitative_analysis.py` - New file
- `scripts/calculate_indicators.py` - Updated to use new classes
- `notebooks/task_2_quantitative_analysis.ipynb` - New notebook
- `README.md` - Updated with Task 2 documentation

## Screenshots/Examples
Run the analysis:
```bash
python scripts/task2_quantitative_analysis.py --stock data/AAPL.csv --output output/task2
```

This generates:
- Price charts with volume
- Technical indicators visualization
- Returns distribution
- Complete analysis results CSV

