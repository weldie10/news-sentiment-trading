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
├── src/                  # Source code modules
│   ├── data_loader.py
│   ├── text_analysis.py
│   └── publisher_analysis.py
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

1. **Run EDA analysis:**
```bash
jupyter notebook notebooks/task_1_eda.ipynb
```

2. **Run tests:**
```bash
pytest tests/
```

## Key Features

- **Exploratory Data Analysis**: Comprehensive analysis of news publication patterns, publisher activity, and temporal trends
- **Text Analysis**: NLP-based topic modeling and keyword extraction from financial headlines
- **Technical Analysis**: TA-Lib integration for calculating technical indicators (RSI, MACD, moving averages)
- **Sentiment Analysis**: Correlation analysis between news sentiment and stock returns

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

