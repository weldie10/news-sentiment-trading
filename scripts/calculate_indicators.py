"""
Script to calculate technical indicators for stock data.

Example:
    python scripts/calculate_indicators.py --input data/stock_data.csv --output data/stock_with_indicators.csv
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.technical_indicators import TechnicalAnalyzer
from src.data_loader import DataLoader
from src.financial_metrics import FinancialMetrics


def main():
    parser = argparse.ArgumentParser(description='Calculate technical indicators')
    parser.add_argument('--input', required=True, help='Input CSV file with OHLCV data')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Load data using DataLoader class
    print(f"Loading data from {args.input}")
    loader = DataLoader()
    df = loader.load_stock_price_data(args.input)
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    ta = TechnicalAnalyzer(close_col='close', high_col='high', low_col='low', volume_col='volume')
    df_with_indicators = ta.calculate_all_indicators(df)
    
    # Calculate financial metrics
    print("Calculating financial metrics...")
    metrics = FinancialMetrics()
    df_with_indicators = metrics.calculate_all_metrics(df_with_indicators, price_col='close')
    
    # Save results
    print(f"Saving results to {args.output}")
    df_with_indicators.to_csv(args.output, index=False)
    print("Done!")


if __name__ == '__main__':
    main()

