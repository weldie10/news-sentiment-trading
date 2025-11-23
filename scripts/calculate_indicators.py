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


def main():
    parser = argparse.ArgumentParser(description='Calculate technical indicators')
    parser.add_argument('--input', required=True, help='Input CSV file with OHLCV data')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Ensure date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate indicators
    print("Calculating technical indicators...")
    ta = TechnicalAnalyzer()
    df_with_indicators = ta.calculate_all_indicators(df)
    
    # Save results
    print(f"Saving results to {args.output}")
    df_with_indicators.to_csv(args.output, index=False)
    print("Done!")


if __name__ == '__main__':
    main()

