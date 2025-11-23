"""
Stock data ingestion script using yfinance.

Downloads stock price data for specified tickers and saves to CSV.

Example:
    python scripts/ingest_stock_data.py --tickers AAPL MSFT --start 2020-01-01 --end 2020-12-31
"""
import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime


def download_stock_data(tickers: list, start_date: str, end_date: str,
                       output_dir: str = 'data') -> pd.DataFrame:
    """
    Download stock price data for given tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save data
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading data for {tickers} from {start_date} to {end_date}")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Flatten multi-index if single ticker
    if len(tickers) == 1:
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"stock_data_{'_'.join(tickers)}_{start_date}_{end_date}.csv"
    filepath = output_path / filename
    
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download stock price data')
    parser.add_argument('--tickers', nargs='+', required=True,
                       help='Stock ticker symbols (e.g., AAPL MSFT)')
    parser.add_argument('--start', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    download_stock_data(args.tickers, args.start, args.end, args.output_dir)

