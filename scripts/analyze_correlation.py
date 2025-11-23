"""
Script to analyze correlation between sentiment and stock returns.

Example:
    python scripts/analyze_correlation.py --sentiment data/daily_sentiment.csv --stock data/stock_data.csv
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.correlation_analysis import CorrelationAnalyzer
from src.sentiment_analysis import SentimentAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment-stock correlation')
    parser.add_argument('--sentiment', required=True, help='CSV file with sentiment data')
    parser.add_argument('--stock', required=True, help='CSV file with stock price data')
    parser.add_argument('--output', help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading sentiment data from {args.sentiment}")
    sentiment_df = pd.read_csv(args.sentiment)
    
    print(f"Loading stock data from {args.stock}")
    stock_df = pd.read_csv(args.stock)
    
    # Initialize analyzers
    corr_analyzer = CorrelationAnalyzer()
    
    # Align data
    print("Aligning sentiment and stock data...")
    aligned_sent, aligned_stock = corr_analyzer.align_data(sentiment_df, stock_df)
    
    # Calculate returns
    print("Calculating daily returns...")
    returns = corr_analyzer.calculate_daily_returns(aligned_stock)
    
    # Get sentiment scores
    if 'mean_sentiment' in aligned_sent.columns:
        sentiment = aligned_sent['mean_sentiment']
    else:
        sentiment = aligned_sent['sentiment']
    
    # Calculate correlation
    print("Calculating correlation...")
    correlation = corr_analyzer.calculate_correlation(sentiment, returns)
    
    print("\n=== Correlation Results ===")
    print(f"Pearson r: {correlation['pearson_r']:.4f}")
    print(f"P-value: {correlation['p_value']:.4f}")
    print(f"Significant: {correlation['is_significant']}")
    print(f"Observations: {correlation['n_observations']}")
    
    # Analyze lagged correlations
    print("\nAnalyzing lagged correlations...")
    lagged = corr_analyzer.analyze_lagged_correlations(sentiment_df, stock_df)
    
    print("\n=== Lagged Correlation Results ===")
    print(lagged.to_string(index=False))
    
    # Save results if output specified
    if args.output:
        results = {
            'correlation': correlation,
            'lagged_correlations': lagged
        }
        lagged.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

