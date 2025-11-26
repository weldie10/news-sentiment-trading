"""
Task 3: Correlation between News and Stock Movement

This script demonstrates comprehensive correlation analysis including:
- Date alignment between news and stock datasets
- Sentiment analysis on news headlines
- Calculation of daily stock returns
- Correlation analysis between sentiment and returns

Example:
    python scripts/task3_correlation_analysis.py --news data/raw_analyst_ratings.csv --stock data/AAPL.csv --output output/task3_analysis
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.sentiment_analysis import SentimentAnalyzer
from src.correlation_analysis import CorrelationAnalyzer
from src.visualization import StockVisualizer


def main():
    parser = argparse.ArgumentParser(description='Task 3: Correlation Analysis')
    parser.add_argument('--news', required=True, help='Path to news/analyst ratings CSV file')
    parser.add_argument('--stock', required=True, help='Path to stock price CSV file')
    parser.add_argument('--stock-ticker', default=None, 
                       help='Stock ticker to filter news (e.g., AAPL). If None, uses first stock in data')
    parser.add_argument('--output', default='output/task3_analysis',
                       help='Output directory for results (default: output/task3_analysis)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Task 3: Correlation between News and Stock Movement")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[Step 1] Loading and preparing data...")
    loader = DataLoader()
    
    # Load news data
    print(f"  Loading news data from {args.news}...")
    news_df = loader.load_analyst_ratings(args.news)
    print(f"  ✓ Loaded {len(news_df):,} news articles")
    
    # Filter by stock ticker if specified
    if args.stock_ticker:
        news_df = news_df[news_df['stock'] == args.stock_ticker.upper()]
        print(f"  ✓ Filtered to {len(news_df):,} articles for {args.stock_ticker.upper()}")
    elif 'stock' in news_df.columns:
        # Use first stock in dataset
        first_stock = news_df['stock'].iloc[0]
        news_df = news_df[news_df['stock'] == first_stock]
        print(f"  ✓ Filtered to {len(news_df):,} articles for {first_stock}")
    
    # Load stock data
    print(f"  Loading stock data from {args.stock}...")
    stock_df = loader.load_stock_price_data(args.stock)
    print(f"  ✓ Loaded {len(stock_df):,} stock price records")
    
    # Display date ranges
    news_info = loader.get_data_info(news_df)
    stock_info = loader.get_data_info(stock_df)
    print(f"\n  News date range: {news_info['date_range'][0]} to {news_info['date_range'][1]}")
    print(f"  Stock date range: {stock_info['date_range'][0]} to {stock_info['date_range'][1]}")
    
    # Step 2: Perform sentiment analysis
    print("\n[Step 2] Performing sentiment analysis on news headlines...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze sentiment for all headlines
    print("  Analyzing sentiment using combined method (TextBlob + VADER)...")
    news_df['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(
        news_df['headline'], method='combined'
    )
    
    # Classify sentiment
    news_df['sentiment_label'] = news_df['sentiment'].apply(
        sentiment_analyzer.classify_sentiment
    )
    
    print(f"  ✓ Analyzed sentiment for {len(news_df):,} headlines")
    print(f"  ✓ Positive: {(news_df['sentiment_label'] == 'positive').sum():,}")
    print(f"  ✓ Negative: {(news_df['sentiment_label'] == 'negative').sum():,}")
    print(f"  ✓ Neutral: {(news_df['sentiment_label'] == 'neutral').sum():,}")
    
    # Aggregate daily sentiment
    print("\n  Aggregating daily sentiment scores...")
    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(
        news_df, date_col='date', sentiment_col='sentiment'
    )
    print(f"  ✓ Aggregated to {len(daily_sentiment):,} trading days")
    
    # Step 3: Normalize dates and align data
    print("\n[Step 3] Normalizing dates and aligning news and stock data...")
    correlation_analyzer = CorrelationAnalyzer()
    
    # Align data by date
    aligned_sent, aligned_stock = correlation_analyzer.align_data(
        daily_sentiment, stock_df, 
        sentiment_date_col='date', 
        stock_date_col='date'
    )
    
    print(f"  ✓ Aligned {len(aligned_sent):,} matching dates")
    
    # Step 4: Calculate stock movements (daily returns)
    print("\n[Step 4] Calculating daily stock returns...")
    returns = correlation_analyzer.calculate_daily_returns(aligned_stock, price_col='close')
    aligned_stock['daily_return'] = returns
    
    print(f"  ✓ Calculated daily returns for {returns.notna().sum():,} days")
    print(f"  Average daily return: {returns.mean():.4f}%")
    print(f"  Standard deviation: {returns.std():.4f}%")
    
    # Step 5: Correlation analysis
    print("\n[Step 5] Performing correlation analysis...")
    
    # Get sentiment scores
    if 'mean_sentiment' in aligned_sent.columns:
        sentiment_scores = aligned_sent['mean_sentiment']
    else:
        sentiment_scores = aligned_sent['sentiment']
    
    # Calculate correlation
    correlation_result = correlation_analyzer.calculate_correlation(
        sentiment_scores, returns
    )
    
    print("\n=== Correlation Results ===")
    print(f"Pearson correlation coefficient (r): {correlation_result['pearson_r']:.4f}")
    print(f"P-value: {correlation_result['p_value']:.4f}")
    print(f"Significant (p < 0.05): {correlation_result['is_significant']}")
    print(f"Number of observations: {correlation_result['n_observations']:,}")
    
    # Analyze lagged correlations
    print("\n[Step 6] Analyzing lagged correlations...")
    lagged_correlations = correlation_analyzer.analyze_lagged_correlations(
        daily_sentiment, stock_df, max_lag=5
    )
    
    print("\n=== Lagged Correlation Results ===")
    print(lagged_correlations.to_string(index=False))
    
    # Find best lag
    significant_lags = lagged_correlations[lagged_correlations['is_significant']]
    if len(significant_lags) > 0:
        best_lag = significant_lags.loc[significant_lags['pearson_r'].abs().idxmax()]
        print(f"\nBest lag: {int(best_lag['lag'])} days")
        print(f"  Correlation: {best_lag['pearson_r']:.4f}")
        print(f"  P-value: {best_lag['p_value']:.4f}")
    
    # Step 6: Visualize results
    print("\n[Step 7] Creating visualizations...")
    viz = StockVisualizer(figsize=(14, 8))
    
    # Plot 1: Correlation scatter plot
    print("  Creating correlation scatter plot...")
    fig1 = viz.plot_correlation(
        sentiment_scores, returns,
        title='News Sentiment vs Stock Returns',
        xlabel='Average Daily Sentiment Score',
        ylabel='Daily Returns (%)'
    )
    viz.save_plot(output_dir / 'sentiment_vs_returns.png')
    viz.close()
    
    # Plot 2: Time series of sentiment and returns
    print("  Creating time series plot...")
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot sentiment
    ax1.plot(aligned_sent['date'], sentiment_scores, linewidth=1.5, 
            color='blue', label='Average Sentiment')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Sentiment Score', fontsize=12)
    ax1.set_title('Daily News Sentiment Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot returns
    ax2.plot(aligned_stock['date'], returns, linewidth=1.5, 
            color='green', label='Daily Returns')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Daily Returns (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Daily Stock Returns Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_returns_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Step 7: Save results
    print("\n[Step 8] Saving results...")
    
    # Save aligned data
    aligned_data = aligned_sent.copy()
    aligned_data['daily_return'] = returns.values
    aligned_data.to_csv(output_dir / 'aligned_data.csv', index=False)
    print(f"  ✓ Saved aligned data to {output_dir / 'aligned_data.csv'}")
    
    # Save correlation results
    results_summary = {
        'correlation': correlation_result,
        'lagged_correlations': lagged_correlations.to_dict('records')
    }
    import json
    with open(output_dir / 'correlation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"  ✓ Saved correlation results to {output_dir / 'correlation_results.json'}")
    
    # Save lagged correlations CSV
    lagged_correlations.to_csv(output_dir / 'lagged_correlations.csv', index=False)
    print(f"  ✓ Saved lagged correlations to {output_dir / 'lagged_correlations.csv'}")
    
    print("\n" + "=" * 60)
    print("Task 3 Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - aligned_data.csv: Aligned sentiment and returns data")
    print(f"  - correlation_results.json: Correlation statistics")
    print(f"  - lagged_correlations.csv: Lagged correlation analysis")
    print(f"  - sentiment_vs_returns.png: Correlation scatter plot")
    print(f"  - sentiment_returns_timeseries.png: Time series visualization")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()

