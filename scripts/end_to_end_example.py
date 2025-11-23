"""
End-to-end example demonstrating the complete analysis pipeline.

This script shows how to:
1. Load and preprocess news data
2. Perform text analysis
3. Calculate sentiment scores
4. Load stock price data
5. Calculate technical indicators
6. Analyze correlation between sentiment and stock returns

Example:
    python scripts/end_to_end_example.py
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_analyst_ratings, get_data_info
from src.text_analysis import get_top_keywords, extract_financial_keywords
from src.sentiment_analysis import SentimentAnalyzer
from src.technical_indicators import TechnicalAnalyzer
from src.correlation_analysis import CorrelationAnalyzer


def main():
    """Run end-to-end analysis pipeline."""
    print("=" * 60)
    print("End-to-End News Sentiment Trading Analysis")
    print("=" * 60)
    
    # Step 1: Load news data
    print("\n[Step 1] Loading news data...")
    try:
        df = load_analyst_ratings('data/raw_analyst_ratings.csv')
        info = get_data_info(df)
        print(f"✓ Loaded {info['total_records']:,} records")
        print(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        print(f"  Unique publishers: {info['unique_publishers']:,}")
        print(f"  Unique stocks: {info['unique_stocks']:,}")
    except FileNotFoundError:
        print("✗ Data file not found. Using sample data...")
        # Create sample data for demonstration
        df = pd.DataFrame({
            'headline': [
                'Apple stock rises 5% after strong earnings',
                'Microsoft shares fall on weak guidance',
                'Tesla receives FDA approval for new product',
                'Amazon price target raised to $200',
                'Google maintains buy rating'
            ],
            'date': pd.to_datetime([
                '2020-06-05', '2020-06-06', '2020-06-07', '2020-06-08', '2020-06-09'
            ]),
            'publisher': ['Publisher A', 'Publisher B', 'Publisher A', 'Publisher C', 'Publisher B'],
            'stock': ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']
        })
        print(f"✓ Created sample dataset with {len(df)} records")
    
    # Step 2: Text Analysis
    print("\n[Step 2] Performing text analysis...")
    print("  Extracting top keywords...")
    keywords = get_top_keywords(df.head(1000), 'headline', top_n=10)
    print(f"✓ Top 10 keywords:")
    for keyword, count in keywords.items():
        print(f"    {keyword}: {count}")
    
    print("  Extracting financial keywords...")
    financial_terms = extract_financial_keywords(df.head(1000))
    print(f"✓ Found {len(financial_terms)} financial terms")
    top_terms = sorted(financial_terms.items(), key=lambda x: x[1], reverse=True)[:5]
    for term, count in top_terms:
        print(f"    {term}: {count}")
    
    # Step 3: Sentiment Analysis
    print("\n[Step 3] Analyzing sentiment...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze sentiment on sample (for performance)
    sample_size = min(100, len(df))
    sample_df = df.head(sample_size).copy()
    
    print(f"  Analyzing sentiment for {sample_size} headlines...")
    sample_df['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(
        sample_df['headline'], method='combined'
    )
    
    print(f"✓ Calculated sentiment scores")
    print(f"  Mean sentiment: {sample_df['sentiment'].mean():.3f}")
    print(f"  Positive headlines: {(sample_df['sentiment'] > 0.1).sum()}")
    print(f"  Negative headlines: {(sample_df['sentiment'] < -0.1).sum()}")
    
    # Aggregate daily sentiment
    print("  Aggregating daily sentiment...")
    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(sample_df)
    print(f"✓ Aggregated to {len(daily_sentiment)} days")
    if len(daily_sentiment) > 0:
        print(f"  Average daily sentiment: {daily_sentiment['mean_sentiment'].mean():.3f}")
    
    # Step 4: Stock Data (if available)
    print("\n[Step 4] Stock price analysis...")
    try:
        # Try to load stock data
        stock_df = pd.read_csv('data/stock_data_AAPL_2020-01-01_2020-12-31.csv')
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        print(f"✓ Loaded stock data with {len(stock_df)} records")
        
        # Calculate technical indicators
        print("  Calculating technical indicators...")
        ta = TechnicalAnalyzer(close_col='Close', high_col='High', low_col='Low')
        stock_with_indicators = ta.calculate_all_indicators(stock_df)
        print(f"✓ Calculated indicators (SMA, EMA, RSI, MACD, Bollinger Bands)")
        
        # Step 5: Correlation Analysis
        print("\n[Step 5] Correlation analysis...")
        corr_analyzer = CorrelationAnalyzer()
        
        # Align sentiment and stock data
        aligned_sent, aligned_stock = corr_analyzer.align_data(
            daily_sentiment, stock_df, sentiment_date_col='date', stock_date_col='Date'
        )
        
        if len(aligned_sent) > 1 and 'mean_sentiment' in aligned_sent.columns:
            # Calculate returns
            returns = corr_analyzer.calculate_daily_returns(
                aligned_stock, price_col='Close'
            )
            
            # Calculate correlation
            correlation = corr_analyzer.calculate_correlation(
                aligned_sent['mean_sentiment'], returns
            )
            
            print(f"✓ Correlation Results:")
            print(f"  Pearson r: {correlation['pearson_r']:.4f}")
            print(f"  P-value: {correlation['p_value']:.4f}")
            print(f"  Significant: {correlation['is_significant']}")
            print(f"  Observations: {correlation['n_observations']}")
            
            # Lagged correlation
            print("  Analyzing lagged correlations...")
            lagged = corr_analyzer.analyze_lagged_correlations(
                daily_sentiment, stock_df, max_lag=3
            )
            print(f"✓ Lagged correlation analysis complete")
            print(lagged.to_string(index=False))
        else:
            print("⚠ Insufficient aligned data for correlation analysis")
    
    except FileNotFoundError:
        print("⚠ Stock data file not found. Skipping stock analysis.")
        print("  To include stock analysis, run:")
        print("    python scripts/ingest_stock_data.py --tickers AAPL --start 2020-01-01 --end 2020-12-31")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

