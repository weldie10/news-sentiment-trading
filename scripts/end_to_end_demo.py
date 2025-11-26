"""
End-to-end demonstration with sample outputs.

This script demonstrates the complete workflow from data loading through
all three tasks, with documented sample outputs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.technical_indicators import TechnicalAnalyzer
from src.sentiment_analysis import SentimentAnalyzer
from src.correlation_analysis import CorrelationAnalyzer
from src.financial_metrics import FinancialMetrics
from src.visualization import StockVisualizer


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")
    
    # Sample news data
    news_data = pd.DataFrame({
        'headline': [
            'Apple stock rises 5% after strong earnings report',
            'Microsoft shares fall on weak quarterly guidance',
            'Tesla receives FDA approval for new product line',
            'Amazon price target raised to $200 by analysts',
            'Google maintains strong buy rating from major firms',
            'Apple announces new product launch event',
            'Microsoft cloud revenue exceeds expectations',
            'Tesla stock surges on positive delivery numbers',
            'Amazon reports record holiday sales',
            'Google AI breakthrough attracts investor attention'
        ],
        'date': pd.to_datetime([
            '2020-01-02 10:00:00',
            '2020-01-03 11:00:00',
            '2020-01-06 09:00:00',
            '2020-01-07 14:00:00',
            '2020-01-08 15:00:00',
            '2020-01-09 10:30:00',
            '2020-01-10 11:15:00',
            '2020-01-13 09:45:00',
            '2020-01-14 13:20:00',
            '2020-01-15 16:00:00'
        ]),
        'publisher': ['TechNews', 'FinanceDaily', 'MarketWatch', 'TechNews', 'FinanceDaily',
                     'TechNews', 'MarketWatch', 'FinanceDaily', 'TechNews', 'MarketWatch'],
        'stock': ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL'],
        'url': [f'http://example.com/{i}' for i in range(10)]
    })
    
    # Sample stock data (AAPL)
    dates = pd.date_range('2020-01-02', '2020-01-15', freq='B')  # Business days only
    np.random.seed(42)
    base_price = 150.0
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
    
    stock_data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(len(dates)) * 0.5,
        'High': prices + abs(np.random.randn(len(dates)) * 1.5),
        'Low': prices - abs(np.random.randn(len(dates)) * 1.5),
        'Close': prices,
        'Volume': np.random.randint(50000000, 150000000, len(dates))
    })
    
    return news_data, stock_data


def main():
    """Run end-to-end demonstration."""
    print("=" * 80)
    print("END-TO-END DEMONSTRATION: News Sentiment Trading Analysis")
    print("=" * 80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    output_dir = Path('output/end_to_end_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create and load sample data
    print("\n[STEP 1] Data Loading and Preparation")
    print("-" * 80)
    news_df, stock_df = create_sample_data()
    
    loader = DataLoader()
    # Save sample data
    news_df.to_csv(output_dir / 'sample_news.csv', index=False)
    stock_df.to_csv(output_dir / 'sample_stock.csv', index=False)
    
    # Load using DataLoader
    stock_df_loaded = loader.load_stock_price_data(str(output_dir / 'sample_stock.csv'))
    
    print(f"✓ Loaded {len(news_df)} news articles")
    print(f"✓ Loaded {len(stock_df_loaded)} stock price records")
    print(f"✓ Date range: {stock_df_loaded['date'].min()} to {stock_df_loaded['date'].max()}")
    
    # Step 2: Technical Analysis
    print("\n[STEP 2] Technical Indicators Calculation")
    print("-" * 80)
    ta = TechnicalAnalyzer(close_col='close', high_col='high', low_col='low', volume_col='volume')
    df_with_indicators = ta.calculate_all_indicators(stock_df_loaded)
    
    print("✓ Calculated indicators:")
    print(f"  - SMA (20-day): {df_with_indicators['sma'].notna().sum()} values")
    print(f"  - EMA (20-day): {df_with_indicators['ema'].notna().sum()} values")
    print(f"  - RSI (14-day): {df_with_indicators['rsi'].notna().sum()} values")
    print(f"  - MACD: {df_with_indicators['macd'].notna().sum()} values")
    print(f"  - Bollinger Bands: {df_with_indicators['bb_upper'].notna().sum()} values")
    
    # Step 3: Financial Metrics
    print("\n[STEP 3] Financial Metrics Calculation")
    print("-" * 80)
    metrics = FinancialMetrics(risk_free_rate=0.02)
    df_with_metrics = metrics.calculate_all_metrics(df_with_indicators, price_col='close')
    
    latest_metrics = {
        'average_daily_return': float(df_with_metrics['daily_return'].mean()),
        'volatility_30d': float(df_with_metrics['volatility_30d'].iloc[-1]) if pd.notna(df_with_metrics['volatility_30d'].iloc[-1]) else None,
        'sharpe_252d': float(df_with_metrics['sharpe_252d'].iloc[-1]) if pd.notna(df_with_metrics['sharpe_252d'].iloc[-1]) else None
    }
    
    print("✓ Calculated metrics:")
    print(f"  - Average daily return: {latest_metrics['average_daily_return']:.4f}%")
    if latest_metrics['volatility_30d']:
        print(f"  - Volatility (30-day): {latest_metrics['volatility_30d']:.4f}%")
    if latest_metrics['sharpe_252d']:
        print(f"  - Sharpe ratio (252-day): {latest_metrics['sharpe_252d']:.4f}")
    
    # Step 4: Sentiment Analysis
    print("\n[STEP 4] Sentiment Analysis")
    print("-" * 80)
    sentiment_analyzer = SentimentAnalyzer()
    news_df['sentiment'] = sentiment_analyzer.analyze_sentiment_batch(news_df['headline'], method='combined')
    news_df['sentiment_label'] = news_df['sentiment'].apply(sentiment_analyzer.classify_sentiment)
    
    sentiment_dist = news_df['sentiment_label'].value_counts().to_dict()
    print(f"✓ Analyzed sentiment for {len(news_df)} headlines")
    print(f"  - Positive: {sentiment_dist.get('positive', 0)}")
    print(f"  - Negative: {sentiment_dist.get('negative', 0)}")
    print(f"  - Neutral: {sentiment_dist.get('neutral', 0)}")
    print(f"  - Average sentiment: {news_df['sentiment'].mean():.4f}")
    
    # Aggregate daily sentiment
    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(news_df, date_col='date', sentiment_col='sentiment')
    print(f"✓ Aggregated to {len(daily_sentiment)} trading days")
    
    # Step 5: Correlation Analysis
    print("\n[STEP 5] Correlation Analysis")
    print("-" * 80)
    correlation_analyzer = CorrelationAnalyzer()
    
    # Align data
    aligned_sent, aligned_stock = correlation_analyzer.align_data(
        daily_sentiment, stock_df_loaded,
        sentiment_date_col='date',
        stock_date_col='date'
    )
    
    print(f"✓ Aligned {len(aligned_sent)} matching dates")
    
    # Calculate returns
    returns = correlation_analyzer.calculate_daily_returns(aligned_stock, price_col='close')
    
    # Get sentiment scores
    sentiment_scores = aligned_sent['mean_sentiment'] if 'mean_sentiment' in aligned_sent.columns else aligned_sent['sentiment']
    
    # Calculate correlation
    correlation_result = correlation_analyzer.calculate_correlation(sentiment_scores, returns)
    
    print("✓ Correlation Results:")
    print(f"  - Pearson r: {correlation_result['pearson_r']:.4f}")
    print(f"  - P-value: {correlation_result['p_value']:.4f}")
    print(f"  - Significant (p < 0.05): {correlation_result['is_significant']}")
    print(f"  - Observations: {correlation_result['n_observations']}")
    
    # Step 6: Save Results
    print("\n[STEP 6] Saving Results")
    print("-" * 80)
    
    # Save DataFrames
    df_with_metrics.to_csv(output_dir / 'stock_with_metrics.csv', index=False)
    news_df.to_csv(output_dir / 'news_with_sentiment.csv', index=False)
    aligned_sent.to_csv(output_dir / 'aligned_sentiment.csv', index=False)
    
    # Save summary
    summary = {
        'execution_time': datetime.now().isoformat(),
        'data_summary': {
            'news_articles': len(news_df),
            'stock_records': len(stock_df_loaded),
            'aligned_dates': len(aligned_sent)
        },
        'sentiment_summary': {
            'positive': sentiment_dist.get('positive', 0),
            'negative': sentiment_dist.get('negative', 0),
            'neutral': sentiment_dist.get('neutral', 0),
            'average_sentiment': float(news_df['sentiment'].mean())
        },
        'financial_metrics': latest_metrics,
        'correlation_results': {
            'pearson_r': float(correlation_result['pearson_r']) if pd.notna(correlation_result['pearson_r']) else None,
            'p_value': float(correlation_result['p_value']) if pd.notna(correlation_result['p_value']) else None,
            'is_significant': bool(correlation_result['is_significant']),
            'n_observations': int(correlation_result['n_observations'])
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved all results to {output_dir}/")
    print(f"  - stock_with_metrics.csv")
    print(f"  - news_with_sentiment.csv")
    print(f"  - aligned_sentiment.csv")
    print(f"  - summary.json")
    
    # Step 7: Visualization
    print("\n[STEP 7] Creating Visualizations")
    print("-" * 80)
    viz = StockVisualizer(figsize=(12, 6))
    
    try:
        fig = viz.plot_price_data(df_with_metrics, price_col='close', title='Sample Stock Price with Volume')
        viz.save_plot(output_dir / 'price_chart.png')
        viz.close()
        print("✓ Created price chart")
        
        fig = viz.plot_correlation(sentiment_scores, returns, 
                                   title='Sentiment vs Returns Correlation',
                                   xlabel='Average Daily Sentiment',
                                   ylabel='Daily Returns (%)')
        viz.save_plot(output_dir / 'correlation_chart.png')
        viz.close()
        print("✓ Created correlation chart")
    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")
    
    print("\n" + "=" * 80)
    print("END-TO-END DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nSample Output Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

