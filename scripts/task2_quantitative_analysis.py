"""
Task 2: Quantitative Analysis using TA-Lib and Financial Metrics

This script demonstrates comprehensive quantitative analysis including:
- Loading and preparing stock price data
- Calculating technical indicators with TA-Lib
- Computing financial metrics
- Creating visualizations

Example:
    python scripts/task2_quantitative_analysis.py --stock data/AAPL.csv --output output/task2_analysis
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.technical_indicators import TechnicalAnalyzer
from src.financial_metrics import FinancialMetrics
from src.visualization import StockVisualizer


def main():
    parser = argparse.ArgumentParser(description='Task 2: Quantitative Analysis')
    parser.add_argument('--stock', required=True, help='Path to stock price CSV file')
    parser.add_argument('--output', default='output/task2_analysis', 
                       help='Output directory for results (default: output/task2_analysis)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Task 2: Quantitative Analysis")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n[Step 1] Loading and preparing stock price data...")
    loader = DataLoader()
    df = loader.load_stock_price_data(args.stock)
    
    # Get data info
    info = loader.get_data_info(df)
    print(f"✓ Loaded {info['total_records']:,} records")
    print(f"✓ Date range: {info['date_range'][0]} to {info['date_range'][1]}")
    print(f"✓ Columns: {', '.join([c for c in df.columns if c != 'date'])}")
    
    # Step 2: Calculate technical indicators with TA-Lib
    print("\n[Step 2] Calculating technical indicators with TA-Lib...")
    ta = TechnicalAnalyzer(close_col='close', high_col='high', low_col='low', volume_col='volume')
    
    # Calculate all indicators
    df_with_indicators = ta.calculate_all_indicators(df)
    print("✓ Calculated indicators:")
    print("  - Simple Moving Average (SMA)")
    print("  - Exponential Moving Average (EMA)")
    print("  - Relative Strength Index (RSI)")
    print("  - MACD (Moving Average Convergence Divergence)")
    print("  - Bollinger Bands")
    
    # Step 3: Calculate financial metrics
    print("\n[Step 3] Calculating financial metrics...")
    metrics = FinancialMetrics(risk_free_rate=0.02)
    df_with_metrics = metrics.calculate_all_metrics(df_with_indicators, price_col='close')
    print("✓ Calculated metrics:")
    print("  - Daily returns")
    print("  - Log returns")
    print("  - Volatility (30-day and 252-day)")
    print("  - Sharpe ratio (252-day)")
    print("  - Maximum drawdown (252-day)")
    
    # Calculate summary statistics
    print("\n[Summary Statistics]")
    print(f"Average daily return: {df_with_metrics['daily_return'].mean():.4f}%")
    print(f"Volatility (30-day): {df_with_metrics['volatility_30d'].iloc[-1]:.4f}%")
    print(f"Volatility (252-day): {df_with_metrics['volatility_252d'].iloc[-1]:.4f}%")
    print(f"Sharpe ratio (252-day): {df_with_metrics['sharpe_252d'].iloc[-1]:.4f}")
    print(f"Maximum drawdown (252-day): {df_with_metrics['max_drawdown_252d'].iloc[-1]:.4f}%")
    
    # Step 4: Visualize the data
    print("\n[Step 4] Creating visualizations...")
    viz = StockVisualizer(figsize=(14, 8))
    
    # Plot 1: Price with volume
    print("  Creating price chart with volume...")
    fig1 = viz.plot_price_data(df_with_metrics, price_col='close', 
                               title=f'Stock Price with Volume - {Path(args.stock).stem}')
    viz.save_plot(output_dir / 'price_with_volume.png')
    viz.close()
    
    # Plot 2: Price with moving averages and Bollinger Bands
    print("  Creating price chart with indicators...")
    fig2 = viz.plot_price_with_indicators(
        df_with_metrics, 
        indicators=['sma', 'ema', 'bb_upper', 'bb_lower'],
        title=f'Price with Technical Indicators - {Path(args.stock).stem}'
    )
    viz.save_plot(output_dir / 'price_with_indicators.png')
    viz.close()
    
    # Plot 3: Technical indicators subplots
    print("  Creating technical indicators analysis...")
    fig3 = viz.plot_technical_indicators(
        df_with_metrics,
        title=f'Technical Indicators Analysis - {Path(args.stock).stem}'
    )
    viz.save_plot(output_dir / 'technical_indicators.png')
    viz.close()
    
    # Plot 4: Returns distribution
    print("  Creating returns distribution...")
    fig4 = viz.plot_returns_distribution(
        df_with_metrics['daily_return'],
        title=f'Daily Returns Distribution - {Path(args.stock).stem}'
    )
    viz.save_plot(output_dir / 'returns_distribution.png')
    viz.close()
    
    print(f"✓ Saved all visualizations to {output_dir}/")
    
    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    output_csv = output_dir / 'analysis_results.csv'
    df_with_metrics.to_csv(output_csv, index=False)
    print(f"✓ Saved analysis results to {output_csv}")
    
    print("\n" + "=" * 60)
    print("Task 2 Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - analysis_results.csv: Full dataset with indicators and metrics")
    print(f"  - price_with_volume.png: Price chart with volume")
    print(f"  - price_with_indicators.png: Price with moving averages and Bollinger Bands")
    print(f"  - technical_indicators.png: RSI, MACD, and Bollinger Bands analysis")
    print(f"  - returns_distribution.png: Distribution of daily returns")


if __name__ == '__main__':
    main()

