"""
Sample data loading flow demonstration.

This script demonstrates how to load and preprocess analyst ratings data,
showing expected inputs and outputs at each step.

Example:
    python scripts/sample_data_loader.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_analyst_ratings, get_data_info
import pandas as pd


def main():
    """Demonstrate data loading workflow."""
    print("=" * 60)
    print("Sample Data Loading Flow")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data from 'data/raw_analyst_ratings.csv'...")
    print("Input: CSV file path")
    print("Expected columns: headline, url, publisher, date, stock")
    
    try:
        df = load_analyst_ratings('data/raw_analyst_ratings.csv')
        print(f"✓ Successfully loaded data")
    except FileNotFoundError:
        print("✗ Data file not found. Creating sample data for demonstration...")
        # Create sample data
        df = pd.DataFrame({
            'headline': [
                'Apple stock rises 5% after strong earnings',
                'Microsoft shares fall on weak guidance',
                'Tesla receives FDA approval'
            ],
            'url': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3'],
            'publisher': ['Publisher A', 'Publisher B', 'Publisher A'],
            'date': pd.to_datetime(['2020-06-05 10:00:00', '2020-06-06 11:00:00', '2020-06-07 09:00:00']),
            'stock': ['AAPL', 'MSFT', 'TSLA']
        })
        # Add temporal features manually for demo
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['date_only'] = df['date'].dt.date
        print(f"✓ Created sample dataset with {len(df)} records")
    
    # Step 2: Display data info
    print("\n[Step 2] Extracting data information...")
    info = get_data_info(df)
    
    print("\nOutput: Data information dictionary")
    print(f"  Total records: {info['total_records']:,}")
    print(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
    print(f"  Unique publishers: {info['unique_publishers']:,}")
    print(f"  Unique stocks: {info['unique_stocks']:,}")
    print(f"  Columns: {len(info['columns'])} columns")
    
    # Step 3: Show sample data
    print("\n[Step 3] Sample data preview...")
    print("\nFirst 3 rows:")
    print(df[['headline', 'publisher', 'stock', 'date', 'year', 'month', 'day_of_week']].head(3).to_string(index=False))
    
    # Step 4: Show data quality
    print("\n[Step 4] Data quality check...")
    missing = info['missing_values']
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    if missing_cols:
        print("  Missing values found:")
        for col, count in missing_cols.items():
            print(f"    {col}: {count:,} ({count/info['total_records']*100:.2f}%)")
    else:
        print("  ✓ No missing values in key columns")
    
    print("\n" + "=" * 60)
    print("Data Loading Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run text analysis: python scripts/sample_text_analysis.py")
    print("  2. Run sentiment analysis: python scripts/sample_sentiment.py")
    print("  3. Run full pipeline: python scripts/end_to_end_example.py")


if __name__ == '__main__':
    main()

