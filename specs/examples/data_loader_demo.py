"""
Banking Data Loader Demo Script

This script demonstrates how to use the BankingDataLoader to work with
the banking marketing dataset. It shows various features including:

1. Database connection and validation
2. Loading data with different parameters
3. Exporting data to CSV
4. Generating data summaries
5. Creating filtered datasets

Usage:
    python examples/data_loader_demo.py

Requirements:
    - data/raw/bmarket.db must exist
    - pandas, sqlite3 (standard library)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import BankingDataLoader, create_initial_dataset_snapshot
import pandas as pd


def main():
    """Main demonstration function."""
    print("Banking Data Loader Demo")
    print("=" * 50)
    
    try:
        # 1. Initialize the data loader
        print("\n1. Initializing BankingDataLoader...")
        loader = BankingDataLoader()
        print(f"   ✓ Connected to database: {loader.db_path}")
        print(f"   ✓ Default table: {loader.default_table}")
        
        # 2. Get table information
        print("\n2. Getting table information...")
        table_info = loader.get_table_info()
        print(f"   ✓ Table: {table_info['table_name']}")
        print(f"   ✓ Rows: {table_info['row_count']:,}")
        print(f"   ✓ Columns: {table_info['column_count']}")
        
        print("\n   Column Details:")
        for col in table_info['columns']:
            print(f"     - {col['name']} ({col['type']})")
        
        # 3. Load sample data
        print("\n3. Loading sample data...")
        df_sample = loader.load_data(limit=5)
        print(f"   ✓ Loaded {len(df_sample)} rows")
        print("\n   Sample data:")
        print(df_sample.to_string(index=False))
        
        # 4. Generate data summary
        print("\n4. Generating data summary...")
        summary = loader.get_data_summary()
        print(f"   ✓ Sample shape: {summary['sample_shape']}")
        print(f"   ✓ Memory usage: {summary['memory_usage_mb']} MB")
        
        print("\n   Data types:")
        for col, dtype in summary['data_types'].items():
            print(f"     - {col}: {dtype}")
        
        # 5. Load filtered data
        print("\n5. Loading filtered data (subscribers only)...")
        df_subscribers = loader.load_data(
            query="SELECT * FROM bank_marketing WHERE \"Subscription Status\" = 'yes' LIMIT 10"
        )
        print(f"   ✓ Found {len(df_subscribers)} subscribers in sample")
        
        # Show subscription distribution
        subscription_counts = df_subscribers['Subscription Status'].value_counts()
        print(f"   ✓ All records have 'yes' status: {subscription_counts}")
        
        # 6. Export sample to CSV
        print("\n6. Exporting sample data to CSV...")
        export_summary = loader.export_to_csv(
            'data/processed/demo_sample.csv',
            query="SELECT * FROM bank_marketing LIMIT 100"
        )
        print(f"   ✓ Exported to: {export_summary['file_path']}")
        print(f"   ✓ Rows exported: {export_summary['row_count']}")
        print(f"   ✓ File size: {export_summary['file_size_mb']} MB")
        
        # 7. Demonstrate data quality insights
        print("\n7. Data quality insights...")
        df_quality = loader.load_data(limit=1000)
        
        # Check for missing values
        missing_counts = df_quality.isnull().sum()
        print("   Missing values per column:")
        for col, count in missing_counts.items():
            if count > 0:
                percentage = (count / len(df_quality)) * 100
                print(f"     - {col}: {count} ({percentage:.1f}%)")
        
        # Check unique values in key columns
        print("\n   Unique values in key columns:")
        key_columns = ['Subscription Status', 'Contact Method', 'Credit Default']
        for col in key_columns:
            if col in df_quality.columns:
                unique_vals = df_quality[col].unique()
                print(f"     - {col}: {list(unique_vals)}")
        
        # 8. Performance demonstration
        print("\n8. Performance demonstration...")
        import time
        
        start_time = time.time()
        df_large = loader.load_data(limit=10000)
        load_time = time.time() - start_time
        
        print(f"   ✓ Loaded {len(df_large):,} rows in {load_time:.2f} seconds")
        print(f"   ✓ Loading speed: {len(df_large)/load_time:.0f} rows/second")
        
        memory_usage = df_large.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   ✓ Memory usage: {memory_usage:.2f} MB")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("- Use loader.load_data() to get the full dataset")
        print("- Apply data cleaning and preprocessing")
        print("- Create features for machine learning")
        print("- Export processed data for analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\nTroubleshooting:")
        print("- Ensure data/raw/bmarket.db exists")
        print("- Check that all dependencies are installed")
        print("- Verify database file is not corrupted")
        return 1
    
    return 0


def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    print("\n" + "=" * 50)
    print("Advanced Usage Examples")
    print("=" * 50)
    
    loader = BankingDataLoader()
    
    # Example 1: Age-based analysis
    print("\n1. Age-based analysis...")
    df_age = loader.load_data(
        query="""
        SELECT Age, "Subscription Status", COUNT(*) as count
        FROM bank_marketing 
        WHERE Age LIKE '%3_ years' OR Age LIKE '%4_ years'
        GROUP BY Age, "Subscription Status"
        ORDER BY Age
        LIMIT 20
        """
    )
    print("   Age distribution for 30s and 40s:")
    print(df_age.to_string(index=False))
    
    # Example 2: Occupation analysis
    print("\n2. Occupation-based subscription rates...")
    df_occupation = loader.load_data(
        query="""
        SELECT Occupation, 
               COUNT(*) as total,
               SUM(CASE WHEN "Subscription Status" = 'yes' THEN 1 ELSE 0 END) as subscribed
        FROM bank_marketing 
        GROUP BY Occupation
        ORDER BY subscribed DESC
        LIMIT 10
        """
    )
    print("   Top occupations by subscription count:")
    print(df_occupation.to_string(index=False))
    
    # Example 3: Campaign effectiveness
    print("\n3. Campaign call effectiveness...")
    df_campaign = loader.load_data(
        query="""
        SELECT "Campaign Calls",
               COUNT(*) as total,
               SUM(CASE WHEN "Subscription Status" = 'yes' THEN 1 ELSE 0 END) as subscribed
        FROM bank_marketing 
        WHERE "Campaign Calls" BETWEEN 1 AND 10
        GROUP BY "Campaign Calls"
        ORDER BY "Campaign Calls"
        """
    )
    print("   Subscription rate by number of campaign calls:")
    print(df_campaign.to_string(index=False))


if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        try:
            demonstrate_advanced_usage()
        except Exception as e:
            print(f"\n⚠️  Advanced demo failed: {str(e)}")
    
    sys.exit(exit_code)
