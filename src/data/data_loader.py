"""
Data Loader Module for Banking Marketing Dataset

This module provides robust data loading functionality for the banking marketing
campaign dataset stored in SQLite format. The primary data source is the
`bmarket.db` database located in the `data/raw/` directory.

Key Features:
- Robust SQLite connection handling with comprehensive error management
- Context managers for automatic resource cleanup
- Detailed logging and error reporting
- Data validation and quality checks
- CSV export functionality for data backup and analysis

Data Source: data/raw/bmarket.db
Primary Table: bank_marketing (41,188 records)

Author: AI Assistant
Date: 2024
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import os


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Custom exception for database connection issues."""
    pass


class DataLoadError(Exception):
    """Custom exception for data loading issues."""
    pass


class BankingDataLoader:
    """
    A robust data loader for the banking marketing dataset.

    This class provides methods to connect to the SQLite database,
    load data with proper error handling, and export data for analysis.

    Data Source: data/raw/bmarket.db
    Primary Table: bank_marketing

    Attributes:
        db_path (Path): Path to the SQLite database file
        default_table (str): Default table name to query
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the BankingDataLoader.

        Args:
            db_path (Optional[str]): Path to the SQLite database.
                                   Defaults to 'data/raw/bmarket.db'

        Raises:
            DatabaseConnectionError: If the database file doesn't exist
        """
        if db_path is None:
            # Default path relative to project root
            self.db_path = Path("data/raw/bmarket.db")
        else:
            self.db_path = Path(db_path)

        self.default_table = "bank_marketing"

        # Validate database exists
        if not self.db_path.exists():
            raise DatabaseConnectionError(
                f"Database file not found: {self.db_path.absolute()}"
            )

        logger.info(f"Initialized BankingDataLoader with database: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for SQLite database connections.

        Provides automatic connection cleanup and comprehensive error handling.

        Yields:
            sqlite3.Connection: Database connection object

        Raises:
            DatabaseConnectionError: If connection fails
        """
        conn = None
        try:
            logger.debug(f"Connecting to database: {self.db_path}")
            conn = sqlite3.connect(str(self.db_path))

            # Enable row factory for better data access
            conn.row_factory = sqlite3.Row

            # Test connection with a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

            logger.debug("Database connection established successfully")
            yield conn

        except sqlite3.Error as e:
            error_msg = f"SQLite error: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to database: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")

    def get_table_info(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive information about a database table.

        Args:
            table_name (Optional[str]): Name of the table to analyze.
                                      Defaults to 'bank_marketing'

        Returns:
            Dict[str, Any]: Dictionary containing table information including:
                - table_name: Name of the table
                - row_count: Number of rows
                - columns: List of column information
                - schema: Table schema details

        Raises:
            DataLoadError: If table information cannot be retrieved
        """
        if table_name is None:
            table_name = self.default_table

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                if not columns:
                    raise DataLoadError(f"Table '{table_name}' not found or has no columns")

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # Format column information
                column_info = []
                for col in columns:
                    column_info.append({
                        'name': col['name'],
                        'type': col['type'],
                        'nullable': not col['notnull'],
                        'default': col['dflt_value'],
                        'primary_key': bool(col['pk'])
                    })

                table_info = {
                    'table_name': table_name,
                    'row_count': row_count,
                    'column_count': len(column_info),
                    'columns': column_info
                }

                logger.info(f"Retrieved info for table '{table_name}': {row_count} rows, {len(column_info)} columns")
                return table_info

        except sqlite3.Error as e:
            error_msg = f"SQLite error getting table info: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error getting table info: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def load_data(self,
                  table_name: Optional[str] = None,
                  query: Optional[str] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from the banking marketing database.

        This method loads data from the specified table or executes a custom query.
        The primary data source is the 'bank_marketing' table in bmarket.db.

        Args:
            table_name (Optional[str]): Name of the table to load.
                                      Defaults to 'bank_marketing'
            query (Optional[str]): Custom SQL query to execute.
                                 If provided, table_name is ignored.
            limit (Optional[int]): Maximum number of rows to return.
                                 Useful for testing and sampling.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame

        Raises:
            DataLoadError: If data loading fails

        Example:
            >>> loader = BankingDataLoader()
            >>> # Load all data
            >>> df = loader.load_data()
            >>> # Load with limit
            >>> df_sample = loader.load_data(limit=1000)
            >>> # Custom query
            >>> df_filtered = loader.load_data(
            ...     query="SELECT * FROM bank_marketing WHERE Age > '30 years'"
            ... )
        """
        if table_name is None:
            table_name = self.default_table

        try:
            with self.get_connection() as conn:
                if query is not None:
                    # Execute custom query
                    logger.info(f"Executing custom query: {query[:100]}...")
                    df = pd.read_sql_query(query, conn)
                else:
                    # Build standard query
                    sql_query = f"SELECT * FROM {table_name}"
                    if limit is not None:
                        sql_query += f" LIMIT {limit}"

                    logger.info(f"Loading data from table '{table_name}'" +
                              (f" with limit {limit}" if limit else ""))
                    df = pd.read_sql_query(sql_query, conn)

                logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                return df

        except sqlite3.Error as e:
            error_msg = f"SQLite error loading data: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading data: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def export_to_csv(self,
                      output_path: str,
                      table_name: Optional[str] = None,
                      query: Optional[str] = None) -> Dict[str, Any]:
        """
        Export banking marketing data to CSV format.

        This method loads data from the database and exports it to a CSV file
        for backup, analysis, and sharing purposes.

        Args:
            output_path (str): Path where the CSV file will be saved
            table_name (Optional[str]): Name of the table to export.
                                      Defaults to 'bank_marketing'
            query (Optional[str]): Custom SQL query for data selection.
                                 If provided, table_name is ignored.

        Returns:
            Dict[str, Any]: Export summary containing:
                - file_path: Path to the exported file
                - row_count: Number of rows exported
                - column_count: Number of columns exported
                - file_size_mb: Size of the exported file in MB

        Raises:
            DataLoadError: If export fails

        Example:
            >>> loader = BankingDataLoader()
            >>> summary = loader.export_to_csv('data/raw/initial_dataset.csv')
            >>> print(f"Exported {summary['row_count']} rows to {summary['file_path']}")
        """
        try:
            # Load data
            df = self.load_data(table_name=table_name, query=query)

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export to CSV
            logger.info(f"Exporting {len(df)} rows to {output_path}")
            df.to_csv(output_path, index=False)

            # Get file size
            file_size_bytes = output_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)

            export_summary = {
                'file_path': str(output_path.absolute()),
                'row_count': len(df),
                'column_count': len(df.columns),
                'file_size_mb': round(file_size_mb, 2)
            }

            logger.info(f"Export completed successfully: {export_summary}")
            return export_summary

        except Exception as e:
            error_msg = f"Error exporting to CSV: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e

    def get_data_summary(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the banking marketing dataset.

        Args:
            table_name (Optional[str]): Name of the table to analyze.
                                      Defaults to 'bank_marketing'

        Returns:
            Dict[str, Any]: Data summary including basic statistics and info

        Raises:
            DataLoadError: If summary generation fails
        """
        try:
            # Get table info
            table_info = self.get_table_info(table_name)

            # Load a sample for analysis
            df_sample = self.load_data(table_name=table_name, limit=1000)

            summary = {
                'table_info': table_info,
                'data_types': df_sample.dtypes.to_dict(),
                'sample_shape': df_sample.shape,
                'memory_usage_mb': round(df_sample.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }

            logger.info(f"Generated data summary for table '{table_info['table_name']}'")
            return summary

        except Exception as e:
            error_msg = f"Error generating data summary: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e


def create_initial_dataset_snapshot() -> Dict[str, Any]:
    """
    Create an initial snapshot of the banking marketing dataset.

    This function loads the complete bank_marketing table from bmarket.db
    and exports it as a CSV file for backup and reference purposes.

    Data Source: data/raw/bmarket.db (bank_marketing table)
    Output: data/raw/initial_dataset.csv

    Returns:
        Dict[str, Any]: Summary of the snapshot creation process

    Raises:
        DataLoadError: If snapshot creation fails

    Business Value:
        - Provides baseline dataset for marketing campaign optimization
        - Creates backup copy for data integrity
        - Enables easy data sharing and analysis
    """
    try:
        logger.info("Creating initial dataset snapshot from bmarket.db")

        # Initialize data loader
        loader = BankingDataLoader()

        # Get data summary
        summary = loader.get_data_summary()

        # Export complete dataset
        export_summary = loader.export_to_csv('data/raw/initial_dataset.csv')

        # Combine summaries
        snapshot_summary = {
            'source_database': 'data/raw/bmarket.db',
            'source_table': 'bank_marketing',
            'output_file': export_summary['file_path'],
            'total_rows': export_summary['row_count'],
            'total_columns': export_summary['column_count'],
            'file_size_mb': export_summary['file_size_mb'],
            'data_types': summary['data_types'],
            'creation_status': 'success'
        }

        logger.info("Initial dataset snapshot created successfully")
        logger.info(f"Dataset: {snapshot_summary['total_rows']} rows, "
                   f"{snapshot_summary['total_columns']} columns, "
                   f"{snapshot_summary['file_size_mb']} MB")

        return snapshot_summary

    except Exception as e:
        error_msg = f"Failed to create initial dataset snapshot: {str(e)}"
        logger.error(error_msg)
        raise DataLoadError(error_msg) from e


if __name__ == "__main__":
    """
    Demo script showing basic usage of the BankingDataLoader.

    This script demonstrates:
    1. Database connection and validation
    2. Data loading with error handling
    3. CSV export functionality
    4. Initial dataset snapshot creation
    """
    try:
        print("Banking Data Loader Demo")
        print("=" * 50)

        # Create initial dataset snapshot
        snapshot_summary = create_initial_dataset_snapshot()

        print("\nSnapshot Summary:")
        for key, value in snapshot_summary.items():
            print(f"  {key}: {value}")

        print("\nDemo completed successfully!")

    except Exception as e:
        print(f"Demo failed: {str(e)}")
        logger.error(f"Demo execution failed: {str(e)}")
