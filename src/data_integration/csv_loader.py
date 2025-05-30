"""
CSV Loader for Phase 4 Data Integration

This module provides efficient CSV-based data loading with validation
for the banking marketing dataset. Optimized for 41K records with
performance target of >97K records/second.

Key Features:
- Direct CSV operations for optimal performance
- Built-in data validation and integrity checks
- Memory-efficient loading strategies
- Comprehensive error handling
- Performance monitoring and reporting
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os

# Optional psutil import for performance monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVLoaderError(Exception):
    """Custom exception for CSV Loader errors."""

    pass


class CSVLoader:
    """
    Efficient CSV-based data loader for Phase 4 data integration.

    Optimized for loading the cleaned banking marketing dataset from Phase 3
    with comprehensive validation and performance monitoring.
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the CSV Loader.

        Args:
            file_path (Optional[str]): Path to the CSV file.
                                     Defaults to Phase 3 output path.

        Raises:
            CSVLoaderError: If the file doesn't exist or is not accessible
        """
        if file_path is None:
            # Default path relative to project root - handle different working directories
            possible_paths = [
                Path("data/processed/cleaned-db.csv"),  # From project root
                Path("../data/processed/cleaned-db.csv"),  # From tests directory
                Path(
                    "../../data/processed/cleaned-db.csv"
                ),  # From deeper test directories
            ]

            self.file_path = None
            for path in possible_paths:
                if path.exists():
                    self.file_path = path
                    break

            if self.file_path is None:
                # Try to find project root and construct absolute path
                current_dir = Path.cwd()
                while current_dir.parent != current_dir:  # Not at filesystem root
                    potential_file = (
                        current_dir / "data" / "processed" / "cleaned-db.csv"
                    )
                    if potential_file.exists():
                        self.file_path = potential_file
                        break
                    current_dir = current_dir.parent

                if self.file_path is None:
                    raise CSVLoaderError(
                        f"CSV file not found in any expected location. Searched: {[str(p.absolute()) for p in possible_paths]}"
                    )
        else:
            self.file_path = Path(file_path)

        # Validate file existence
        if not self.file_path.exists():
            raise CSVLoaderError(f"CSV file not found: {self.file_path.absolute()}")

        # Initialize performance tracking
        self.performance_metrics = {
            "load_time": 0.0,
            "records_per_second": 0.0,
            "memory_usage_mb": 0.0,
            "file_size_mb": 0.0,
        }

        # Get file size
        self.performance_metrics["file_size_mb"] = (
            self.file_path.stat().st_size / 1024 / 1024
        )

        logger.info(f"Initialized CSVLoader for file: {self.file_path}")
        logger.info(f"File size: {self.performance_metrics['file_size_mb']:.2f} MB")

    def load_data(
        self,
        validate: bool = True,
        chunk_size: Optional[int] = None,
        columns: Optional[list] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load data from CSV file with optional validation and optimization.

        Args:
            validate (bool): Whether to perform basic validation
            chunk_size (Optional[int]): Load data in chunks for memory efficiency
            columns (Optional[list]): Specific columns to load
            nrows (Optional[int]): Number of rows to load (for sampling)

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            CSVLoaderError: If loading fails or validation errors occur
        """
        try:
            # Start performance monitoring
            start_time = time.time()

            # Initialize memory tracking if psutil is available
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            else:
                initial_memory = 0

            # Load data based on parameters
            if chunk_size is not None:
                df = self._load_chunked_data(chunk_size, columns, nrows)
            else:
                df = self._load_full_data(columns, nrows)

            # Calculate performance metrics
            load_time = time.time() - start_time

            if PSUTIL_AVAILABLE:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
            else:
                memory_used = 0  # Fallback when psutil not available

            self.performance_metrics.update(
                {
                    "load_time": load_time,
                    "records_per_second": (
                        len(df) / load_time if load_time > 0 else float("inf")
                    ),
                    "memory_usage_mb": memory_used,
                }
            )

            # Perform validation if requested
            if validate:
                self._validate_loaded_data(df)

            # Log performance results
            self._log_performance_metrics(df)

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV data: {str(e)}")
            raise CSVLoaderError(f"Data loading failed: {str(e)}")

    def _load_full_data(
        self, columns: Optional[list], nrows: Optional[int]
    ) -> pd.DataFrame:
        """Load full dataset in one operation."""
        kwargs = {}
        if columns is not None:
            kwargs["usecols"] = columns
        if nrows is not None:
            kwargs["nrows"] = nrows

        logger.info(f"Loading data with parameters: {kwargs}")
        return pd.read_csv(self.file_path, **kwargs)

    def _load_chunked_data(
        self, chunk_size: int, columns: Optional[list], nrows: Optional[int]
    ) -> pd.DataFrame:
        """Load data in chunks for memory efficiency."""
        kwargs = {"chunksize": chunk_size}
        if columns is not None:
            kwargs["usecols"] = columns

        chunks = []
        total_rows = 0

        logger.info(f"Loading data in chunks of {chunk_size}")

        for chunk in pd.read_csv(self.file_path, **kwargs):
            chunks.append(chunk)
            total_rows += len(chunk)

            # Stop if we've reached the desired number of rows
            if nrows is not None and total_rows >= nrows:
                # Trim the last chunk if necessary
                if total_rows > nrows:
                    excess_rows = total_rows - nrows
                    chunks[-1] = chunks[-1].iloc[:-excess_rows]
                break

        return pd.concat(chunks, ignore_index=True)

    def _validate_loaded_data(self, df: pd.DataFrame) -> None:
        """Perform basic validation on loaded data."""
        # Check if DataFrame is empty
        if df.empty:
            raise CSVLoaderError("Loaded DataFrame is empty")

        # Check for expected structure (if loading full dataset)
        if len(df) > 40000:  # Likely full dataset
            expected_records = 41188
            expected_features = 33

            if len(df) != expected_records:
                logger.warning(
                    f"Record count mismatch: expected {expected_records}, got {len(df)}"
                )

            if len(df.columns) != expected_features:
                logger.warning(
                    f"Feature count mismatch: expected {expected_features}, got {len(df.columns)}"
                )

        # Check for basic data integrity
        if df.isnull().all().any():
            raise CSVLoaderError("Found columns with all missing values")

        logger.info("Basic data validation passed")

    def _log_performance_metrics(self, df: pd.DataFrame) -> None:
        """Log performance metrics."""
        metrics = self.performance_metrics

        logger.info("=== CSV Loader Performance Metrics ===")
        logger.info(f"Records loaded: {len(df):,}")
        logger.info(f"Features loaded: {len(df.columns)}")
        logger.info(f"Load time: {metrics['load_time']:.3f} seconds")
        logger.info(f"Performance: {metrics['records_per_second']:,.0f} records/second")
        logger.info(f"Memory usage: {metrics['memory_usage_mb']:.2f} MB")
        logger.info(f"File size: {metrics['file_size_mb']:.2f} MB")

        # Check if performance meets standards
        performance_standard = 97000  # records per second
        if metrics["records_per_second"] >= performance_standard:
            logger.info(
                f"✅ Performance standard met: {metrics['records_per_second']:,.0f} >= {performance_standard:,} records/second"
            )
        else:
            logger.warning(
                f"⚠️ Performance below standard: {metrics['records_per_second']:,.0f} < {performance_standard:,} records/second"
            )

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics from the last load operation.

        Returns:
            Dict[str, float]: Performance metrics including load time,
                            records per second, and memory usage
        """
        return self.performance_metrics.copy()

    def load_sample(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Load a sample of the data for testing or exploration.

        Args:
            n_rows (int): Number of rows to load

        Returns:
            pd.DataFrame: Sample data
        """
        logger.info(f"Loading sample of {n_rows} rows")
        return self.load_data(nrows=n_rows)

    def load_columns(self, columns: list) -> pd.DataFrame:
        """
        Load specific columns only.

        Args:
            columns (list): List of column names to load

        Returns:
            pd.DataFrame: Data with specified columns only
        """
        logger.info(f"Loading specific columns: {columns}")
        return self.load_data(columns=columns)

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the CSV file.

        Returns:
            Dict[str, Any]: File information including path, size, and modification time
        """
        stat = self.file_path.stat()
        return {
            "file_path": str(self.file_path.absolute()),
            "file_size_bytes": stat.st_size,
            "file_size_mb": stat.st_size / 1024 / 1024,
            "last_modified": stat.st_mtime,
            "exists": self.file_path.exists(),
            "is_readable": os.access(self.file_path, os.R_OK),
        }
