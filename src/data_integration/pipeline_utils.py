"""
Pipeline Utilities for Phase 4 Data Integration

This module provides pipeline integration utilities for ML workflow,
including data splitting, memory optimization, error handling,
and performance monitoring.

Key Features:
- Stratified data splitting to preserve target distribution
- Memory-efficient data processing
- Comprehensive error handling and recovery
- Performance monitoring and optimization
- Pipeline integration support for Phase 4 → Phase 5 flow
"""

import pandas as pd
import numpy as np
import logging
import time
import gc
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import warnings

# Optional psutil import for performance monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineUtilsError(Exception):
    """Custom exception for Pipeline Utilities errors."""

    pass


class PipelineUtils:
    """
    Pipeline integration utilities for Phase 4 data integration.

    Provides comprehensive utilities for data splitting, memory optimization,
    error handling, and performance monitoring to support seamless
    Phase 4 → Phase 5 data flow.
    """

    def __init__(self):
        """Initialize Pipeline Utilities with default configuration."""
        self.performance_metrics = {
            "operation_times": {},
            "memory_usage": {},
            "data_splits": {},
            "error_counts": {},
        }

        # Default configuration
        self.config = {
            "random_state": 42,
            "test_size": 0.2,
            "validation_size": 0.2,
            "stratify_target": True,
            "memory_threshold_mb": 1000,
            "performance_standard": 97000,  # records per second
        }

        logger.info("Initialized PipelineUtils with default configuration")

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str = "Subscription Status",
        test_size: float = None,
        validation_size: float = None,
        stratify: bool = None,
        random_state: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets with stratification.

        Args:
            df (pd.DataFrame): Input data to split
            target_column (str): Name of target column for stratification
            test_size (float): Proportion for test set
            validation_size (float): Proportion for validation set
            stratify (bool): Whether to stratify splits
            random_state (int): Random state for reproducibility

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with train, validation, test splits

        Raises:
            PipelineUtilsError: If splitting fails
        """
        try:
            start_time = time.time()

            # Use defaults if not provided
            test_size = test_size or self.config["test_size"]
            validation_size = validation_size or self.config["validation_size"]
            stratify = (
                stratify if stratify is not None else self.config["stratify_target"]
            )
            random_state = random_state or self.config["random_state"]

            # Validate inputs
            if target_column not in df.columns:
                raise PipelineUtilsError(
                    f"Target column '{target_column}' not found in data"
                )

            if test_size + validation_size >= 1.0:
                raise PipelineUtilsError(
                    "Combined test_size and validation_size must be < 1.0"
                )

            # Prepare stratification
            stratify_column = df[target_column] if stratify else None

            # First split: separate test set
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=stratify_column,
                random_state=random_state,
            )

            # Second split: separate train and validation from remaining data
            if validation_size > 0:
                # Adjust validation size for remaining data
                val_size_adjusted = validation_size / (1 - test_size)
                stratify_temp = y_temp if stratify else None

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_size_adjusted,
                    stratify=stratify_temp,
                    random_state=random_state,
                )

                # Combine features and target
                train_df = pd.concat([X_train, y_train], axis=1)
                val_df = pd.concat([X_val, y_val], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

                splits = {"train": train_df, "validation": val_df, "test": test_df}
            else:
                # Only train/test split
                train_df = pd.concat([X_temp, y_temp], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

                splits = {"train": train_df, "test": test_df}

            # Record performance metrics
            split_time = time.time() - start_time
            self.performance_metrics["operation_times"]["data_splitting"] = split_time

            # Record split information
            split_info = {
                "total_records": len(df),
                "train_records": len(splits["train"]),
                "test_records": len(splits["test"]),
                "train_ratio": len(splits["train"]) / len(df),
                "test_ratio": len(splits["test"]) / len(df),
            }

            if "validation" in splits:
                split_info.update(
                    {
                        "validation_records": len(splits["validation"]),
                        "validation_ratio": len(splits["validation"]) / len(df),
                    }
                )

            self.performance_metrics["data_splits"] = split_info

            # Validate stratification if used
            if stratify:
                self._validate_stratification(df, splits, target_column)

            # Log split summary
            self._log_split_summary(splits, split_info, split_time)

            return splits

        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise PipelineUtilsError(f"Data splitting failed: {str(e)}")

    def optimize_memory_usage(
        self, df: pd.DataFrame, aggressive: bool = False
    ) -> pd.DataFrame:
        """
        Optimize memory usage of DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame
            aggressive (bool): Whether to use aggressive optimization

        Returns:
            pd.DataFrame: Memory-optimized DataFrame
        """
        try:
            start_time = time.time()
            initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB

            logger.info(
                f"Starting memory optimization. Initial memory: {initial_memory:.2f} MB"
            )

            # Create a copy to avoid modifying original
            df_optimized = df.copy()

            # Optimize integer columns
            for col in df_optimized.select_dtypes(include=["int64"]).columns:
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()

                if col_min >= 0:  # Unsigned integers
                    if col_max < 255:
                        df_optimized[col] = df_optimized[col].astype("uint8")
                    elif col_max < 65535:
                        df_optimized[col] = df_optimized[col].astype("uint16")
                    elif col_max < 4294967295:
                        df_optimized[col] = df_optimized[col].astype("uint32")
                else:  # Signed integers
                    if col_min > -128 and col_max < 127:
                        df_optimized[col] = df_optimized[col].astype("int8")
                    elif col_min > -32768 and col_max < 32767:
                        df_optimized[col] = df_optimized[col].astype("int16")
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df_optimized[col] = df_optimized[col].astype("int32")

            # Optimize float columns
            for col in df_optimized.select_dtypes(include=["float64"]).columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

            # Optimize object columns (categorical)
            if aggressive:
                for col in df_optimized.select_dtypes(include=["object"]).columns:
                    num_unique_values = df_optimized[col].nunique()
                    num_total_values = len(df_optimized[col])

                    # Convert to category if it saves memory
                    if num_unique_values / num_total_values < 0.5:
                        df_optimized[col] = df_optimized[col].astype("category")

            # Calculate memory savings
            final_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
            memory_saved = initial_memory - final_memory
            memory_reduction_percent = (memory_saved / initial_memory) * 100

            optimization_time = time.time() - start_time

            # Record performance metrics
            self.performance_metrics["memory_usage"]["optimization"] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_saved_mb": memory_saved,
                "reduction_percent": memory_reduction_percent,
                "optimization_time": optimization_time,
            }

            logger.info(f"Memory optimization completed in {optimization_time:.3f}s")
            logger.info(
                f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB"
            )
            logger.info(
                f"Memory savings: {memory_saved:.2f} MB ({memory_reduction_percent:.1f}%)"
            )

            return df_optimized

        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
            raise PipelineUtilsError(f"Memory optimization failed: {str(e)}")

    def monitor_performance(self, operation_name: str, func, *args, **kwargs):
        """
        Monitor performance of an operation.

        Args:
            operation_name (str): Name of the operation
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Any: Result of the function execution
        """
        try:
            # Get initial metrics
            start_time = time.time()

            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024**2  # MB
            else:
                initial_memory = 0  # Fallback when psutil not available

            # Execute operation
            result = func(*args, **kwargs)

            # Calculate metrics
            execution_time = time.time() - start_time

            if PSUTIL_AVAILABLE:
                final_memory = process.memory_info().rss / 1024**2  # MB
                memory_used = final_memory - initial_memory
            else:
                memory_used = 0  # Fallback when psutil not available

            # Record metrics
            self.performance_metrics["operation_times"][operation_name] = execution_time
            self.performance_metrics["memory_usage"][operation_name] = memory_used

            # Log performance
            logger.info(
                f"Operation '{operation_name}' completed in {execution_time:.3f}s"
            )
            logger.info(f"Memory usage: {memory_used:.2f} MB")

            # Check against performance standards
            if hasattr(result, "__len__") and len(result) > 1000:
                records_per_second = (
                    len(result) / execution_time if execution_time > 0 else float("inf")
                )
                if records_per_second >= self.config["performance_standard"]:
                    logger.info(
                        f"✅ Performance standard met: {records_per_second:,.0f} records/second"
                    )
                else:
                    logger.warning(
                        f"⚠️ Performance below standard: {records_per_second:,.0f} records/second"
                    )

            return result

        except Exception as e:
            self.performance_metrics["error_counts"][operation_name] = (
                self.performance_metrics["error_counts"].get(operation_name, 0) + 1
            )
            logger.error(
                f"Performance monitoring failed for '{operation_name}': {str(e)}"
            )
            raise

    def handle_data_access_error(
        self, error: Exception, file_path: str, retry_count: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Handle data access errors with retry logic.

        Args:
            error (Exception): The original error
            file_path (str): Path to the data file
            retry_count (int): Number of retry attempts

        Returns:
            Optional[pd.DataFrame]: Data if recovery successful, None otherwise
        """
        logger.warning(f"Data access error occurred: {str(error)}")
        logger.info(f"Attempting recovery with {retry_count} retries...")

        for attempt in range(retry_count):
            try:
                logger.info(f"Retry attempt {attempt + 1}/{retry_count}")

                # Wait before retry
                time.sleep(0.5 * (attempt + 1))

                # Try different loading strategies
                if attempt == 0:
                    # Try with error handling
                    df = pd.read_csv(file_path, on_bad_lines="skip")
                elif attempt == 1:
                    # Try with different encoding
                    df = pd.read_csv(file_path, encoding="latin-1")
                else:
                    # Try with minimal columns
                    df = pd.read_csv(file_path, nrows=1000)

                logger.info(f"Recovery successful on attempt {attempt + 1}")
                return df

            except Exception as retry_error:
                logger.warning(
                    f"Retry attempt {attempt + 1} failed: {str(retry_error)}"
                )
                continue

        logger.error(f"All retry attempts failed for file: {file_path}")
        return None

    def _validate_stratification(
        self,
        original_df: pd.DataFrame,
        splits: Dict[str, pd.DataFrame],
        target_column: str,
    ) -> None:
        """Validate that stratification preserved target distribution."""
        original_dist = (
            original_df[target_column].value_counts(normalize=True).sort_index()
        )

        for split_name, split_df in splits.items():
            split_dist = (
                split_df[target_column].value_counts(normalize=True).sort_index()
            )

            # Check if distributions are similar (within 5% tolerance)
            max_diff = abs(original_dist - split_dist).max()
            if max_diff > 0.05:
                logger.warning(
                    f"Stratification may not be perfect for {split_name} split. Max difference: {max_diff:.3f}"
                )
            else:
                logger.info(f"✅ Stratification validated for {split_name} split")

    def _log_split_summary(
        self,
        splits: Dict[str, pd.DataFrame],
        split_info: Dict[str, Any],
        split_time: float,
    ) -> None:
        """Log data splitting summary."""
        logger.info("=== Data Splitting Summary ===")
        logger.info(f"Total records: {split_info['total_records']:,}")
        logger.info(f"Splitting time: {split_time:.3f} seconds")

        for split_name, split_df in splits.items():
            ratio = split_info.get(
                f"{split_name}_ratio", len(split_df) / split_info["total_records"]
            )
            logger.info(
                f"{split_name.capitalize()} set: {len(split_df):,} records ({ratio:.1%})"
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Dict[str, Any]: Performance metrics and statistics
        """
        return {
            "operation_times": self.performance_metrics["operation_times"].copy(),
            "memory_usage": self.performance_metrics["memory_usage"].copy(),
            "data_splits": self.performance_metrics["data_splits"].copy(),
            "error_counts": self.performance_metrics["error_counts"].copy(),
            "total_operations": len(self.performance_metrics["operation_times"]),
            "total_errors": sum(self.performance_metrics["error_counts"].values()),
        }

    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self.performance_metrics = {
            "operation_times": {},
            "memory_usage": {},
            "data_splits": {},
            "error_counts": {},
        }
        logger.info("Performance metrics cleared")

    def configure(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                logger.info(f"Configuration updated: {key} = {value} (was {old_value})")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Dict[str, Any]: Current configuration parameters
        """
        return self.config.copy()
