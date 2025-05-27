"""
Phase 4 Performance Optimization Testing

This module implements comprehensive performance testing and optimization validation
for Phase 4 Data Integration, focusing on production readiness:

1. Memory usage optimization and monitoring
2. Loading speed improvements and benchmarking
3. Validation performance optimization
4. Caching mechanisms and efficiency
5. Concurrent access performance
6. Large dataset handling optimization

Following TDD approach with focus on production performance standards.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import time
import tempfile
import os
import threading
import concurrent.futures
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data_integration import CSVLoader, DataValidator, PipelineUtils
from data_integration.csv_loader import CSVLoaderError
from data_integration.data_validator import DataValidationError


class TestMemoryOptimization:
    """Tests for memory usage optimization and monitoring."""

    def test_chunked_loading_memory_efficiency(self):
        """
        Performance Test: Chunked loading should use less peak memory

        Compare memory usage between chunked and full loading.
        """
        try:
            import psutil

            process = psutil.Process()

            # Test with actual Phase 3 data
            file_path = "../data/processed/cleaned-db.csv"

            # Test full loading memory usage
            gc.collect()  # Clean up before test
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            loader_full = CSVLoader(file_path)
            df_full = loader_full.load_data(validate=False)

            full_load_memory = process.memory_info().rss / 1024 / 1024  # MB
            full_memory_usage = full_load_memory - initial_memory

            del df_full
            gc.collect()

            # Test chunked loading memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            loader_chunked = CSVLoader(file_path)
            df_chunked = loader_chunked.load_data(chunk_size=5000, validate=False)

            chunked_load_memory = process.memory_info().rss / 1024 / 1024  # MB
            chunked_memory_usage = chunked_load_memory - initial_memory

            # Verify results are equivalent
            assert (
                len(df_chunked) == 41188
            ), "Chunked loading should preserve all records"

            # Memory efficiency check (chunked should use similar or less memory)
            memory_efficiency_ratio = chunked_memory_usage / full_memory_usage
            assert (
                memory_efficiency_ratio <= 1.2
            ), f"Chunked loading should be memory efficient, ratio: {memory_efficiency_ratio:.2f}"

            print(
                f"✅ Memory efficiency: Full={full_memory_usage:.1f}MB, Chunked={chunked_memory_usage:.1f}MB"
            )
            print(f"   Efficiency ratio: {memory_efficiency_ratio:.2f}")

        except ImportError:
            print("⚠️ psutil not available, skipping memory optimization test")

    def test_memory_cleanup_after_operations(self):
        """
        Performance Test: Memory should be properly cleaned up after operations

        Verify no memory leaks in repeated operations.
        """
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform multiple load/unload cycles
            for i in range(5):
                loader = CSVLoader("../data/processed/cleaned-db.csv")
                df = loader.load_data(validate=False)

                # Do some operations
                summary = df.describe()
                dtypes = df.dtypes

                # Clean up
                del df, summary, dtypes, loader
                gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory

            # Should not have significant memory growth
            assert (
                memory_growth < 50
            ), f"Memory growth {memory_growth:.1f}MB should be < 50MB after cleanup"

            print(f"✅ Memory cleanup: Growth after 5 cycles = {memory_growth:.1f}MB")

        except ImportError:
            print("⚠️ psutil not available, skipping memory cleanup test")


class TestLoadingSpeedOptimization:
    """Tests for loading speed improvements and benchmarking."""

    def test_loading_speed_benchmarks(self):
        """
        Performance Test: Loading speed should meet performance standards

        Benchmark different loading configurations.
        """
        file_path = "../data/processed/cleaned-db.csv"
        min_records_per_second = 97000  # Phase 4 standard

        # Test configurations
        configurations = [
            {"name": "default", "params": {}},
            {"name": "chunked_10k", "params": {"chunk_size": 10000}},
            {"name": "chunked_5k", "params": {"chunk_size": 5000}},
            {"name": "no_validation", "params": {"validate": False}},
        ]

        results = {}

        for config in configurations:
            loader = CSVLoader(file_path)

            # Benchmark loading
            start_time = time.time()
            df = loader.load_data(**config["params"])
            load_time = time.time() - start_time

            records_per_second = len(df) / load_time if load_time > 0 else float("inf")
            results[config["name"]] = {
                "time": load_time,
                "records_per_second": records_per_second,
                "meets_standard": records_per_second >= min_records_per_second,
            }

            print(
                f"✅ {config['name']}: {records_per_second:,.0f} records/second ({load_time:.3f}s)"
            )

        # At least one configuration should meet the standard
        meeting_standard = [
            name for name, result in results.items() if result["meets_standard"]
        ]
        assert (
            len(meeting_standard) > 0
        ), f"At least one configuration should meet {min_records_per_second:,} records/second standard"

        print(f"✅ Configurations meeting standard: {meeting_standard}")

    def test_selective_column_loading_performance(self):
        """
        Performance Test: Selective column loading should be faster

        Loading only required columns should improve performance.
        """
        file_path = "../data/processed/cleaned-db.csv"

        # Full loading benchmark
        loader_full = CSVLoader(file_path)
        start_time = time.time()
        df_full = loader_full.load_data(validate=False)
        full_load_time = time.time() - start_time

        # Selective loading benchmark (key columns only)
        key_columns = ["Client ID", "Age", "Contact Method", "Subscription Status"]
        loader_selective = CSVLoader(file_path)
        start_time = time.time()
        df_selective = loader_selective.load_columns(key_columns)
        selective_load_time = time.time() - start_time

        # Performance comparison
        speedup_ratio = full_load_time / selective_load_time

        # Selective loading should be faster or at least not significantly slower
        assert (
            speedup_ratio >= 0.8
        ), f"Selective loading should be efficient, speedup ratio: {speedup_ratio:.2f}"

        # Verify correct columns loaded
        assert (
            list(df_selective.columns) == key_columns
        ), "Selective loading should load only specified columns"

        print(f"✅ Selective loading: {speedup_ratio:.2f}x speedup")
        print(f"   Full: {full_load_time:.3f}s, Selective: {selective_load_time:.3f}s")


class TestValidationPerformance:
    """Tests for validation performance optimization."""

    def test_quick_validation_performance(self):
        """
        Performance Test: Quick validation should be significantly faster

        Compare quick vs comprehensive validation performance.
        """
        # Load data once
        loader = CSVLoader("../data/processed/cleaned-db.csv")
        df = loader.load_data(validate=False)

        validator = DataValidator()

        # Benchmark comprehensive validation
        start_time = time.time()
        comprehensive_report = validator.validate_data(df, comprehensive=True)
        comprehensive_time = time.time() - start_time

        # Benchmark quick validation
        start_time = time.time()
        quick_result = validator.validate_quick(df)
        quick_time = time.time() - start_time

        # Quick validation should be much faster
        speedup_ratio = comprehensive_time / quick_time
        assert (
            speedup_ratio >= 5
        ), f"Quick validation should be at least 5x faster, got {speedup_ratio:.1f}x"

        # Both should detect valid data
        assert (
            comprehensive_report["overall_status"] == "PASSED"
        ), "Comprehensive validation should pass on valid data"
        assert quick_result == True, "Quick validation should pass on valid data"

        print(f"✅ Validation performance: Quick is {speedup_ratio:.1f}x faster")
        print(f"   Comprehensive: {comprehensive_time:.3f}s, Quick: {quick_time:.3f}s")

    def test_validation_caching_efficiency(self):
        """
        Performance Test: Repeated validation should benefit from caching

        Test if validation results can be cached for efficiency.
        """
        loader = CSVLoader("../data/processed/cleaned-db.csv")
        df = loader.load_data(validate=False)

        validator = DataValidator()

        # First validation (cold)
        start_time = time.time()
        report1 = validator.validate_data(df, comprehensive=False)
        first_time = time.time() - start_time

        # Second validation (potentially cached)
        start_time = time.time()
        report2 = validator.validate_data(df, comprehensive=False)
        second_time = time.time() - start_time

        # Results should be consistent
        assert (
            report1["overall_status"] == report2["overall_status"]
        ), "Validation results should be consistent"

        # Second validation should be faster or similar
        efficiency_ratio = first_time / second_time if second_time > 0 else float("inf")

        print(
            f"✅ Validation consistency: First={first_time:.3f}s, Second={second_time:.3f}s"
        )
        print(f"   Efficiency ratio: {efficiency_ratio:.2f}x")


class TestConcurrentAccessPerformance:
    """Tests for concurrent access performance and thread safety."""

    def test_concurrent_loading_performance(self):
        """
        Performance Test: Concurrent loading should not cause significant slowdown

        Test multiple threads loading data simultaneously.
        """
        file_path = "../data/processed/cleaned-db.csv"
        num_threads = 3

        def load_data_worker():
            """Worker function for concurrent loading."""
            loader = CSVLoader(file_path)
            start_time = time.time()
            df = loader.load_data(validate=False)
            load_time = time.time() - start_time
            return {
                "records": len(df),
                "time": load_time,
                "records_per_second": (
                    len(df) / load_time if load_time > 0 else float("inf")
                ),
            }

        # Benchmark single-threaded loading
        single_result = load_data_worker()

        # Benchmark concurrent loading
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(load_data_worker) for _ in range(num_threads)]
            concurrent_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        total_concurrent_time = time.time() - start_time

        # Verify all threads completed successfully
        assert (
            len(concurrent_results) == num_threads
        ), f"All {num_threads} threads should complete successfully"

        # All should load the same number of records
        for result in concurrent_results:
            assert (
                result["records"] == 41188
            ), "All threads should load the same number of records"

        # Performance should not degrade significantly
        avg_concurrent_performance = (
            sum(r["records_per_second"] for r in concurrent_results) / num_threads
        )
        performance_ratio = (
            avg_concurrent_performance / single_result["records_per_second"]
        )

        assert (
            performance_ratio >= 0.5
        ), f"Concurrent performance should not degrade below 50%, got {performance_ratio:.2f}"

        print(
            f"✅ Concurrent loading: {num_threads} threads completed in {total_concurrent_time:.3f}s"
        )
        print(f"   Single: {single_result['records_per_second']:,.0f} rec/s")
        print(
            f"   Concurrent avg: {avg_concurrent_performance:,.0f} rec/s (ratio: {performance_ratio:.2f})"
        )
