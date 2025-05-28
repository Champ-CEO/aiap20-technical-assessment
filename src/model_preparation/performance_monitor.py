"""
Performance Monitor Module

Implements performance monitoring for >97K records/second processing standard.
Tracks processing times and validates performance across all model preparation operations.

Key Features:
- Processing speed monitoring (>97K records/second standard)
- Operation timing and performance validation
- Memory usage tracking
- Performance reporting and analysis
- Bottleneck identification
"""

import time
import logging

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitor for model preparation operations.

    Tracks processing times, memory usage, and validates performance
    against the >97K records/second standard.
    """

    def __init__(self, performance_standard: int = 97000):
        """
        Initialize performance monitor.

        Args:
            performance_standard (int): Required records per second
        """
        self.performance_standard = performance_standard

        # Performance tracking
        self.operation_metrics = {}
        self.session_metrics = {
            "start_time": time.time(),
            "operations_completed": 0,
            "total_records_processed": 0,
            "performance_violations": [],
        }

    def monitor_operation(self, operation_name: str, record_count: int):
        """
        Context manager for monitoring operation performance.

        Args:
            operation_name (str): Name of the operation being monitored
            record_count (int): Number of records being processed

        Returns:
            Context manager for performance monitoring
        """
        return self._operation_context(operation_name, record_count)

    @contextmanager
    def _operation_context(self, operation_name: str, record_count: int):
        """Context manager implementation for operation monitoring."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield

        finally:
            # Calculate performance metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()

            operation_time = end_time - start_time
            records_per_second = (
                record_count / operation_time if operation_time > 0 else float("inf")
            )
            memory_delta = end_memory - start_memory

            # Record metrics
            self.operation_metrics[operation_name] = {
                "operation_time": operation_time,
                "record_count": record_count,
                "records_per_second": records_per_second,
                "memory_usage_mb": end_memory,
                "memory_delta_mb": memory_delta,
                "meets_standard": records_per_second >= self.performance_standard,
                "timestamp": time.time(),
            }

            # Update session metrics
            self.session_metrics["operations_completed"] += 1
            self.session_metrics["total_records_processed"] += record_count

            # Check for performance violations
            if records_per_second < self.performance_standard:
                violation = {
                    "operation": operation_name,
                    "actual_rate": records_per_second,
                    "required_rate": self.performance_standard,
                    "deficit": self.performance_standard - records_per_second,
                    "timestamp": time.time(),
                }
                self.session_metrics["performance_violations"].append(violation)

                logger.warning(
                    f"Performance violation in {operation_name}: {records_per_second:,.0f} records/sec (required: {self.performance_standard:,})"
                )
            else:
                logger.info(
                    f"Performance OK for {operation_name}: {records_per_second:,.0f} records/sec"
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0

    def performance_decorator(self, operation_name: str):
        """
        Decorator for monitoring function performance.

        Args:
            operation_name (str): Name of the operation

        Returns:
            Decorator function
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Try to infer record count from arguments
                record_count = self._infer_record_count(args, kwargs)

                with self.monitor_operation(operation_name, record_count):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def _infer_record_count(self, args: tuple, kwargs: dict) -> int:
        """Infer record count from function arguments."""
        # Look for DataFrame in arguments
        for arg in args:
            if hasattr(arg, "__len__") and hasattr(
                arg, "columns"
            ):  # Likely a DataFrame
                return len(arg)

        # Look for DataFrame in keyword arguments
        for value in kwargs.values():
            if hasattr(value, "__len__") and hasattr(
                value, "columns"
            ):  # Likely a DataFrame
                return len(value)

        # Default to 1000 if can't infer
        return 1000

    def get_operation_metrics(
        self, operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for operations.

        Args:
            operation_name (str, optional): Specific operation to get metrics for

        Returns:
            Dict[str, Any]: Performance metrics
        """
        if operation_name:
            return self.operation_metrics.get(operation_name, {})
        else:
            return self.operation_metrics.copy()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session performance."""
        session_time = time.time() - self.session_metrics["start_time"]

        summary = {
            "session_duration": session_time,
            "operations_completed": self.session_metrics["operations_completed"],
            "total_records_processed": self.session_metrics["total_records_processed"],
            "average_records_per_second": (
                self.session_metrics["total_records_processed"] / session_time
                if session_time > 0
                else 0
            ),
            "performance_violations": len(
                self.session_metrics["performance_violations"]
            ),
            "performance_compliance_rate": (
                (
                    self.session_metrics["operations_completed"]
                    - len(self.session_metrics["performance_violations"])
                )
                / max(self.session_metrics["operations_completed"], 1)
            ),
        }

        return summary

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        session_summary = self.get_session_summary()

        # Analyze operation performance
        operation_analysis = {}
        if self.operation_metrics:
            rates = [
                metrics["records_per_second"]
                for metrics in self.operation_metrics.values()
            ]
            operation_analysis = {
                "fastest_operation": max(
                    self.operation_metrics.items(),
                    key=lambda x: x[1]["records_per_second"],
                ),
                "slowest_operation": min(
                    self.operation_metrics.items(),
                    key=lambda x: x[1]["records_per_second"],
                ),
                "average_rate": sum(rates) / len(rates),
                "operations_meeting_standard": sum(
                    1 for rate in rates if rate >= self.performance_standard
                ),
                "operations_below_standard": sum(
                    1 for rate in rates if rate < self.performance_standard
                ),
            }

        # Memory analysis
        memory_analysis = {}
        if self.operation_metrics:
            memory_usages = [
                metrics["memory_usage_mb"]
                for metrics in self.operation_metrics.values()
            ]
            memory_deltas = [
                metrics["memory_delta_mb"]
                for metrics in self.operation_metrics.values()
            ]

            memory_analysis = {
                "peak_memory_mb": max(memory_usages) if memory_usages else 0,
                "average_memory_mb": (
                    sum(memory_usages) / len(memory_usages) if memory_usages else 0
                ),
                "total_memory_delta_mb": sum(memory_deltas) if memory_deltas else 0,
                "largest_memory_increase": max(memory_deltas) if memory_deltas else 0,
            }

        report = {
            "performance_standard": self.performance_standard,
            "session_summary": session_summary,
            "operation_analysis": operation_analysis,
            "memory_analysis": memory_analysis,
            "performance_violations": self.session_metrics["performance_violations"],
            "detailed_metrics": self.operation_metrics,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Check overall compliance
        session_summary = self.get_session_summary()
        if session_summary["performance_compliance_rate"] < 0.8:
            recommendations.append(
                "Performance compliance is below 80%. Consider optimizing slow operations."
            )

        # Check for memory issues
        if self.operation_metrics:
            memory_deltas = [
                metrics["memory_delta_mb"]
                for metrics in self.operation_metrics.values()
            ]
            if any(delta > 500 for delta in memory_deltas):  # 500MB increase
                recommendations.append(
                    "Large memory increases detected. Consider memory optimization."
                )

        # Check for consistently slow operations
        slow_operations = [
            name
            for name, metrics in self.operation_metrics.items()
            if metrics["records_per_second"] < self.performance_standard * 0.5
        ]

        if slow_operations:
            recommendations.append(
                f"Operations consistently below 50% of standard: {', '.join(slow_operations)}"
            )

        # General recommendations
        if len(self.session_metrics["performance_violations"]) > 0:
            recommendations.append(
                "Consider using vectorized operations and optimizing data structures."
            )
            recommendations.append(
                "Review algorithm complexity and consider parallel processing."
            )

        if not recommendations:
            recommendations.append(
                "Performance is meeting standards. Continue current optimization practices."
            )

        return recommendations

    def validate_performance_standard(self) -> bool:
        """Check if overall session performance meets the standard."""
        session_summary = self.get_session_summary()
        return (
            session_summary["average_records_per_second"] >= self.performance_standard
        )

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.operation_metrics = {}
        self.session_metrics = {
            "start_time": time.time(),
            "operations_completed": 0,
            "total_records_processed": 0,
            "performance_violations": [],
        }

        logger.info("Performance metrics reset")


class ProcessingTimer:
    """Simple timer for measuring processing times."""

    def __init__(self):
        """Initialize processing timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None

    def stop(self):
        """Stop the timer."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        return self.elapsed_time

    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time."""
        if self.start_time is None:
            return None

        if self.end_time is None:
            return time.time() - self.start_time

        return self.elapsed_time

    def calculate_rate(self, record_count: int) -> Optional[float]:
        """Calculate processing rate in records per second."""
        elapsed = self.get_elapsed_time()
        if elapsed is None or elapsed == 0:
            return None

        return record_count / elapsed

    @contextmanager
    def time_operation(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


# Global performance monitor instance
_global_monitor = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def monitor_performance(operation_name: str):
    """Decorator for monitoring performance using global monitor."""
    monitor = get_global_monitor()
    return monitor.performance_decorator(operation_name)
