"""
Performance Monitor Module

Implements performance monitoring for model evaluation operations.
Validates >97K records/second standard and tracks evaluation performance.

Key Features:
- Real-time performance monitoring
- >97K records/second standard validation
- Performance violation detection
- Evaluation speed optimization
- Performance reporting
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance standard
PERFORMANCE_STANDARD = 97000  # >97K records/second

# Try to import psutil for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


class PerformanceMonitor:
    """
    Performance monitor for model evaluation operations.

    Tracks processing times, memory usage, and validates performance
    against the >97K records/second standard.
    """

    def __init__(self, performance_standard: int = PERFORMANCE_STANDARD):
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
            "peak_memory_usage": 0,
        }

    @contextmanager
    def monitor_operation(self, operation_name: str, record_count: int = 0):
        """
        Context manager for monitoring operation performance.

        Args:
            operation_name (str): Name of the operation
            record_count (int): Number of records being processed
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            # Calculate metrics
            operation_time = end_time - start_time
            memory_delta = end_memory - start_memory
            records_per_second = (
                record_count / operation_time if operation_time > 0 else 0
            )

            # Store operation metrics
            self.operation_metrics[operation_name] = {
                "operation_time": operation_time,
                "record_count": record_count,
                "records_per_second": records_per_second,
                "memory_usage": {
                    "start": start_memory,
                    "end": end_memory,
                    "delta": memory_delta,
                },
                "timestamp": start_time,
                "meets_standard": records_per_second >= self.performance_standard,
            }

            # Update session metrics
            self.session_metrics["operations_completed"] += 1
            self.session_metrics["total_records_processed"] += record_count
            self.session_metrics["peak_memory_usage"] = max(
                self.session_metrics["peak_memory_usage"], end_memory
            )

            # Check for performance violations
            if record_count > 0 and records_per_second < self.performance_standard:
                violation = {
                    "operation": operation_name,
                    "actual_rate": records_per_second,
                    "required_rate": self.performance_standard,
                    "deficit": self.performance_standard - records_per_second,
                    "timestamp": start_time,
                    "record_count": record_count,
                }
                self.session_metrics["performance_violations"].append(violation)

                logger.warning(
                    f"Performance violation in {operation_name}: "
                    f"{records_per_second:,.0f} records/sec "
                    f"(required: {self.performance_standard:,})"
                )
            elif record_count > 0:
                logger.info(
                    f"Performance OK for {operation_name}: "
                    f"{records_per_second:,.0f} records/sec"
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

    def monitor_evaluation_pipeline(self, pipeline_func, *args, **kwargs):
        """
        Monitor the complete evaluation pipeline performance.

        Args:
            pipeline_func: Function to monitor
            *args, **kwargs: Arguments for the function

        Returns:
            Tuple: (function_result, performance_metrics)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            # Execute the pipeline
            result = pipeline_func(*args, **kwargs)

            end_time = time.time()
            end_memory = self._get_memory_usage()

            # Calculate pipeline metrics
            total_time = end_time - start_time
            total_records = self.session_metrics["total_records_processed"]
            overall_rate = total_records / total_time if total_time > 0 else 0

            pipeline_metrics = {
                "total_execution_time": total_time,
                "total_records_processed": total_records,
                "overall_records_per_second": overall_rate,
                "meets_performance_standard": overall_rate >= self.performance_standard,
                "memory_usage": {
                    "start": start_memory,
                    "end": end_memory,
                    "peak": self.session_metrics["peak_memory_usage"],
                    "delta": end_memory - start_memory,
                },
                "operations_completed": self.session_metrics["operations_completed"],
                "performance_violations": len(
                    self.session_metrics["performance_violations"]
                ),
            }

            logger.info(
                f"Pipeline completed: {total_time:.2f}s, "
                f"{overall_rate:,.0f} records/sec, "
                f"{len(self.session_metrics['performance_violations'])} violations"
            )

            return result, pipeline_metrics

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        session_duration = time.time() - self.session_metrics["start_time"]
        total_records = self.session_metrics["total_records_processed"]
        overall_rate = total_records / session_duration if session_duration > 0 else 0

        # Analyze operation performance
        operation_analysis = {}
        for op_name, metrics in self.operation_metrics.items():
            operation_analysis[op_name] = {
                "records_per_second": metrics["records_per_second"],
                "meets_standard": metrics["meets_standard"],
                "operation_time": metrics["operation_time"],
                "record_count": metrics["record_count"],
                "efficiency_ratio": metrics["records_per_second"]
                / self.performance_standard,
            }

        # Performance summary
        operations_meeting_standard = sum(
            1
            for metrics in self.operation_metrics.values()
            if metrics["meets_standard"]
        )
        total_operations = len(self.operation_metrics)

        performance_report = {
            "session_summary": {
                "session_duration": session_duration,
                "total_records_processed": total_records,
                "overall_records_per_second": overall_rate,
                "meets_overall_standard": overall_rate >= self.performance_standard,
                "operations_completed": self.session_metrics["operations_completed"],
                "peak_memory_usage_mb": self.session_metrics["peak_memory_usage"],
            },
            "operation_analysis": operation_analysis,
            "performance_compliance": {
                "operations_meeting_standard": operations_meeting_standard,
                "total_operations": total_operations,
                "compliance_rate": operations_meeting_standard
                / max(total_operations, 1),
                "performance_violations": len(
                    self.session_metrics["performance_violations"]
                ),
            },
            "performance_violations": self.session_metrics["performance_violations"],
            "recommendations": self._generate_performance_recommendations(),
        }

        return performance_report

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""

        recommendations = []

        # Check overall performance
        session_duration = time.time() - self.session_metrics["start_time"]
        total_records = self.session_metrics["total_records_processed"]
        overall_rate = total_records / session_duration if session_duration > 0 else 0

        if overall_rate < self.performance_standard:
            recommendations.append(
                f"Overall performance ({overall_rate:,.0f} records/sec) below standard "
                f"({self.performance_standard:,} records/sec)"
            )

        # Check for slow operations
        slow_operations = [
            op_name
            for op_name, metrics in self.operation_metrics.items()
            if not metrics["meets_standard"] and metrics["record_count"] > 0
        ]

        if slow_operations:
            recommendations.append(
                f"Optimize slow operations: {', '.join(slow_operations)}"
            )

        # Memory usage recommendations
        if self.session_metrics["peak_memory_usage"] > 1000:  # > 1GB
            recommendations.append(
                f"High memory usage detected ({self.session_metrics['peak_memory_usage']:.0f} MB) - "
                "consider memory optimization"
            )

        # Performance violations
        if self.session_metrics["performance_violations"]:
            recommendations.append(
                f"{len(self.session_metrics['performance_violations'])} performance violations detected - "
                "review operation efficiency"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Performance meets all standards - no optimization needed"
            )
        else:
            recommendations.extend(
                [
                    "Consider batch processing for large datasets",
                    "Implement parallel processing where possible",
                    "Profile memory usage and optimize data structures",
                    "Cache frequently accessed data",
                ]
            )

        return recommendations

    def validate_performance_standard(self, operation_name: str = None) -> bool:
        """
        Validate if performance meets the standard.

        Args:
            operation_name (str, optional): Specific operation to check

        Returns:
            bool: True if performance meets standard
        """
        if operation_name:
            if operation_name in self.operation_metrics:
                return self.operation_metrics[operation_name]["meets_standard"]
            return False
        else:
            # Check overall performance
            session_duration = time.time() - self.session_metrics["start_time"]
            total_records = self.session_metrics["total_records_processed"]
            overall_rate = (
                total_records / session_duration if session_duration > 0 else 0
            )
            return overall_rate >= self.performance_standard

    def get_operation_metrics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific operation."""
        return self.operation_metrics.get(operation_name)

    def reset_session(self):
        """Reset session metrics for a new evaluation session."""
        self.operation_metrics.clear()
        self.session_metrics = {
            "start_time": time.time(),
            "operations_completed": 0,
            "total_records_processed": 0,
            "performance_violations": [],
            "peak_memory_usage": 0,
        }
        logger.info("Performance monitoring session reset")

    def log_performance_summary(self):
        """Log a summary of performance metrics."""

        session_duration = time.time() - self.session_metrics["start_time"]
        total_records = self.session_metrics["total_records_processed"]
        overall_rate = total_records / session_duration if session_duration > 0 else 0

        logger.info("=" * 60)
        logger.info("PERFORMANCE MONITORING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Session Duration: {session_duration:.2f} seconds")
        logger.info(f"Total Records Processed: {total_records:,}")
        logger.info(f"Overall Rate: {overall_rate:,.0f} records/second")
        logger.info(
            f"Performance Standard: {self.performance_standard:,} records/second"
        )
        logger.info(
            f"Standard Met: {'✅ YES' if overall_rate >= self.performance_standard else '❌ NO'}"
        )
        logger.info(
            f"Operations Completed: {self.session_metrics['operations_completed']}"
        )
        logger.info(
            f"Performance Violations: {len(self.session_metrics['performance_violations'])}"
        )
        logger.info(
            f"Peak Memory Usage: {self.session_metrics['peak_memory_usage']:.1f} MB"
        )
        logger.info("=" * 60)
