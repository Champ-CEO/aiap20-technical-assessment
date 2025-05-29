#!/usr/bin/env python3
"""
Phase 8 Step 1 Test Runner - Model Evaluation TDD Implementation

Automated test runner for Phase 8 Model Evaluation Step 1 tests following TDD approach.
Executes both smoke tests and critical tests to validate requirements definition.

Test Categories:
- SMOKE TESTS (6): Core model evaluation requirements
- CRITICAL TESTS (6): Business-focused evaluation requirements

Expected Outcome: 12 failing tests (TDD red phase) to guide Step 2 implementation.
Based on Phase 7 completion: 5 production-ready models with comprehensive performance analysis.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_pytest_tests(test_file, test_class=None, verbose=True):
    """Run pytest tests and capture results."""
    cmd = ["python", "-m", "pytest", str(test_file)]

    if test_class:
        cmd.append(f"::{test_class}")

    if verbose:
        cmd.extend(["-v", "-s"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300,  # 5 minute timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Test execution timed out",
            "success": False,
        }
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False}


def analyze_test_results(results):
    """Analyze test results and extract key metrics."""
    stdout = results.get("stdout", "")
    stderr = results.get("stderr", "")

    # Extract test counts from pytest output
    passed_count = stdout.count("PASSED")
    failed_count = stdout.count("FAILED")
    error_count = stdout.count("ERROR")
    skipped_count = stdout.count("SKIPPED")

    # Look for pytest summary line (e.g., "12 passed in 2.68s")
    import re

    summary_match = re.search(r"(\d+) passed", stdout)
    if summary_match:
        passed_count = int(summary_match.group(1))

    failed_match = re.search(r"(\d+) failed", stdout)
    if failed_match:
        failed_count = int(failed_match.group(1))

    error_match = re.search(r"(\d+) error", stdout)
    if error_match:
        error_count = int(error_match.group(1))

    # Look for specific test patterns
    smoke_tests = 0
    critical_tests = 0

    smoke_test_patterns = [
        "test_phase7_integration_smoke_test",
        "test_performance_metrics_smoke_test",
        "test_model_comparison_smoke_test",
        "test_visualization_smoke_test",
        "test_report_generation_smoke_test",
        "test_pipeline_integration_smoke_test",
    ]

    critical_test_patterns = [
        "test_production_deployment_validation_critical",
        "test_performance_monitoring_critical",
        "test_business_metrics_validation_critical",
        "test_feature_importance_validation_critical",
        "test_speed_performance_critical",
        "test_ensemble_evaluation_critical",
    ]

    for pattern in smoke_test_patterns:
        if pattern in stdout:
            smoke_tests += 1

    for pattern in critical_test_patterns:
        if pattern in stdout:
            critical_tests += 1

    total_tests = passed_count + failed_count + error_count

    return {
        "total_tests": total_tests,
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "skipped": skipped_count,
        "smoke_tests": smoke_tests,
        "critical_tests": critical_tests,
        "success_rate": passed_count / max(1, total_tests) if total_tests > 0 else 0,
        "tdd_phase": (
            "red"
            if failed_count > passed_count
            else "green" if passed_count > 0 else "unknown"
        ),
    }


def main():
    """Main test runner function."""
    print("=" * 80)
    print("PHASE 8 STEP 1: MODEL EVALUATION TDD TEST RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    print()

    # Test file path
    test_file = project_root / "tests" / "unit" / "test_model_evaluation.py"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1

    print(f"📁 Test file: {test_file}")
    print()

    # Run smoke tests
    print("🔥 RUNNING SMOKE TESTS (6 tests)")
    print("-" * 50)
    smoke_start = time.time()
    smoke_results = run_pytest_tests(test_file, "TestPhase8ModelEvaluationSmoke")
    smoke_duration = time.time() - smoke_start
    smoke_analysis = analyze_test_results(smoke_results)

    print(f"Smoke tests completed in {smoke_duration:.2f}s")
    print(
        f"Results: {smoke_analysis['passed']} passed, {smoke_analysis['failed']} failed, {smoke_analysis['errors']} errors"
    )
    print()

    # Run critical tests
    print("⚡ RUNNING CRITICAL TESTS (6 tests)")
    print("-" * 50)
    critical_start = time.time()
    critical_results = run_pytest_tests(test_file, "TestPhase8ModelEvaluationCritical")
    critical_duration = time.time() - critical_start
    critical_analysis = analyze_test_results(critical_results)

    print(f"Critical tests completed in {critical_duration:.2f}s")
    print(
        f"Results: {critical_analysis['passed']} passed, {critical_analysis['failed']} failed, {critical_analysis['errors']} errors"
    )
    print()

    # Combined analysis
    total_duration = smoke_duration + critical_duration
    total_tests = smoke_analysis["total_tests"] + critical_analysis["total_tests"]
    total_passed = smoke_analysis["passed"] + critical_analysis["passed"]
    total_failed = smoke_analysis["failed"] + critical_analysis["failed"]
    total_errors = smoke_analysis["errors"] + critical_analysis["errors"]

    # TDD Phase Analysis
    print("📊 TDD PHASE ANALYSIS")
    print("-" * 50)

    if total_failed > total_passed:
        tdd_phase = "🔴 RED PHASE"
        tdd_status = "EXPECTED - Ready for Step 2 Implementation"
    elif total_passed > total_failed:
        tdd_phase = "🟢 GREEN PHASE"
        tdd_status = "UNEXPECTED - Implementation may already exist"
    else:
        tdd_phase = "🟡 MIXED PHASE"
        tdd_status = "PARTIAL - Some implementation exists"

    print(f"TDD Phase: {tdd_phase}")
    print(f"Status: {tdd_status}")
    print()

    # Summary Report
    print("📋 SUMMARY REPORT")
    print("-" * 50)
    print(f"Total execution time: {total_duration:.2f}s")
    print(f"Total tests executed: {total_tests}")
    print(f"  • Smoke tests: {smoke_analysis['smoke_tests']}/6")
    print(f"  • Critical tests: {critical_analysis['critical_tests']}/6")
    print()
    print(f"Test Results:")
    print(f"  • ✅ Passed: {total_passed}")
    print(f"  • ❌ Failed: {total_failed}")
    print(f"  • 🚫 Errors: {total_errors}")
    print(f"  • ⏭️  Skipped: {smoke_analysis['skipped'] + critical_analysis['skipped']}")
    print()

    # Step 2 Recommendations
    print("🎯 STEP 2 IMPLEMENTATION RECOMMENDATIONS")
    print("-" * 50)

    if total_failed >= 10:  # Expected TDD red phase
        print("✅ READY FOR STEP 2 IMPLEMENTATION")
        print("   • Tests define clear requirements")
        print("   • Implementation modules needed:")
        print("     - src/model_evaluation/__init__.py")
        print("     - src/model_evaluation/evaluator.py")
        print("     - src/model_evaluation/comparator.py")
        print("     - src/model_evaluation/visualizer.py")
        print("     - src/model_evaluation/reporter.py")
        print("     - src/model_evaluation/pipeline.py")
        print("   • Focus on making tests pass incrementally")
    else:
        print("⚠️  UNEXPECTED TEST STATE")
        print("   • Review test failures and implementation status")
        print("   • Some model evaluation functionality may already exist")
        print("   • Proceed with caution to Step 2")

    print()
    print("=" * 80)
    print("PHASE 8 STEP 1 COMPLETE - TDD REQUIREMENTS DEFINED")
    print("=" * 80)

    # Return appropriate exit code
    return 0 if total_tests > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
