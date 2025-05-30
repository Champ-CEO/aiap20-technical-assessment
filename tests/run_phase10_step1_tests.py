#!/usr/bin/env python3
"""
Phase 10 Step 1: Pipeline Integration TDD Test Runner

Executes comprehensive test suite for Phase 10 Step 1 requirements validation.
This script runs all Phase 10 Step 1 tests to validate TDD red phase before
implementing core functionality in Step 2.

Test Categories:
1. Smoke Tests (5 tests) - Core pipeline integration requirements
2. Critical Tests (6 tests) - Production requirements validation
3. Integration Tests (6 tests) - End-to-end integration scenarios

Expected Result: TDD Red Phase - Most tests should fail until Step 2 implementation

Usage:
    python tests/run_phase10_step1_tests.py
    python tests/run_phase10_step1_tests.py --verbose
    python tests/run_phase10_step1_tests.py --smoke-only
    python tests/run_phase10_step1_tests.py --critical-only
    python tests/run_phase10_step1_tests.py --integration-only
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

# Ensure we're in the project root
script_dir = Path(__file__).parent.parent
if script_dir.name != "aiap20":
    os.chdir(script_dir)
    print(f"ℹ️ Changed working directory to project root: {script_dir}")

# Test file paths
SMOKE_TESTS = "tests/smoke/test_phase10_pipeline_integration_smoke.py"
CRITICAL_TESTS = "tests/unit/test_phase10_pipeline_integration_critical.py"
INTEGRATION_TESTS = "tests/integration/test_phase10_comprehensive_integration.py"


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status}: {description}")

        return success

    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def run_smoke_tests(verbose=False):
    """Run Phase 10 Step 1 smoke tests."""
    cmd = ["pytest", SMOKE_TESTS, "-v" if verbose else "-q"]
    return run_command(cmd, "Phase 10 Step 1: Smoke Tests (5 tests)")


def run_critical_tests(verbose=False):
    """Run Phase 10 Step 1 critical tests."""
    cmd = ["pytest", CRITICAL_TESTS, "-v" if verbose else "-q"]
    return run_command(cmd, "Phase 10 Step 1: Critical Tests (6 tests)")


def run_integration_tests(verbose=False):
    """Run Phase 10 Step 1 integration tests."""
    cmd = ["pytest", INTEGRATION_TESTS, "-v" if verbose else "-q"]
    return run_command(cmd, "Phase 10 Step 1: Integration Tests (6 tests)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 10 Step 1: Pipeline Integration TDD Test Runner"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose test output"
    )
    parser.add_argument(
        "--smoke-only", action="store_true", help="Run only smoke tests"
    )
    parser.add_argument(
        "--critical-only", action="store_true", help="Run only critical tests"
    )
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )

    args = parser.parse_args()

    print("🚀 Phase 10 Step 1: Pipeline Integration TDD Test Runner")
    print("=" * 60)
    print("📋 Test Categories:")
    print("   • Smoke Tests (5 tests) - Core pipeline integration requirements")
    print("   • Critical Tests (6 tests) - Production requirements validation")
    print("   • Integration Tests (6 tests) - End-to-end integration scenarios")
    print(
        "\n🎯 Expected Result: TDD Red Phase - Most tests should fail until Step 2 implementation"
    )
    print("=" * 60)

    start_time = time.time()
    results = {}

    # Run selected test categories
    if args.smoke_only:
        results["smoke"] = run_smoke_tests(args.verbose)
    elif args.critical_only:
        results["critical"] = run_critical_tests(args.verbose)
    elif args.integration_only:
        results["integration"] = run_integration_tests(args.verbose)
    else:
        # Run all test categories
        results["smoke"] = run_smoke_tests(args.verbose)
        results["critical"] = run_critical_tests(args.verbose)
        results["integration"] = run_integration_tests(args.verbose)

    # Calculate results
    total_time = time.time() - start_time
    passed_categories = sum(1 for success in results.values() if success)
    total_categories = len(results)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 PHASE 10 STEP 1 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"📈 Categories passed: {passed_categories}/{total_categories}")

    for category, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   • {category.title()} Tests: {status}")

    # TDD Red Phase Analysis
    print(f"\n🔍 TDD RED PHASE ANALYSIS:")
    if passed_categories == 0:
        print("✅ Perfect TDD Red Phase - All tests failed as expected")
        print("   This indicates proper TDD implementation where tests define")
        print("   requirements before implementation.")
    elif passed_categories < total_categories:
        print(
            f"⚠️  Partial TDD Red Phase - {total_categories - passed_categories} categories failed"
        )
        print("   Some tests passed, which may indicate partial implementation")
        print("   or tests that don't properly validate implementation.")
    else:
        print("❌ Unexpected TDD Green Phase - All tests passed")
        print("   This suggests either implementation already exists or")
        print("   tests are not properly validating implementation requirements.")

    # Next steps
    print(f"\n📋 NEXT STEPS:")
    print("1. 🔧 Phase 10 Step 2: Core Functionality Implementation")
    print("   - Implement pipeline integration modules")
    print("   - Create main.py and run.sh execution scripts")
    print("   - Integrate all Phase 9 optimization modules")
    print("   - Implement performance monitoring and business metrics")

    print("\n2. 🧪 Phase 10 Step 3: Comprehensive Testing and Refinement")
    print("   - Run all tests to validate implementation")
    print("   - Performance optimization (>97K records/second)")
    print("   - End-to-end pipeline validation")
    print("   - Generate comprehensive documentation")

    print(
        f"\n📄 Test Results Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    # Return appropriate exit code
    if args.smoke_only or args.critical_only or args.integration_only:
        return 0 if list(results.values())[0] else 1
    else:
        # For TDD red phase, we expect failures, so return 0 if most tests failed
        return 0 if passed_categories <= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
