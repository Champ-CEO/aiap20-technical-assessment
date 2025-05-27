#!/usr/bin/env python3
"""
Phase 4: Data Integration and Validation - Test Runner

This script runs all Phase 4 tests following the TDD approach outlined in TASKS.md:
1. Smoke Tests: Data integration core requirements
2. Critical Tests: Data quality requirements  
3. Integration Tests: Phase 3 → Phase 4 pipeline integration

Usage:
    python tests/run_phase4_tests.py
    python tests/run_phase4_tests.py --smoke-only
    python tests/run_phase4_tests.py --critical-only
    python tests/run_phase4_tests.py --integration-only
    python tests/run_phase4_tests.py --verbose
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0, end_time - start_time


def run_phase4_smoke_tests(verbose=False):
    """Run Phase 4 smoke tests."""
    verbose_flag = "-v" if verbose else ""
    command = f"python -m pytest tests/smoke/test_phase4_data_integration_smoke.py {verbose_flag}"
    return run_command(command, "Phase 4 Smoke Tests - Data Integration Core Requirements")


def run_phase4_critical_tests(verbose=False):
    """Run Phase 4 critical tests."""
    verbose_flag = "-v" if verbose else ""
    command = f"python -m pytest tests/unit/test_phase4_data_quality_validation.py {verbose_flag}"
    return run_command(command, "Phase 4 Critical Tests - Data Quality Requirements")


def run_phase4_integration_tests(verbose=False):
    """Run Phase 4 integration tests."""
    verbose_flag = "-v" if verbose else ""
    command = f"python -m pytest tests/integration/test_phase4_pipeline_integration.py {verbose_flag}"
    return run_command(command, "Phase 4 Integration Tests - Pipeline Integration")


def run_all_phase4_tests(verbose=False):
    """Run all Phase 4 tests."""
    verbose_flag = "-v" if verbose else ""
    command = f"python -m pytest tests/smoke/test_phase4_data_integration_smoke.py tests/unit/test_phase4_data_quality_validation.py tests/integration/test_phase4_pipeline_integration.py {verbose_flag}"
    return run_command(command, "All Phase 4 Tests - Complete Test Suite")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Phase 4 Data Integration and Validation Test Runner")
    parser.add_argument("--smoke-only", action="store_true", help="Run only smoke tests")
    parser.add_argument("--critical-only", action="store_true", help="Run only critical tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("Phase 4: Data Integration and Validation - Test Runner")
    print("=" * 60)
    print("TDD Approach: Tests define requirements before implementation")
    print("Test Categories:")
    print("  1. Smoke Tests: Data integration core requirements")
    print("  2. Critical Tests: Data quality requirements")
    print("  3. Integration Tests: Phase 3 → Phase 4 pipeline integration")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    if args.smoke_only:
        success, duration = run_phase4_smoke_tests(args.verbose)
        results.append(("Smoke Tests", success, duration))
    elif args.critical_only:
        success, duration = run_phase4_critical_tests(args.verbose)
        results.append(("Critical Tests", success, duration))
    elif args.integration_only:
        success, duration = run_phase4_integration_tests(args.verbose)
        results.append(("Integration Tests", success, duration))
    else:
        # Run all tests in TDD order
        print("\nRunning Phase 4 tests in TDD order...")
        
        # Step 1: Smoke Tests
        success, duration = run_phase4_smoke_tests(args.verbose)
        results.append(("Smoke Tests", success, duration))
        
        # Step 2: Critical Tests
        success, duration = run_phase4_critical_tests(args.verbose)
        results.append(("Critical Tests", success, duration))
        
        # Step 3: Integration Tests
        success, duration = run_phase4_integration_tests(args.verbose)
        results.append(("Integration Tests", success, duration))
        
        # Step 4: All tests together
        success, duration = run_all_phase4_tests(args.verbose)
        results.append(("Complete Test Suite", success, duration))
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE 4 TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status:10} ({duration:.2f}s)")
        if not success:
            all_passed = False
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    if all_passed:
        print("\n🎉 ALL PHASE 4 TESTS PASSED!")
        print("✅ Data Integration Core Requirements: VALIDATED")
        print("✅ Data Quality Requirements: VALIDATED") 
        print("✅ Phase 3 → Phase 4 Pipeline Integration: VALIDATED")
        print("\nPhase 4 is ready for implementation!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the failed tests and fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
