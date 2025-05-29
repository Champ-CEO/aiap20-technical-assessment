#!/usr/bin/env python3
"""
Phase 7 Step 1 Test Runner

Runs Phase 7 Step 1: Smoke Tests and Critical Tests for Model Implementation
following TDD approach.

This script executes all Phase 7 Step 1 tests and provides comprehensive reporting
for the transition to Step 2 (Core Functionality Implementation).

Test Categories:
1. Smoke Tests (6 tests): Core model implementation requirements
2. Critical Tests (7 tests): Performance and business requirements

Expected Results: All tests should initially FAIL (TDD red phase)
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_file, test_description):
    """Run a specific test suite and return results."""
    print(f"\n{'='*80}")
    print(f"Running {test_description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            str(test_file), 
            '-v', 
            '--tb=short',
            '--no-header'
        ], capture_output=True, text=True, cwd=project_root)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Test Duration: {duration:.2f} seconds")
        print(f"Return Code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Parse test results
        lines = result.stdout.split('\n')
        passed_tests = []
        failed_tests = []
        
        for line in lines:
            if '::test_' in line:
                if 'PASSED' in line:
                    passed_tests.append(line.split('::')[1].split(' ')[0])
                elif 'FAILED' in line:
                    failed_tests.append(line.split('::')[1].split(' ')[0])
        
        return {
            'return_code': result.returncode,
            'duration': duration,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_tests': len(passed_tests) + len(failed_tests),
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return {
            'return_code': -1,
            'duration': 0,
            'passed_tests': [],
            'failed_tests': [],
            'total_tests': 0,
            'error': str(e)
        }

def main():
    """Main test runner function."""
    print("Phase 7 Step 1: Smoke Tests and Critical Tests for Model Implementation")
    print("TDD Approach: Tests should initially FAIL (red phase)")
    print(f"Project Root: {project_root}")
    
    # Test files to run
    test_suites = [
        {
            'file': project_root / 'tests' / 'unit' / 'test_phase7_model_implementation_smoke.py',
            'description': 'Phase 7 Model Implementation Smoke Tests',
            'category': 'Smoke Tests',
            'expected_tests': 6
        },
        {
            'file': project_root / 'tests' / 'unit' / 'test_phase7_model_implementation_critical.py',
            'description': 'Phase 7 Model Implementation Critical Tests',
            'category': 'Critical Tests',
            'expected_tests': 7
        }
    ]
    
    # Run all test suites
    all_results = {}
    total_start_time = time.time()
    
    for suite in test_suites:
        if suite['file'].exists():
            results = run_test_suite(suite['file'], suite['description'])
            all_results[suite['category']] = results
        else:
            print(f"\n❌ Test file not found: {suite['file']}")
            all_results[suite['category']] = {
                'return_code': -1,
                'error': 'Test file not found'
            }
    
    total_duration = time.time() - total_start_time
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("PHASE 7 STEP 1 TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for category, results in all_results.items():
        if 'total_tests' in results:
            total_tests += results['total_tests']
            total_passed += len(results['passed_tests'])
            total_failed += len(results['failed_tests'])
            
            print(f"\n{category}:")
            print(f"  Total Tests: {results['total_tests']}")
            print(f"  Passed: {len(results['passed_tests'])}")
            print(f"  Failed: {len(results['failed_tests'])}")
            print(f"  Duration: {results['duration']:.2f}s")
            
            if results['passed_tests']:
                print(f"  Passed Tests: {', '.join(results['passed_tests'])}")
            
            if results['failed_tests']:
                print(f"  Failed Tests: {', '.join(results['failed_tests'])}")
        else:
            print(f"\n{category}: ERROR - {results.get('error', 'Unknown error')}")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total Test Suites: {len(test_suites)}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
    print(f"  Total Duration: {total_duration:.2f}s")
    
    # TDD Analysis
    print(f"\nTDD ANALYSIS:")
    if total_failed > total_passed:
        print("✅ TDD RED PHASE: More tests failing than passing (expected for Step 1)")
        print("   Ready to proceed to Step 2: Core Functionality Implementation")
    elif total_passed > total_failed:
        print("⚠️  TDD GREEN PHASE: More tests passing than failing")
        print("   This suggests some functionality already exists")
    else:
        print("⚠️  TDD MIXED PHASE: Equal passed/failed tests")
    
    # Recommendations for Step 2
    print(f"\nRECOMMENDations FOR STEP 2:")
    print("1. Implement core model functionality to make smoke tests pass")
    print("2. Focus on Phase 6 integration points")
    print("3. Implement 45-feature dataset compatibility")
    print("4. Add model training and prediction capabilities")
    print("5. Implement serialization functionality")
    print("6. Address performance and business requirements")
    
    # Expected test counts validation
    expected_smoke_tests = 6
    expected_critical_tests = 7
    expected_total = expected_smoke_tests + expected_critical_tests
    
    if total_tests == expected_total:
        print(f"\n✅ Test count validation: {total_tests}/{expected_total} tests found")
    else:
        print(f"\n⚠️  Test count validation: {total_tests}/{expected_total} tests found")
        print("   Some tests may be missing or not discovered")
    
    # Exit with appropriate code
    if total_tests == 0:
        print("\n❌ No tests were executed")
        return 1
    elif total_failed == total_tests:
        print("\n✅ All tests failed (expected for TDD Step 1)")
        return 0  # Success for TDD red phase
    else:
        print(f"\n⚠️  Mixed results: {total_passed} passed, {total_failed} failed")
        return 0  # Still success, but note the mixed results

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
