"""
Phase 3 Test Runner

Executes all Phase 3 tests according to the priority structure defined in TASKS.md:
- Priority 1: Critical Data Quality Tests
- Priority 2: Data Standardization Tests  
- Priority 3: Data Validation Tests
- Priority 4: Performance and Quality Assurance Tests
- Integration Tests
- Smoke Tests

Usage:
    python tests/run_phase3_tests.py [--priority=1,2,3,4] [--smoke] [--integration] [--all]
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_tests(test_files, test_description, verbose=False):
    """
    Run specified test files and return results.
    
    Args:
        test_files: List of test file paths
        test_description: Description of test category
        verbose: Whether to show detailed output
    
    Returns:
        Tuple of (success, results_summary)
    """
    print(f"\n{'='*60}")
    print(f"Running {test_description}")
    print(f"{'='*60}")
    
    all_success = True
    results_summary = []
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"⚠️  Test file not found: {test_file}")
            results_summary.append(f"❌ {test_file.name}: FILE NOT FOUND")
            all_success = False
            continue
        
        print(f"\n📋 Running: {test_file.name}")
        print("-" * 40)
        
        # Run pytest on the specific file
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_file),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ PASSED ({duration:.2f}s)")
                results_summary.append(f"✅ {test_file.name}: PASSED ({duration:.2f}s)")
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                results_summary.append(f"❌ {test_file.name}: FAILED ({duration:.2f}s)")
                all_success = False
                
                if verbose:
                    print("\nSTDOUT:")
                    print(result.stdout)
                    print("\nSTDERR:")
                    print(result.stderr)
                else:
                    # Show just the failure summary
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'FAILED' in line or 'ERROR' in line:
                            print(f"  {line}")
                            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results_summary.append(f"❌ {test_file.name}: ERROR - {str(e)}")
            all_success = False
    
    return all_success, results_summary


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Phase 3 tests")
    parser.add_argument("--priority", type=str, help="Run specific priority tests (1,2,3,4)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all Phase 3 tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Define test file paths
    test_dir = Path(__file__).parent
    
    priority_tests = {
        1: [test_dir / "unit" / "test_phase3_critical_data_quality.py"],
        2: [test_dir / "unit" / "test_phase3_data_standardization.py"],
        3: [test_dir / "unit" / "test_phase3_data_validation.py"],
        4: [test_dir / "unit" / "test_phase3_performance_quality.py"]
    }
    
    integration_tests = [test_dir / "integration" / "test_phase3_pipeline_integration.py"]
    smoke_tests = [test_dir / "smoke" / "test_phase3_smoke.py"]
    
    # Determine which tests to run
    tests_to_run = []
    
    if args.smoke:
        tests_to_run.append(("Smoke Tests (Quick Validation)", smoke_tests))
    
    elif args.integration:
        tests_to_run.append(("Integration Tests (End-to-End)", integration_tests))
    
    elif args.priority:
        priorities = [int(p.strip()) for p in args.priority.split(",")]
        for priority in priorities:
            if priority in priority_tests:
                priority_name = {
                    1: "Priority 1: Critical Data Quality Tests",
                    2: "Priority 2: Data Standardization Tests", 
                    3: "Priority 3: Data Validation Tests",
                    4: "Priority 4: Performance and Quality Assurance Tests"
                }[priority]
                tests_to_run.append((priority_name, priority_tests[priority]))
            else:
                print(f"⚠️  Invalid priority: {priority}. Valid priorities are 1, 2, 3, 4")
    
    elif args.all:
        # Run all tests in order
        tests_to_run.append(("Smoke Tests (Quick Validation)", smoke_tests))
        for priority in [1, 2, 3, 4]:
            priority_name = {
                1: "Priority 1: Critical Data Quality Tests",
                2: "Priority 2: Data Standardization Tests",
                3: "Priority 3: Data Validation Tests", 
                4: "Priority 4: Performance and Quality Assurance Tests"
            }[priority]
            tests_to_run.append((priority_name, priority_tests[priority]))
        tests_to_run.append(("Integration Tests (End-to-End)", integration_tests))
    
    else:
        # Default: Run smoke tests first, then ask user
        print("🚀 Phase 3 Test Runner")
        print("\nRunning smoke tests first for quick validation...")
        tests_to_run.append(("Smoke Tests (Quick Validation)", smoke_tests))
    
    # Execute tests
    overall_success = True
    all_results = []
    
    start_time = time.time()
    
    for test_description, test_files in tests_to_run:
        success, results = run_tests(test_files, test_description, args.verbose)
        overall_success = overall_success and success
        all_results.extend(results)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("PHASE 3 TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(result)
    
    print(f"\nTotal Duration: {total_duration:.2f}s")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nPhase 3 testing completed successfully.")
        print("✅ Critical data quality issues are properly tested")
        print("✅ Data standardization is validated")
        print("✅ Business rules are enforced")
        print("✅ Pipeline integration works end-to-end")
        print("\n📋 Next Steps:")
        print("   - Review any warnings in test output")
        print("   - Run tests on actual data if available")
        print("   - Proceed with Phase 4 implementation")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("\nPlease review the failed tests and fix issues before proceeding.")
        print("\n🔧 Troubleshooting:")
        print("   - Check that all Phase 3 components are implemented")
        print("   - Verify test fixtures are working correctly")
        print("   - Review error messages for specific issues")
        print("   - Run tests with --verbose for detailed output")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
