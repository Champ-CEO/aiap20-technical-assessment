#!/usr/bin/env python3
"""
Phase 6 Model Preparation Test Runner

Runs Phase 6 model preparation tests following TDD approach:
1. Smoke tests first (basic functionality validation)
2. Critical tests (comprehensive requirements validation)
3. Integration tests (end-to-end workflow validation)

Usage:
    python tests/run_phase6_tests.py [--smoke-only] [--critical-only] [--integration-only] [--verbose]
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_type="all", verbose=False):
    """
    Run Phase 6 model preparation tests.
    
    Args:
        test_type (str): Type of tests to run ("smoke", "critical", "integration", "all")
        verbose (bool): Whether to run tests in verbose mode
    """
    
    # Test file paths
    smoke_tests = "tests/smoke/test_phase6_model_preparation_smoke.py"
    critical_tests = "tests/unit/test_phase6_model_preparation_critical.py"
    integration_tests = "tests/integration/test_phase6_model_preparation_integration.py"
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    if verbose:
        base_cmd.extend(["-v", "-s"])
    
    test_results = {}
    
    print("=" * 80)
    print("PHASE 6 MODEL PREPARATION TESTS - TDD APPROACH")
    print("=" * 80)
    print("Following TDD workflow: Smoke → Critical → Integration")
    print("Based on Phase 5 foundation: 41,188 records, 45 features")
    print()
    
    # Run smoke tests
    if test_type in ["smoke", "all"]:
        print("🔥 STEP 1: SMOKE TESTS - Core Requirements Validation")
        print("-" * 60)
        print("Testing basic functionality:")
        print("• Phase 5 data loading (data/featured/featured-db.csv)")
        print("• Feature compatibility (12 engineered features)")
        print("• Data splitting with 45-feature dataset")
        print("• Stratification (11.3% subscription rate preservation)")
        print("• Cross-validation (5-fold CV setup)")
        print("• Metrics calculation (precision, recall, ROI)")
        print()
        
        try:
            cmd = base_cmd + [smoke_tests]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            test_results["smoke"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print("✅ SMOKE TESTS PASSED")
            else:
                print("❌ SMOKE TESTS FAILED")
                if verbose:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
            
        except Exception as e:
            print(f"❌ SMOKE TESTS ERROR: {e}")
            test_results["smoke"] = {"error": str(e)}
        
        print()
    
    # Run critical tests
    if test_type in ["critical", "all"]:
        print("🎯 STEP 2: CRITICAL TESTS - Comprehensive Requirements")
        print("-" * 60)
        print("Testing critical requirements:")
        print("• Phase 5→Phase 6 data flow continuity")
        print("• Feature schema validation (all business features)")
        print("• Stratification with customer segments (Premium: 31.6%, Standard: 57.7%)")
        print("• Cross-validation with class balance preservation")
        print("• Business metrics with customer segment awareness")
        print("• Performance requirements (>97K records/second)")
        print("• Model serialization with 45-feature schema")
        print()
        
        try:
            cmd = base_cmd + [critical_tests]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            test_results["critical"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print("✅ CRITICAL TESTS PASSED")
            else:
                print("❌ CRITICAL TESTS FAILED")
                if verbose:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
            
        except Exception as e:
            print(f"❌ CRITICAL TESTS ERROR: {e}")
            test_results["critical"] = {"error": str(e)}
        
        print()
    
    # Run integration tests
    if test_type in ["integration", "all"]:
        print("🔗 STEP 3: INTEGRATION TESTS - End-to-End Workflow")
        print("-" * 60)
        print("Testing complete workflow:")
        print("• End-to-end Phase 5→Phase 6 pipeline integration")
        print("• Data flow validation with performance monitoring")
        print("• Complete model preparation workflow")
        print("• Business metrics integration")
        print("• Performance benchmarking")
        print()
        
        try:
            cmd = base_cmd + [integration_tests]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            test_results["integration"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print("✅ INTEGRATION TESTS PASSED")
            else:
                print("❌ INTEGRATION TESTS FAILED")
                if verbose:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
            
        except Exception as e:
            print(f"❌ INTEGRATION TESTS ERROR: {e}")
            test_results["integration"] = {"error": str(e)}
        
        print()
    
    # Summary
    print("=" * 80)
    print("PHASE 6 MODEL PREPARATION TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() 
                      if isinstance(result, dict) and result.get("returncode") == 0)
    
    print(f"Total test suites run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print()
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            if "error" in result:
                status = "ERROR"
                details = result["error"]
            elif result.get("returncode") == 0:
                status = "PASSED"
                details = "All tests successful"
            else:
                status = "FAILED"
                details = "Some tests failed"
        else:
            status = "UNKNOWN"
            details = "Unexpected result format"
        
        print(f"• {test_name.upper()} TESTS: {status}")
        if verbose and details:
            print(f"  Details: {details}")
    
    print()
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 6 MODEL PREPARATION TESTS PASSED!")
        print("✅ Ready to proceed to Step 2: Core Functionality Implementation")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Review and fix before proceeding")
        print("❌ TDD Approach: Fix failing tests before implementation")
        return 1

def main():
    """Main function to run Phase 6 tests."""
    parser = argparse.ArgumentParser(description="Run Phase 6 Model Preparation Tests")
    parser.add_argument("--smoke-only", action="store_true", 
                       help="Run only smoke tests")
    parser.add_argument("--critical-only", action="store_true", 
                       help="Run only critical tests")
    parser.add_argument("--integration-only", action="store_true", 
                       help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run tests in verbose mode")
    
    args = parser.parse_args()
    
    # Determine test type
    if args.smoke_only:
        test_type = "smoke"
    elif args.critical_only:
        test_type = "critical"
    elif args.integration_only:
        test_type = "integration"
    else:
        test_type = "all"
    
    # Run tests
    exit_code = run_tests(test_type, args.verbose)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
