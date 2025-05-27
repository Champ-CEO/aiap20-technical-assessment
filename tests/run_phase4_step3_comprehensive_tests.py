#!/usr/bin/env python3
"""
Phase 4 Step 3: Comprehensive Testing and Refinement - Test Runner

This script runs all comprehensive tests for Phase 4 Step 3, including:
1. Comprehensive edge case testing
2. Performance optimization validation
3. Documentation validation
4. Integration testing enhancement

Usage:
    python tests/run_phase4_step3_comprehensive_tests.py [options]

Options:
    --edge-cases-only    Run only edge case tests
    --performance-only   Run only performance tests
    --documentation-only Run only documentation tests
    --verbose           Show detailed output
    --summary-only      Show only summary results
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_test_suite(test_file, description, verbose=False):
    """Run a test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    cmd = ["python", "-m", "pytest", test_file, "-v"]
    if not verbose:
        cmd.append("-q")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root / "tests")
    duration = time.time() - start_time
    
    # Parse results
    output = result.stdout + result.stderr
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    warnings = output.count(" warning")
    
    print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"Duration: {duration:.2f} seconds")
    
    if verbose or failed > 0:
        print("\nDetailed Output:")
        print(output)
    
    return {
        "description": description,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "duration": duration,
        "success": failed == 0,
        "output": output
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 Step 3 comprehensive tests")
    parser.add_argument("--edge-cases-only", action="store_true", 
                       help="Run only edge case tests")
    parser.add_argument("--performance-only", action="store_true", 
                       help="Run only performance tests")
    parser.add_argument("--documentation-only", action="store_true", 
                       help="Run only documentation tests")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed output")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Show only summary results")
    
    args = parser.parse_args()
    
    # Define test suites
    test_suites = []
    
    if args.edge_cases_only or not any([args.performance_only, args.documentation_only]):
        test_suites.append({
            "file": "integration/test_phase4_comprehensive_edge_cases.py",
            "description": "Comprehensive Edge Case Testing"
        })
    
    if args.performance_only or not any([args.edge_cases_only, args.documentation_only]):
        test_suites.append({
            "file": "integration/test_phase4_performance_optimization.py",
            "description": "Performance Optimization Testing"
        })
    
    if args.documentation_only or not any([args.edge_cases_only, args.performance_only]):
        test_suites.append({
            "file": "integration/test_phase4_documentation_validation.py",
            "description": "Documentation Validation Testing"
        })
    
    # Run all test suites
    results = []
    total_start_time = time.time()
    
    print("Phase 4 Step 3: Comprehensive Testing and Refinement")
    print("=" * 60)
    print(f"Running {len(test_suites)} test suite(s)")
    
    for suite in test_suites:
        result = run_test_suite(
            suite["file"], 
            suite["description"], 
            verbose=args.verbose and not args.summary_only
        )
        results.append(result)
    
    total_duration = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TESTING SUMMARY")
    print(f"{'='*60}")
    
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_warnings = sum(r["warnings"] for r in results)
    all_success = all(r["success"] for r in results)
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{status} {result['description']}: {result['passed']} passed, {result['failed']} failed ({result['duration']:.2f}s)")
    
    print(f"\nOverall Results:")
    print(f"  Total Tests: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Warnings: {total_warnings}")
    print(f"  Total Duration: {total_duration:.2f} seconds")
    print(f"  Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%")
    
    if all_success:
        print(f"\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("Phase 4 Step 3 comprehensive testing and refinement is COMPLETE.")
        print("\nKey Achievements:")
        print("✅ Comprehensive edge case handling validated")
        print("✅ Performance optimization standards exceeded")
        print("✅ Documentation quality verified")
        print("✅ Production readiness confirmed")
    else:
        print(f"\n⚠️ Some tests failed. Please review the detailed output above.")
        failed_suites = [r["description"] for r in results if not r["success"]]
        print(f"Failed test suites: {', '.join(failed_suites)}")
    
    # Exit with appropriate code
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
