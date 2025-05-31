#!/usr/bin/env python3
"""
Phase 11 Step 1: Documentation TDD Test Runner

Executes comprehensive documentation tests for Phase 11 Step 1 requirements.
This script runs all smoke and critical tests to validate documentation requirements
before implementing core functionality.

Test Categories:
- Smoke Tests (5 tests): Core documentation requirements
- Critical Tests (8 tests): Comprehensive documentation validation
- Integration Tests (6 tests): Cross-component documentation integration

Expected Result: TDD Red Phase - Tests should fail until Step 2 implementation

Usage:
    python tests/run_phase11_step1_tests.py
    python tests/run_phase11_step1_tests.py --verbose
    python tests/run_phase11_step1_tests.py --smoke-only
    python tests/run_phase11_step1_tests.py --critical-only
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_category(
    category: str, test_files: List[str], verbose: bool = False
) -> Dict[str, Any]:
    """Run a category of tests and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 PHASE 11 STEP 1: {category.upper()} TESTS")
    print(f"{'='*60}")

    results = {
        "category": category,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "execution_time": 0,
        "test_results": [],
    }

    start_time = time.time()

    for test_file in test_files:
        print(f"\n🔄 Running {test_file}...")

        # Prepare pytest command
        cmd = ["python", "-m", "pytest", test_file, "-v"]
        if verbose:
            cmd.append("-s")

        try:
            # Run the test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Parse results
            output_lines = result.stdout.split("\n")
            test_count = 0
            passed_count = 0
            failed_count = 0

            for line in output_lines:
                if "PASSED" in line:
                    test_count += 1
                    passed_count += 1
                elif "FAILED" in line:
                    test_count += 1
                    failed_count += 1

            test_result = {
                "file": test_file,
                "total": test_count,
                "passed": passed_count,
                "failed": failed_count,
                "return_code": result.returncode,
                "output": result.stdout if verbose else "",
                "error": result.stderr if result.stderr else "",
            }

            results["test_results"].append(test_result)
            results["total_tests"] += test_count
            results["passed_tests"] += passed_count
            results["failed_tests"] += failed_count

            # Display immediate results
            if result.returncode == 0:
                print(f"✅ {test_file}: {passed_count}/{test_count} tests passed")
            else:
                print(f"❌ {test_file}: {failed_count}/{test_count} tests failed")
                if verbose and result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file}: Test execution timed out")
            test_result = {
                "file": test_file,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "return_code": -1,
                "output": "",
                "error": "Test execution timed out",
            }
            results["test_results"].append(test_result)
            results["failed_tests"] += 1

        except Exception as e:
            print(f"💥 {test_file}: Execution error - {e}")
            test_result = {
                "file": test_file,
                "total": 0,
                "passed": 0,
                "failed": 1,
                "return_code": -1,
                "output": "",
                "error": str(e),
            }
            results["test_results"].append(test_result)
            results["failed_tests"] += 1

    results["execution_time"] = time.time() - start_time
    return results


def display_summary(all_results: List[Dict[str, Any]]) -> None:
    """Display comprehensive test execution summary."""
    print(f"\n{'='*60}")
    print("📊 PHASE 11 STEP 1: TDD TEST EXECUTION SUMMARY")
    print(f"{'='*60}")

    total_tests = sum(r["total_tests"] for r in all_results)
    total_passed = sum(r["passed_tests"] for r in all_results)
    total_failed = sum(r["failed_tests"] for r in all_results)
    total_time = sum(r["execution_time"] for r in all_results)

    print(f"📈 Overall Results:")
    print(f"   • Total Tests: {total_tests}")
    print(f"   • Passed: {total_passed}")
    print(f"   • Failed: {total_failed}")
    print(
        f"   • Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%"
    )
    print(f"   • Execution Time: {total_time:.2f} seconds")

    print(f"\n📋 Category Breakdown:")
    for result in all_results:
        category = result["category"]
        tests = result["total_tests"]
        passed = result["passed_tests"]
        failed = result["failed_tests"]
        time_taken = result["execution_time"]

        print(f"   • {category}: {passed}/{tests} passed ({time_taken:.2f}s)")

        # Show individual test files
        for test_result in result["test_results"]:
            file_name = Path(test_result["file"]).name
            file_passed = test_result["passed"]
            file_total = test_result["total"]
            status = "✅" if test_result["return_code"] == 0 else "❌"
            print(f"     - {status} {file_name}: {file_passed}/{file_total}")

    # TDD Phase Analysis
    print(f"\n🎯 TDD Phase Analysis:")
    if total_failed > total_passed:
        print("✅ Perfect TDD Red Phase - Tests correctly fail before implementation")
        print("📋 This indicates proper TDD approach where tests define requirements")
        print("🚀 Ready for Phase 11 Step 2: Core Functionality Implementation")
    elif total_passed > total_failed:
        print("⚠️  Unexpected TDD Green Phase - Some tests are passing")
        print("📋 This may indicate documentation is already comprehensive")
        print("🔍 Review test requirements and documentation completeness")
    else:
        print("🔄 Mixed TDD Phase - Partial implementation detected")
        print("📋 Some documentation exists but not comprehensive")
        print("🎯 Continue with Step 2 implementation")

    print(f"\n🎯 Next Steps:")
    print("1. ✅ Phase 11 Step 1 TDD foundation complete")
    print("2. 🚀 Proceed to Phase 11 Step 2: Core Functionality Implementation")
    print("3. 📝 Implement comprehensive documentation based on test requirements")
    print("4. 🧪 Run Phase 11 Step 3: Comprehensive Testing and Refinement")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Phase 11 Step 1: Documentation TDD Test Runner"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--smoke-only", action="store_true", help="Run only smoke tests"
    )
    parser.add_argument(
        "--critical-only", action="store_true", help="Run only critical tests"
    )

    args = parser.parse_args()

    print("🚀 Phase 11 Step 1: Documentation TDD Test Runner")
    print("=" * 60)
    print("📋 Test Configuration:")
    print("   • Phase: 11 (Documentation)")
    print("   • Step: 1 (TDD Foundation)")
    print("   • Approach: Test-Driven Development")
    print("   • Expected: Red Phase (Tests should fail)")
    print("   • Purpose: Define documentation requirements")
    print("=" * 60)

    # Define test files
    smoke_tests = ["tests/smoke/test_phase11_documentation_smoke.py"]

    critical_tests = ["tests/unit/test_phase11_documentation_critical.py"]

    integration_tests = ["tests/integration/test_phase11_documentation_integration.py"]

    # Validate test files exist
    all_test_files = smoke_tests + critical_tests + integration_tests
    missing_files = [f for f in all_test_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ Missing test files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Please ensure all test files are created before running.")
        return 1

    # Run tests based on arguments
    all_results = []

    if not args.critical_only:
        smoke_results = run_test_category("Smoke Tests", smoke_tests, args.verbose)
        all_results.append(smoke_results)

    if not args.smoke_only:
        critical_results = run_test_category(
            "Critical Tests", critical_tests, args.verbose
        )
        all_results.append(critical_results)

        integration_results = run_test_category(
            "Integration Tests", integration_tests, args.verbose
        )
        all_results.append(integration_results)

    # Display comprehensive summary
    display_summary(all_results)

    # Return appropriate exit code
    total_failed = sum(r["failed_tests"] for r in all_results)
    return 0 if total_failed > 0 else 1  # Success in TDD red phase means tests fail


if __name__ == "__main__":
    sys.exit(main())
