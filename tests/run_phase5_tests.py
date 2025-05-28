"""
Phase 5 Feature Engineering Test Runner

Comprehensive test runner for Phase 5 feature engineering following TDD approach.
Executes smoke tests, critical tests, and integration tests in proper sequence.

Test Categories:
1. Smoke Tests (8 tests): Core functionality verification + Phase 4 integration
2. Critical Tests (9 tests): Business requirements validation + quality monitoring + memory optimization
3. Integration Tests (7 tests): Pipeline integration validation + Phase 4 ML pipeline + continuous monitoring

Following streamlined testing approach: critical path over exhaustive coverage.
Based on Phase 3 foundation: 41,188 cleaned records, 33 base features, 100% data quality.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def run_test_category(test_file, category_name, expected_tests):
    """
    Run a specific test category and report results.

    Args:
        test_file: Path to test file
        category_name: Name of test category
        expected_tests: Expected number of tests

    Returns:
        bool: True if all tests passed
    """
    print(f"\n{'='*80}")
    print(f"RUNNING {category_name.upper()}")
    print(f"{'='*80}")
    print(f"Test File: {test_file}")
    print(f"Expected Tests: {expected_tests}")

    start_time = time.time()

    try:
        # Run pytest with verbose output
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        duration = time.time() - start_time

        # Parse results
        output_lines = result.stdout.split("\n")

        # Count passed/failed/skipped tests
        passed_count = 0
        failed_count = 0
        skipped_count = 0

        for line in output_lines:
            if " PASSED " in line:
                passed_count += 1
            elif " FAILED " in line:
                failed_count += 1
            elif " SKIPPED " in line:
                skipped_count += 1

        total_tests = passed_count + failed_count + skipped_count

        # Report results
        print(f"\nResults:")
        print(f"  ✅ Passed: {passed_count}")
        print(f"  ⏭️  Skipped: {skipped_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  📊 Total: {total_tests}/{expected_tests}")
        print(f"  ⏱️  Duration: {duration:.2f} seconds")

        if failed_count > 0:
            print(f"\nFailure Details:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

        # Success if no failures and total tests match expected (passed + skipped = expected)
        success = failed_count == 0 and total_tests == expected_tests

        if success:
            if skipped_count > 0:
                print(
                    f"✅ {category_name} COMPLETED SUCCESSFULLY (with {skipped_count} skipped)"
                )
            else:
                print(f"✅ {category_name} COMPLETED SUCCESSFULLY")
        else:
            print(f"❌ {category_name} FAILED")

        return success

    except Exception as e:
        print(f"❌ Error running {category_name}: {str(e)}")
        return False


def main():
    """Run all Phase 5 feature engineering tests."""
    print("🚀 PHASE 5 FEATURE ENGINEERING TEST SUITE")
    print("=" * 80)
    print("TDD Approach: Smoke Tests → Critical Tests → Integration Tests")
    print("Focus: Critical path over exhaustive coverage")
    print("Foundation: 41,188 records, 33 features, 100% data quality")

    # Test configuration
    test_categories = [
        {
            "file": "tests/smoke/test_phase5_feature_engineering_smoke.py",
            "name": "Smoke Tests - Core Feature Engineering Requirements",
            "expected": 8,
            "description": "Age binning, data flow, output format, critical path verification, Phase 4 integration, data continuity",
        },
        {
            "file": "tests/unit/test_phase5_feature_engineering_critical.py",
            "name": "Critical Tests - Business Feature Requirements",
            "expected": 9,
            "description": "Boundary validation, interaction features, performance, data integrity, Phase 4 continuity, quality monitoring, memory optimization",
        },
        {
            "file": "tests/integration/test_phase5_pipeline_integration.py",
            "name": "Integration Tests - Pipeline Integration",
            "expected": 7,
            "description": "End-to-end flow, continuity, performance, downstream compatibility, Phase 4 ML pipeline, continuous monitoring",
        },
    ]

    # Track overall results
    total_categories = len(test_categories)
    passed_categories = 0
    total_tests_run = 0
    total_passed = 0

    start_time = time.time()

    # Run each test category
    for i, category in enumerate(test_categories, 1):
        print(f"\n📋 Step {i}/{total_categories}: {category['name']}")
        print(f"Description: {category['description']}")

        # Check if test file exists
        test_file = Path(category["file"])
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue

        # Run tests
        success = run_test_category(test_file, category["name"], category["expected"])

        if success:
            passed_categories += 1

        # Note: We'll track actual counts in the summary, not here since we need access to the counts from run_test_category

    # Final summary
    total_duration = time.time() - start_time

    print(f"\n{'='*80}")
    print("PHASE 5 FEATURE ENGINEERING TEST SUMMARY")
    print(f"{'='*80}")

    print(f"📊 Overall Results:")
    print(f"  Categories: {passed_categories}/{total_categories} passed")
    print(f"  Duration: {total_duration:.2f} seconds")
    print(f"  Note: Individual test counts shown in each category above")

    if passed_categories == total_categories:
        print(f"\n🎉 ALL PHASE 5 TESTS PASSED!")
        print(f"✅ Feature engineering requirements validated")
        print(f"✅ TDD approach completed successfully")
        print(f"✅ Ready for Phase 5 implementation")
        return True
    else:
        print(f"\n❌ PHASE 5 TESTS FAILED")
        print(f"Failed categories: {total_categories - passed_categories}")
        print(f"🔧 Fix failing tests before proceeding to implementation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
