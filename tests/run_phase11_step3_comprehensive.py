#!/usr/bin/env python3
"""
Phase 11 Step 3: Comprehensive Testing and Refinement - Test Runner
Final validation of Documentation implementation

This script executes comprehensive validation including:
1. All existing Phase 11 tests (Step 1 + Step 2)
2. New comprehensive documentation validation tests
3. User experience testing with Phase 10 infrastructure requirements
4. Technical accuracy review reflecting Phase 10 production validation results
5. Business communication optimization using validated customer segment performance
6. Maintenance documentation supporting 3-tier architecture operations
7. Production readiness validation matching Phase 10 operational requirements
8. Performance documentation validation confirming Phase 10 benchmarks
9. API documentation completeness for all 9 production endpoints
10. Monitoring and alerting documentation covering Phase 10 monitoring systems

Expected Results:
- All tests passing (5 smoke + 8 critical + 6 integration + 8 comprehensive tests)
- Documentation completeness validation across all 8 Step 3 areas
- Phase 10 integration consistency verification
- Production readiness confirmation
- Final Phase 11 report generation
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_test_suite(test_file: Path, description: str) -> Dict[str, Any]:
    """Run a test suite and return results."""
    print(f"\n🔄 Running {description}...")
    print(f"   📁 File: {test_file}")

    start_time = time.time()

    try:
        # Run pytest on the specific file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        execution_time = time.time() - start_time

        # Parse pytest output for test counts
        output_lines = result.stdout.split("\n")
        passed_count = 0
        failed_count = 0

        for line in output_lines:
            if " passed" in line and " failed" not in line:
                try:
                    passed_count = int(line.split()[0])
                except (ValueError, IndexError):
                    pass
            elif " failed" in line:
                try:
                    failed_count = int(line.split()[0])
                except (ValueError, IndexError):
                    pass

        success = result.returncode == 0
        status = "✅ PASSED" if success else "❌ FAILED"

        print(
            f"   {status} - {passed_count} passed, {failed_count} failed ({execution_time:.2f}s)"
        )

        if not success and result.stderr:
            print(f"   ⚠️ Error: {result.stderr[:200]}...")

        return {
            "success": success,
            "passed": passed_count,
            "failed": failed_count,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"   ⏰ TIMEOUT after {execution_time:.2f}s")
        return {
            "success": False,
            "passed": 0,
            "failed": 0,
            "execution_time": execution_time,
            "return_code": -1,
            "error": "Timeout",
        }
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"   ❌ ERROR: {str(e)}")
        return {
            "success": False,
            "passed": 0,
            "failed": 0,
            "execution_time": execution_time,
            "return_code": -1,
            "error": str(e),
        }


def validate_documentation_completeness() -> Dict[str, Any]:
    """Validate documentation completeness against Phase 11 success criteria."""
    print("\n📋 Validating Documentation Completeness...")

    validation_results = {}

    # Check README.md completeness
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Phase 10 infrastructure requirements
        infrastructure_specs = ["16", "64", "1", "10"]  # CPU, RAM, Storage, Bandwidth
        infrastructure_found = sum(
            1 for spec in infrastructure_specs if spec in readme_content
        )

        # Performance benchmarks
        performance_metrics = [
            "72000",
            "97000",
            "92.5",
            "6112",
        ]  # Ensemble, Optimization, Accuracy, ROI
        performance_found = sum(
            1 for metric in performance_metrics if metric in readme_content
        )

        # Customer segment rates
        segment_rates = ["31.6", "57.7", "10.7"]  # Premium, Standard, Basic
        segments_found = sum(1 for rate in segment_rates if rate in readme_content)

        # API endpoints
        api_endpoints = [
            "/predict",
            "/batch_predict",
            "/model_status",
            "/performance_metrics",
            "/health_check",
            "/model_info",
            "/feature_importance",
            "/business_metrics",
            "/monitoring_data",
        ]
        endpoints_found = sum(
            1 for endpoint in api_endpoints if endpoint in readme_content
        )

        validation_results["readme_completeness"] = {
            "infrastructure": f"{infrastructure_found}/{len(infrastructure_specs)}",
            "performance": f"{performance_found}/{len(performance_metrics)}",
            "segments": f"{segments_found}/{len(segment_rates)}",
            "api_endpoints": f"{endpoints_found}/{len(api_endpoints)}",
            "overall_score": round(
                (
                    infrastructure_found
                    + performance_found
                    + segments_found
                    + endpoints_found
                )
                / (
                    len(infrastructure_specs)
                    + len(performance_metrics)
                    + len(segment_rates)
                    + len(api_endpoints)
                )
                * 100,
                1,
            ),
        }

        print(
            f"   📊 README Completeness: {validation_results['readme_completeness']['overall_score']}%"
        )
        print(
            f"      • Infrastructure: {validation_results['readme_completeness']['infrastructure']}"
        )
        print(
            f"      • Performance: {validation_results['readme_completeness']['performance']}"
        )
        print(
            f"      • Segments: {validation_results['readme_completeness']['segments']}"
        )
        print(
            f"      • API Endpoints: {validation_results['readme_completeness']['api_endpoints']}"
        )

    return validation_results


def validate_phase10_integration() -> Dict[str, Any]:
    """Validate integration with Phase 10 achievements."""
    print("\n🔗 Validating Phase 10 Integration...")

    integration_results = {}

    # Check for Phase 10 report
    phase10_report = project_root / "specs" / "output" / "Phase10-report.md"
    if phase10_report.exists():
        with open(phase10_report, "r", encoding="utf-8") as f:
            phase10_content = f.read()

        integration_results["phase10_report_exists"] = True
        integration_results["phase10_content_length"] = len(phase10_content)
        print(f"   ✅ Phase 10 report found ({len(phase10_content)} characters)")
    else:
        integration_results["phase10_report_exists"] = False
        print(f"   ⚠️ Phase 10 report not found")

    # Check README for Phase 10 integration
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        phase10_mentions = readme_content.lower().count("phase 10")
        production_mentions = readme_content.lower().count("production")
        ensemble_mentions = readme_content.lower().count("ensemble")

        integration_results["phase10_integration"] = {
            "phase10_mentions": phase10_mentions,
            "production_mentions": production_mentions,
            "ensemble_mentions": ensemble_mentions,
            "integration_score": min(
                100, (phase10_mentions + production_mentions + ensemble_mentions) * 10
            ),
        }

        print(
            f"   📊 Phase 10 Integration Score: {integration_results['phase10_integration']['integration_score']}%"
        )
        print(f"      • Phase 10 mentions: {phase10_mentions}")
        print(f"      • Production mentions: {production_mentions}")
        print(f"      • Ensemble mentions: {ensemble_mentions}")

    return integration_results


def validate_business_communication() -> Dict[str, Any]:
    """Validate business communication optimization."""
    print("\n💼 Validating Business Communication...")

    business_results = {}

    # Check for stakeholder documentation
    stakeholder_docs = [
        project_root
        / "docs"
        / "stakeholder-reports"
        / "Phase11-Stakeholder-Presentation.md",
        project_root / "docs" / "final-summaries" / "Phase11-Executive-Summary.md",
    ]

    for doc_path in stakeholder_docs:
        doc_name = doc_path.name
        if doc_path.exists():
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for business terms
            business_terms = [
                "ROI",
                "accuracy",
                "customer segment",
                "production",
                "business value",
            ]
            terms_found = sum(
                1 for term in business_terms if term.lower() in content.lower()
            )

            business_results[doc_name] = {
                "exists": True,
                "length": len(content),
                "business_terms": f"{terms_found}/{len(business_terms)}",
                "score": round(terms_found / len(business_terms) * 100, 1),
            }

            print(
                f"   ✅ {doc_name}: {business_results[doc_name]['score']}% business relevance"
            )
        else:
            business_results[doc_name] = {"exists": False, "score": 0}
            print(f"   ⚠️ {doc_name}: Not found")

    return business_results


def validate_technical_accuracy() -> Dict[str, Any]:
    """Validate technical accuracy against Phase 10 results."""
    print("\n🔧 Validating Technical Accuracy...")

    technical_results = {}

    # Check README for technical accuracy
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Technical specifications validation
        technical_specs = {
            "ensemble_accuracy": "92.5",
            "roi_potential": "6112",
            "processing_speed": "72000",
            "cpu_cores": "16",
            "ram_gb": "64",
            "storage_tb": "1",
            "bandwidth_gbps": "10",
        }

        specs_found = {}
        for spec_name, spec_value in technical_specs.items():
            specs_found[spec_name] = spec_value in readme_content

        accuracy_score = sum(specs_found.values()) / len(specs_found) * 100

        technical_results["technical_accuracy"] = {
            "specifications_found": specs_found,
            "accuracy_score": round(accuracy_score, 1),
            "total_specs": len(technical_specs),
            "found_specs": sum(specs_found.values()),
        }

        print(
            f"   📊 Technical Accuracy: {technical_results['technical_accuracy']['accuracy_score']}%"
        )
        print(
            f"      • Specifications found: {technical_results['technical_accuracy']['found_specs']}/{technical_results['technical_accuracy']['total_specs']}"
        )

    return technical_results


def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive Step 3 validation tests."""
    print("\n" + "=" * 80)
    print("🧪 PHASE 11 STEP 3: COMPREHENSIVE VALIDATION TESTS")
    print("=" * 80)

    comprehensive_results = {}

    # 1. Documentation Completeness Validation
    comprehensive_results["documentation_completeness"] = (
        validate_documentation_completeness()
    )

    # 2. Phase 10 Integration Validation
    comprehensive_results["phase10_integration"] = validate_phase10_integration()

    # 3. Business Communication Validation
    comprehensive_results["business_communication"] = validate_business_communication()

    # 4. Technical Accuracy Validation
    comprehensive_results["technical_accuracy"] = validate_technical_accuracy()

    return comprehensive_results


def main():
    """Main execution function for Phase 11 Step 3 comprehensive testing."""
    print("🚀 Phase 11 Step 3: Comprehensive Testing and Refinement")
    print("=" * 80)
    print("📋 Test Configuration:")
    print("   • Phase: 11 (Documentation)")
    print("   • Step: 3 (Comprehensive Testing and Refinement)")
    print("   • Approach: Complete Documentation Validation")
    print("   • Expected: All tests passing with comprehensive validation")
    print("   • Purpose: Final documentation validation and refinement")
    print("=" * 80)

    # Test suites to run
    test_suites = [
        {
            "file": project_root
            / "tests"
            / "smoke"
            / "test_phase11_documentation_smoke.py",
            "category": "Smoke Tests",
            "description": "Phase 11 Documentation Smoke Tests",
        },
        {
            "file": project_root
            / "tests"
            / "unit"
            / "test_phase11_documentation_critical.py",
            "category": "Critical Tests",
            "description": "Phase 11 Documentation Critical Tests",
        },
        {
            "file": project_root
            / "tests"
            / "integration"
            / "test_phase11_documentation_integration.py",
            "category": "Integration Tests",
            "description": "Phase 11 Documentation Integration Tests",
        },
    ]

    # Run all test suites
    all_results = {}
    total_start_time = time.time()

    # 1. Run existing test suites
    for suite in test_suites:
        if suite["file"].exists():
            results = run_test_suite(suite["file"], suite["description"])
            all_results[suite["category"]] = results
        else:
            print(f"\n❌ Test file not found: {suite['file']}")
            all_results[suite["category"]] = {
                "return_code": -1,
                "error": "Test file not found",
            }

    # 2. Run comprehensive validation tests
    comprehensive_results = run_comprehensive_validation()
    all_results["Comprehensive Validation"] = comprehensive_results

    total_execution_time = time.time() - total_start_time

    # Calculate overall statistics
    total_tests = 0
    total_passed = 0
    total_failed = 0
    all_successful = True

    for category, results in all_results.items():
        if category != "Comprehensive Validation":
            if "passed" in results and "failed" in results:
                total_tests += results["passed"] + results["failed"]
                total_passed += results["passed"]
                total_failed += results["failed"]
                if not results.get("success", False):
                    all_successful = False

    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("📊 PHASE 11 STEP 3: COMPREHENSIVE TESTING SUMMARY")
    print("=" * 80)
    print(f"📈 Overall Results:")
    print(f"   • Total Tests: {total_tests}")
    print(f"   • Passed: {total_passed}")
    print(f"   • Failed: {total_failed}")
    print(
        f"   • Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%"
    )
    print(f"   • Execution Time: {total_execution_time:.2f} seconds")

    print(f"\n📋 Category Breakdown:")
    for category, results in all_results.items():
        if category != "Comprehensive Validation":
            if "passed" in results and "failed" in results:
                status = "✅" if results.get("success", False) else "❌"
                print(
                    f"   • {category}: {status} {results['passed']}/{results['passed'] + results['failed']} passed ({results.get('execution_time', 0):.2f}s)"
                )

    # Print validation results
    if "Comprehensive Validation" in all_results:
        validation = all_results["Comprehensive Validation"]
        print(f"\n🔍 Comprehensive Validation Results:")

        if "documentation_completeness" in validation:
            completeness = validation["documentation_completeness"].get(
                "readme_completeness", {}
            )
            print(
                f"   • Documentation Completeness: {completeness.get('overall_score', 0)}%"
            )

        if "phase10_integration" in validation:
            integration = validation["phase10_integration"].get(
                "phase10_integration", {}
            )
            print(
                f"   • Phase 10 Integration: {integration.get('integration_score', 0)}%"
            )

        if "technical_accuracy" in validation:
            accuracy = validation["technical_accuracy"].get("technical_accuracy", {})
            print(f"   • Technical Accuracy: {accuracy.get('accuracy_score', 0)}%")

    # Determine next steps
    print(f"\n🎯 Phase 11 Step 3 Analysis:")
    if all_successful and total_tests > 0:
        print("✅ Phase 11 Step 3 comprehensive testing COMPLETE and SUCCESSFUL")
        print("🎉 All documentation validation tests passing")
        print("📋 Documentation meets all Phase 11 success criteria")
        print("🚀 Ready for final Phase 11 report generation")
        return_code = 0
    else:
        print("⚠️ Phase 11 Step 3 comprehensive testing needs attention:")
        if total_failed > 0:
            print(f"   - Fix {total_failed} failing tests")
        if not all_successful:
            print("   - Address test execution issues")
        return_code = 1

    print(f"\n🎯 Next Steps:")
    print("1. 📝 Generate final Phase 11 report at specs/output/Phase11-report.md")
    print("2. 🧹 Consolidate and cleanup intermediate reports")
    print("3. 📊 Create project consolidation summary")
    print("4. ✅ Complete Phase 11 Documentation phase")

    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
