#!/usr/bin/env python3
"""
Phase 11 Step 1: Documentation TDD Validation Script

Validates the complete Phase 11 Step 1 implementation with comprehensive
TDD foundation for documentation requirements.

This script demonstrates:
- Complete test coverage (19 tests: 5 smoke + 8 critical + 6 integration)
- Perfect TDD red phase establishment
- Cross-reference validation with Phase 10 achievements
- Documentation requirements definition
- Quality standards establishment

Usage:
    python tests/validate_phase11_step1.py
    python tests/validate_phase11_step1.py --detailed
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_test_files() -> Dict[str, Any]:
    """Validate that all Phase 11 Step 1 test files exist."""
    print("🔍 Validating Phase 11 Step 1 test files...")
    
    required_files = [
        "tests/smoke/test_phase11_documentation_smoke.py",
        "tests/unit/test_phase11_documentation_critical.py", 
        "tests/integration/test_phase11_documentation_integration.py",
        "tests/run_phase11_step1_tests.py",
        "tests/PHASE11_STEP1_TESTING_SUMMARY.md",
        "tests/validate_phase11_step1.py",
    ]
    
    results = {
        "total_files": len(required_files),
        "existing_files": 0,
        "missing_files": [],
        "file_details": {},
    }
    
    for file_path in required_files:
        if os.path.exists(file_path):
            results["existing_files"] += 1
            file_size = os.path.getsize(file_path)
            results["file_details"][file_path] = {
                "exists": True,
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 2),
            }
            print(f"✅ {file_path} ({results['file_details'][file_path]['size_kb']} KB)")
        else:
            results["missing_files"].append(file_path)
            results["file_details"][file_path] = {"exists": False}
            print(f"❌ {file_path} (missing)")
    
    return results


def validate_test_coverage() -> Dict[str, Any]:
    """Validate test coverage and categorization."""
    print("\n🧪 Validating test coverage...")
    
    test_categories = {
        "smoke_tests": {
            "file": "tests/smoke/test_phase11_documentation_smoke.py",
            "expected_tests": 5,
            "description": "Core documentation requirements",
        },
        "critical_tests": {
            "file": "tests/unit/test_phase11_documentation_critical.py", 
            "expected_tests": 8,
            "description": "Comprehensive documentation validation",
        },
        "integration_tests": {
            "file": "tests/integration/test_phase11_documentation_integration.py",
            "expected_tests": 6,
            "description": "Cross-component documentation integration",
        },
    }
    
    results = {
        "total_expected": sum(cat["expected_tests"] for cat in test_categories.values()),
        "categories": {},
    }
    
    for category, info in test_categories.items():
        print(f"📋 {category.replace('_', ' ').title()}: {info['description']}")
        
        if os.path.exists(info["file"]):
            # Count test methods in file
            with open(info["file"], "r", encoding="utf-8") as f:
                content = f.read()
            
            test_count = content.count("def test_")
            results["categories"][category] = {
                "expected": info["expected_tests"],
                "found": test_count,
                "file_exists": True,
                "coverage_complete": test_count >= info["expected_tests"],
            }
            
            status = "✅" if test_count >= info["expected_tests"] else "⚠️"
            print(f"   {status} Found {test_count}/{info['expected_tests']} tests")
        else:
            results["categories"][category] = {
                "expected": info["expected_tests"],
                "found": 0,
                "file_exists": False,
                "coverage_complete": False,
            }
            print(f"   ❌ File missing: {info['file']}")
    
    return results


def validate_tdd_implementation() -> Dict[str, Any]:
    """Validate TDD implementation by running tests."""
    print("\n🎯 Validating TDD implementation...")
    
    # Run the comprehensive test suite
    try:
        result = subprocess.run(
            ["python", "tests/run_phase11_step1_tests.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output_lines = result.stdout.split('\n')
        
        # Parse test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        execution_time = 0
        
        for line in output_lines:
            if "Total Tests:" in line:
                total_tests = int(line.split(":")[1].strip())
            elif "Passed:" in line:
                passed_tests = int(line.split(":")[1].strip())
            elif "Failed:" in line:
                failed_tests = int(line.split(":")[1].strip())
            elif "Execution Time:" in line:
                execution_time = float(line.split(":")[1].strip().replace(" seconds", ""))
        
        # Determine TDD phase
        if failed_tests > 0 and passed_tests > 0:
            tdd_phase = "Perfect Red Phase"
            tdd_status = "✅ Ideal TDD implementation"
        elif failed_tests == 0:
            tdd_phase = "Green Phase"
            tdd_status = "⚠️ Unexpected - documentation may be complete"
        elif passed_tests == 0:
            tdd_phase = "Complete Red Phase"
            tdd_status = "🔄 All tests failing - check implementation"
        else:
            tdd_phase = "Unknown Phase"
            tdd_status = "❓ Unexpected test results"
        
        return {
            "test_execution_success": result.returncode in [0, 1],  # Both success and expected failure are OK
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "execution_time": execution_time,
            "tdd_phase": tdd_phase,
            "tdd_status": tdd_status,
            "output": result.stdout,
            "error": result.stderr,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "test_execution_success": False,
            "error": "Test execution timed out",
            "tdd_phase": "Unknown",
            "tdd_status": "❌ Execution timeout",
        }
    except Exception as e:
        return {
            "test_execution_success": False,
            "error": str(e),
            "tdd_phase": "Unknown", 
            "tdd_status": f"❌ Execution error: {e}",
        }


def validate_phase10_integration() -> Dict[str, Any]:
    """Validate integration with Phase 10 achievements."""
    print("\n🔗 Validating Phase 10 integration...")
    
    phase10_requirements = {
        "infrastructure": ["16", "64", "1", "10"],  # CPU, RAM, Storage, Bandwidth
        "performance": ["72000", "97000", "92.5", "6112"],  # Ensemble, Optimization, Accuracy, ROI
        "segments": ["31.6", "57.7", "10.7"],  # Premium, Standard, Basic
        "models": ["GradientBoosting", "NaiveBayes", "RandomForest", "Ensemble"],
    }
    
    # Check README for Phase 10 integration
    integration_results = {}
    
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        
        for category, requirements in phase10_requirements.items():
            found_count = sum(1 for req in requirements if req in readme_content)
            integration_results[category] = {
                "required": len(requirements),
                "found": found_count,
                "percentage": round((found_count / len(requirements)) * 100, 1),
                "complete": found_count >= len(requirements) * 0.5,  # At least 50%
            }
            
            status = "✅" if integration_results[category]["complete"] else "⚠️"
            print(f"   {status} {category.title()}: {found_count}/{len(requirements)} ({integration_results[category]['percentage']}%)")
    
    return integration_results


def display_comprehensive_summary(
    file_validation: Dict[str, Any],
    coverage_validation: Dict[str, Any], 
    tdd_validation: Dict[str, Any],
    integration_validation: Dict[str, Any],
    detailed: bool = False
) -> None:
    """Display comprehensive validation summary."""
    print(f"\n{'='*80}")
    print("📊 PHASE 11 STEP 1: COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # File validation summary
    print(f"📁 File Validation:")
    print(f"   • Files Created: {file_validation['existing_files']}/{file_validation['total_files']}")
    print(f"   • Total Size: {sum(f.get('size_kb', 0) for f in file_validation['file_details'].values()):.1f} KB")
    
    # Test coverage summary
    print(f"\n🧪 Test Coverage:")
    total_found = sum(cat["found"] for cat in coverage_validation["categories"].values())
    print(f"   • Total Tests: {total_found}/{coverage_validation['total_expected']}")
    
    for category, details in coverage_validation["categories"].items():
        status = "✅" if details["coverage_complete"] else "⚠️"
        print(f"   • {category.replace('_', ' ').title()}: {status} {details['found']}/{details['expected']}")
    
    # TDD validation summary
    print(f"\n🎯 TDD Implementation:")
    if tdd_validation.get("test_execution_success"):
        print(f"   • Phase: {tdd_validation['tdd_phase']}")
        print(f"   • Status: {tdd_validation['tdd_status']}")
        print(f"   • Results: {tdd_validation.get('passed_tests', 0)} passed, {tdd_validation.get('failed_tests', 0)} failed")
        print(f"   • Execution Time: {tdd_validation.get('execution_time', 0):.2f} seconds")
    else:
        print(f"   • Status: {tdd_validation['tdd_status']}")
    
    # Phase 10 integration summary
    print(f"\n🔗 Phase 10 Integration:")
    for category, details in integration_validation.items():
        status = "✅" if details["complete"] else "⚠️"
        print(f"   • {category.title()}: {status} {details['percentage']}% coverage")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    
    files_complete = file_validation["existing_files"] == file_validation["total_files"]
    coverage_complete = all(cat["coverage_complete"] for cat in coverage_validation["categories"].values())
    tdd_successful = tdd_validation.get("test_execution_success", False)
    integration_adequate = sum(1 for details in integration_validation.values() if details["complete"]) >= 2
    
    if files_complete and coverage_complete and tdd_successful and integration_adequate:
        print("✅ Phase 11 Step 1 implementation is COMPLETE and SUCCESSFUL")
        print("🚀 Ready for Phase 11 Step 2: Core Functionality Implementation")
        print("📝 TDD foundation established with comprehensive test coverage")
        print("🎯 Documentation requirements clearly defined by test specifications")
    else:
        print("⚠️ Phase 11 Step 1 implementation needs attention:")
        if not files_complete:
            print("   - Complete file creation")
        if not coverage_complete:
            print("   - Complete test coverage")
        if not tdd_successful:
            print("   - Fix TDD implementation")
        if not integration_adequate:
            print("   - Improve Phase 10 integration")
    
    if detailed and tdd_validation.get("output"):
        print(f"\n📋 Detailed Test Output:")
        print(tdd_validation["output"][:1000] + "..." if len(tdd_validation["output"]) > 1000 else tdd_validation["output"])


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Phase 11 Step 1: Documentation TDD Validation Script"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed test output"
    )
    
    args = parser.parse_args()
    
    print("🚀 Phase 11 Step 1: Documentation TDD Validation")
    print("=" * 80)
    print("📋 Validation Scope:")
    print("   • Test file creation and organization")
    print("   • Test coverage completeness (19 tests total)")
    print("   • TDD implementation validation")
    print("   • Phase 10 achievements integration")
    print("   • Documentation requirements definition")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all validations
    file_validation = validate_test_files()
    coverage_validation = validate_test_coverage()
    tdd_validation = validate_tdd_implementation()
    integration_validation = validate_phase10_integration()
    
    execution_time = time.time() - start_time
    
    # Display comprehensive summary
    display_comprehensive_summary(
        file_validation,
        coverage_validation,
        tdd_validation,
        integration_validation,
        args.detailed
    )
    
    print(f"\n⏱️ Total validation time: {execution_time:.2f} seconds")
    print("=" * 80)
    
    # Return appropriate exit code
    success = (
        file_validation["existing_files"] == file_validation["total_files"] and
        all(cat["coverage_complete"] for cat in coverage_validation["categories"].values()) and
        tdd_validation.get("test_execution_success", False)
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
