#!/usr/bin/env python3
"""
Test Runner Script for AIAP20 Project

This script provides convenient commands for running different test categories.
Streamlined testing approach focusing on critical path verification.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AIAP20 Test Runner - Streamlined testing for critical path verification"
    )
    parser.add_argument(
        "test_type", 
        choices=["all", "smoke", "unit", "integration", "quick", "coverage"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["pytest"]
    if args.verbose:
        base_cmd.append("-v")
    
    # Test commands
    commands = {
        "smoke": {
            "cmd": base_cmd + ["tests/smoke/", "-m", "smoke"],
            "desc": "Smoke Tests - Quick pipeline verification (Business Value: Ensure development environment works correctly)"
        },
        "unit": {
            "cmd": base_cmd + ["tests/unit/", "-m", "unit"],
            "desc": "Unit Tests - Core function verification (Focus: Individual components)"
        },
        "integration": {
            "cmd": base_cmd + ["tests/integration/", "-m", "integration"],
            "desc": "Integration Tests - Component interactions (Focus: Module interactions)"
        },
        "quick": {
            "cmd": base_cmd + ["-m", "not slow"],
            "desc": "Quick Tests - All fast tests (Excludes slow-running tests)"
        },
        "coverage": {
            "cmd": base_cmd + ["--cov=src", "--cov-report=html", "--cov-report=term-missing"],
            "desc": "Coverage Tests - All tests with coverage report"
        },
        "all": {
            "cmd": base_cmd,
            "desc": "All Tests - Complete test suite"
        }
    }
    
    if args.test_type not in commands:
        print(f"❌ Unknown test type: {args.test_type}")
        return 1
    
    # Run the selected test type
    cmd_info = commands[args.test_type]
    success = run_command(cmd_info["cmd"], cmd_info["desc"])
    
    if success:
        print(f"\n✅ {args.test_type.title()} tests completed successfully!")
        
        # Show next steps
        if args.test_type == "smoke":
            print("\n💡 Next steps:")
            print("   - Run 'python tests/run_tests.py unit' for unit tests")
            print("   - Run 'python tests/run_tests.py integration' for integration tests")
            print("   - Run 'python tests/run_tests.py all' for complete test suite")
        
        elif args.test_type == "coverage":
            print("\n📊 Coverage report generated:")
            print("   - Open 'htmlcov/index.html' in your browser to view detailed coverage")
        
        return 0
    else:
        print(f"\n❌ {args.test_type.title()} tests failed!")
        return 1


if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent.parent
    if script_dir.name != "aiap20":
        # Change to project root directory
        os.chdir(script_dir)
        print(f"ℹ️ Changed working directory to project root: {script_dir}")
    
    sys.exit(main())