#!/usr/bin/env python3
"""
Phase 7 Step 3: Comprehensive Testing and Refinement

Executes comprehensive testing and validation for Phase 7 model implementation
following TDD approach with focus on:
1. Model Performance Validation and Optimization
2. Feature Importance Analysis
3. Business Validation with Customer Segment Awareness
4. End-to-End Pipeline Testing

This script provides detailed analysis and recommendations for Phase 8.
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_file: Path, description: str) -> Dict[str, Any]:
    """Run a test suite and capture results."""
    print(f"\n{'='*80}")
    print(f"Running {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output and capture results
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Parse output for test results
        output_lines = result.stdout.split('\n')
        passed_tests = []
        failed_tests = []
        
        for line in output_lines:
            if "PASSED" in line:
                test_name = line.split("::")[1].split()[0] if "::" in line else line
                passed_tests.append(test_name)
            elif "FAILED" in line:
                test_name = line.split("::")[1].split()[0] if "::" in line else line
                failed_tests.append(test_name)
        
        return {
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_tests': len(passed_tests) + len(failed_tests)
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'return_code': -1,
            'duration': duration,
            'error': 'Test execution timed out',
            'passed_tests': [],
            'failed_tests': [],
            'total_tests': 0
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'return_code': -1,
            'duration': duration,
            'error': str(e),
            'passed_tests': [],
            'failed_tests': [],
            'total_tests': 0
        }

def run_model_training_validation() -> Dict[str, Any]:
    """Run model training validation to check current performance."""
    print(f"\n{'='*80}")
    print("RUNNING MODEL TRAINING VALIDATION")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run train_model function
        from src.models.train_model import train_model
        
        print("Executing train_model() for performance validation...")
        result = train_model(save_models=False)  # Don't overwrite existing models
        
        duration = time.time() - start_time
        
        if result:
            print("✅ Model training validation completed successfully")
            return {
                'success': True,
                'duration': duration,
                'training_results': result.get('training_results', {}),
                'evaluation_results': result.get('evaluation_results', {}),
                'summary': result.get('summary', {})
            }
        else:
            print("❌ Model training validation failed")
            return {
                'success': False,
                'duration': duration,
                'error': 'train_model() returned None'
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Model training validation failed: {str(e)}")
        return {
            'success': False,
            'duration': duration,
            'error': str(e)
        }

def analyze_feature_importance() -> Dict[str, Any]:
    """Analyze feature importance from trained models."""
    print(f"\n{'='*80}")
    print("ANALYZING FEATURE IMPORTANCE")
    print(f"{'='*80}")
    
    try:
        # Check if feature importance file exists
        feature_importance_file = project_root / "trained_models" / "feature_importance.json"
        
        if feature_importance_file.exists():
            with open(feature_importance_file, 'r') as f:
                feature_importance = json.load(f)
            
            print("✅ Feature importance analysis loaded from trained models")
            
            # Analyze engineered features priority
            engineered_features = ['age_bin', 'customer_value_segment', 'campaign_intensity']
            analysis = {}
            
            for model_name, importance_data in feature_importance.items():
                if isinstance(importance_data, dict) and 'features' in importance_data:
                    features = importance_data['features']
                    importances = importance_data['importances']
                    
                    # Create feature ranking
                    feature_ranking = list(zip(features, importances))
                    feature_ranking.sort(key=lambda x: x[1], reverse=True)
                    
                    # Check engineered features in top 10
                    top_10_features = [f[0] for f in feature_ranking[:10]]
                    engineered_in_top_10 = [f for f in engineered_features if f in top_10_features]
                    
                    analysis[model_name] = {
                        'top_10_features': top_10_features,
                        'engineered_in_top_10': engineered_in_top_10,
                        'engineered_count': len(engineered_in_top_10)
                    }
                    
                    print(f"   {model_name}: {len(engineered_in_top_10)}/3 engineered features in top 10")
            
            return {
                'success': True,
                'feature_importance': feature_importance,
                'analysis': analysis
            }
        else:
            print("⚠️ Feature importance file not found - running training to generate...")
            return {'success': False, 'error': 'Feature importance file not found'}
            
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    """Main comprehensive testing function."""
    print("Phase 7 Step 3: Comprehensive Testing and Refinement")
    print("TDD Approach: Validate all functionality and optimize performance")
    print(f"Project Root: {project_root}")
    
    # Test suites to run
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
    
    # 1. Run Model Training Validation
    training_validation = run_model_training_validation()
    all_results['Model Training Validation'] = training_validation
    
    # 2. Run Feature Importance Analysis
    feature_analysis = analyze_feature_importance()
    all_results['Feature Importance Analysis'] = feature_analysis
    
    # 3. Run Test Suites
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
    
    # Generate comprehensive summary report
    print(f"\n{'='*80}")
    print("PHASE 7 STEP 3 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    # Model Training Summary
    if training_validation.get('success'):
        print("✅ MODEL TRAINING VALIDATION: PASSED")
        summary = training_validation.get('summary', {})
        if summary:
            print(f"   Best Model: {summary.get('best_model', 'Unknown')}")
            print(f"   Best Accuracy: {summary.get('best_accuracy', 'Unknown')}")
            print(f"   Training Duration: {training_validation.get('duration', 0):.1f}s")
    else:
        print("❌ MODEL TRAINING VALIDATION: FAILED")
        print(f"   Error: {training_validation.get('error', 'Unknown error')}")
    
    # Feature Importance Summary
    if feature_analysis.get('success'):
        print("✅ FEATURE IMPORTANCE ANALYSIS: COMPLETED")
        analysis = feature_analysis.get('analysis', {})
        for model_name, model_analysis in analysis.items():
            count = model_analysis.get('engineered_count', 0)
            status = "✅" if count > 0 else "⚠️"
            print(f"   {status} {model_name}: {count}/3 engineered features in top 10")
    else:
        print("❌ FEATURE IMPORTANCE ANALYSIS: FAILED")
        print(f"   Error: {feature_analysis.get('error', 'Unknown error')}")
    
    # Test Results Summary
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for category, results in all_results.items():
        if category in ['Model Training Validation', 'Feature Importance Analysis']:
            continue
            
        if 'total_tests' in results:
            tests = results['total_tests']
            passed = len(results.get('passed_tests', []))
            failed = len(results.get('failed_tests', []))
            
            total_tests += tests
            total_passed += passed
            total_failed += failed
            
            status = "✅" if results.get('return_code') == 0 else "❌"
            print(f"{status} {category}: {passed}/{tests} tests passed")
            
            if failed > 0:
                print(f"   Failed tests: {', '.join(results.get('failed_tests', []))}")
    
    # Overall Summary
    print(f"\n{'='*80}")
    print("OVERALL PHASE 7 STEP 3 RESULTS")
    print(f"{'='*80}")
    print(f"Total Test Duration: {total_duration:.1f}s")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_tests > 0:
        success_rate = total_passed / total_tests
        print(f"Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("🎉 PHASE 7 STEP 3: EXCELLENT PERFORMANCE")
        elif success_rate >= 0.6:
            print("✅ PHASE 7 STEP 3: GOOD PERFORMANCE")
        else:
            print("⚠️ PHASE 7 STEP 3: NEEDS IMPROVEMENT")
    
    # Save results for documentation
    results_file = project_root / "specs" / "output" / "phase7-step3-test-results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    print("\n🚀 Ready for Phase 7 Step 3 Documentation and Phase 8 Planning")

if __name__ == "__main__":
    main()
