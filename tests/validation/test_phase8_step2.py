#!/usr/bin/env python3
"""
Phase 8 Step 2 Implementation Test

Quick test to validate the core functionality of Phase 8 Step 2 implementation.
Tests the main components without running the full pipeline.
"""

import sys
import time
import traceback
sys.path.append('.')

def test_model_evaluator():
    """Test ModelEvaluator functionality."""
    print("🧪 Testing ModelEvaluator...")
    try:
        from src.model_evaluation.evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        print(f"   ✅ Data loaded: {evaluator.data.shape}")
        
        models = evaluator.load_models()
        print(f"   ✅ Models loaded: {len([m for m in models.values() if m is not None])}/5")
        
        # Test single model evaluation
        result = evaluator.evaluate_model("GradientBoosting")
        print(f"   ✅ GradientBoosting evaluation: {result['accuracy']:.4f} accuracy")
        print(f"   ✅ Performance: {result['performance']['records_per_second']:,.0f} records/sec")
        
        return True
    except Exception as e:
        print(f"   ❌ ModelEvaluator test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_comparator():
    """Test ModelComparator functionality."""
    print("🧪 Testing ModelComparator...")
    try:
        from src.model_evaluation.comparator import ModelComparator
        from src.model_evaluation.evaluator import ModelEvaluator
        
        # Get evaluation results
        evaluator = ModelEvaluator()
        evaluator.load_models()
        eval_results = {}
        
        # Evaluate just 2 models for speed
        for model_name in ["GradientBoosting", "NaiveBayes"]:
            eval_results[model_name] = evaluator.evaluate_model(model_name)
        
        # Test comparison
        comparator = ModelComparator()
        comparison_results = comparator.compare_models(eval_results)
        
        print(f"   ✅ Models compared: {len(comparison_results['model_metrics'])}")
        print(f"   ✅ Rankings generated: {len(comparison_results['rankings'])}")
        
        return True
    except Exception as e:
        print(f"   ❌ ModelComparator test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_business_calculator():
    """Test BusinessMetricsCalculator functionality."""
    print("🧪 Testing BusinessMetricsCalculator...")
    try:
        from src.model_evaluation.business_calculator import BusinessMetricsCalculator
        from src.model_evaluation.evaluator import ModelEvaluator
        import numpy as np
        
        # Get sample data
        evaluator = ModelEvaluator()
        evaluator.load_models()
        
        # Use small sample for testing
        sample_size = 1000
        y_true = evaluator.y_test.values[:sample_size]
        
        # Get predictions from one model
        model = evaluator.models["GradientBoosting"]
        X_sample = evaluator.X_test.iloc[:sample_size]
        y_pred = model.predict(X_sample)
        y_pred_proba = model.predict_proba(X_sample)[:, 1]
        
        # Test business metrics
        calculator = BusinessMetricsCalculator()
        roi_results = calculator.calculate_marketing_roi(y_true, y_pred, y_pred_proba)
        
        print(f"   ✅ ROI calculated: {roi_results['overall_roi']:.2%}")
        print(f"   ✅ Segments analyzed: {len(roi_results['segment_roi'])}")
        
        return True
    except Exception as e:
        print(f"   ❌ BusinessMetricsCalculator test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_visualizer():
    """Test ModelVisualizer functionality."""
    print("🧪 Testing ModelVisualizer...")
    try:
        from src.model_evaluation.visualizer import ModelVisualizer
        
        # Create mock data
        mock_evaluation_results = {
            "GradientBoosting": {
                "accuracy": 0.898,
                "f1_score": 0.873,
                "auc_score": 0.801,
                "performance": {"records_per_second": 10000}
            },
            "NaiveBayes": {
                "accuracy": 0.895,
                "f1_score": 0.871,
                "auc_score": 0.757,
                "performance": {"records_per_second": 255000}
            }
        }
        
        visualizer = ModelVisualizer()
        chart_path = visualizer.generate_performance_comparison(mock_evaluation_results)
        
        print(f"   ✅ Performance chart generated: {chart_path}")
        
        return True
    except Exception as e:
        print(f"   ❌ ModelVisualizer test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_main_function():
    """Test the main evaluate_models function with timeout."""
    print("🧪 Testing main evaluate_models function...")
    try:
        from src.model_evaluation.pipeline import evaluate_models
        
        print("   ⏳ Running evaluation (this may take a moment)...")
        start_time = time.time()
        
        # Run evaluation
        results = evaluate_models()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   ✅ Evaluation completed in {duration:.2f}s")
        print(f"   ✅ Results keys: {list(results.keys())}")
        
        # Check if models were evaluated
        if 'detailed_evaluation' in results:
            evaluated_models = len([r for r in results['detailed_evaluation'].values() if r is not None])
            print(f"   ✅ Models evaluated: {evaluated_models}/5")
        
        return True
    except Exception as e:
        print(f"   ❌ Main function test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 8 STEP 2 IMPLEMENTATION TEST")
    print("=" * 60)
    print()
    
    tests = [
        ("ModelEvaluator", test_model_evaluator),
        ("ModelComparator", test_model_comparator),
        ("BusinessMetricsCalculator", test_business_calculator),
        ("ModelVisualizer", test_visualizer),
        ("Main Function", test_main_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Phase 8 Step 2 implementation is working!")
    else:
        print("⚠️  Some tests failed - review implementation")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
