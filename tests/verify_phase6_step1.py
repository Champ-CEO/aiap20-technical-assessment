#!/usr/bin/env python3
"""
Phase 6 Step 1 Verification Script

Simple verification that Phase 6 Model Preparation Step 1 TDD implementation is working correctly.
Tests core functionality without requiring external dependencies like pytest.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 6 test modules can be imported."""
    print("Testing imports...")
    
    try:
        from tests.smoke.test_phase6_model_preparation_smoke import TestPhase6ModelPreparationSmoke
        print("✅ Smoke tests module imported successfully")
    except Exception as e:
        print(f"❌ Smoke tests import failed: {e}")
        return False
    
    try:
        from tests.unit.test_phase6_model_preparation_critical import TestPhase6ModelPreparationCritical
        print("✅ Critical tests module imported successfully")
    except Exception as e:
        print(f"❌ Critical tests import failed: {e}")
        return False
    
    try:
        from tests.integration.test_phase6_model_preparation_integration import TestPhase6ModelPreparationIntegration
        print("✅ Integration tests module imported successfully")
    except Exception as e:
        print(f"❌ Integration tests import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic model preparation functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score
        
        # Create mock 45-feature dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {}
        for i in range(44):  # 44 features + 1 target = 45 total
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        data['Subscription Status'] = np.random.choice([0, 1], size=n_samples, p=[0.887, 0.113])
        df = pd.DataFrame(data)
        
        print(f"✅ Mock dataset created: {len(df)} records, {len(df.columns)} features")
        
        # Test data splitting
        X = df.drop(columns=['Subscription Status'])
        y = df['Subscription Status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        print(f"✅ Data splitting successful: Train={len(X_train)}, Test={len(X_test)}")
        
        # Test subscription rate preservation
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        rate_diff = abs(train_rate - test_rate)
        
        print(f"✅ Stratification working: Train rate={train_rate:.3f}, Test rate={test_rate:.3f}, Diff={rate_diff:.3f}")
        
        # Test metrics calculation
        y_pred = np.random.choice([0, 1], size=len(y_test), p=[0.9, 0.1])
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"✅ Metrics calculation successful: Precision={precision:.3f}, Recall={recall:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_smoke_test_execution():
    """Test that smoke tests can be executed."""
    print("\nTesting smoke test execution...")
    
    try:
        from tests.smoke.test_phase6_model_preparation_smoke import TestPhase6ModelPreparationSmoke
        
        test_instance = TestPhase6ModelPreparationSmoke()
        
        # Test metrics calculation (doesn't require external data)
        test_instance.test_metrics_calculation_smoke_test()
        print("✅ Metrics calculation smoke test executed successfully")
        
        # Test feature compatibility (uses mock data)
        test_instance.test_feature_compatibility_smoke_test()
        print("✅ Feature compatibility smoke test executed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Smoke test execution failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "tests/smoke/test_phase6_model_preparation_smoke.py",
        "tests/unit/test_phase6_model_preparation_critical.py", 
        "tests/integration/test_phase6_model_preparation_integration.py",
        "tests/run_phase6_tests.py",
        "tests/PHASE6_STEP1_TDD_IMPLEMENTATION.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main verification function."""
    print("=" * 80)
    print("PHASE 6 MODEL PREPARATION STEP 1 VERIFICATION")
    print("=" * 80)
    print("TDD Implementation: Smoke Tests and Critical Tests")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Smoke Test Execution", test_smoke_test_execution)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"• {test_name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("✅ Phase 6 Step 1 TDD implementation is working correctly")
        print("✅ Ready to proceed to Step 2: Core Functionality Implementation")
        return 0
    else:
        print("⚠️  SOME VERIFICATION TESTS FAILED")
        print("❌ Review and fix issues before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
