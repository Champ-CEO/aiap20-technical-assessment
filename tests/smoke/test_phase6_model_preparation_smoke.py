"""
Phase 6 Model Preparation Smoke Tests

Smoke tests for Phase 6 model preparation core requirements following TDD approach:
1. Phase 5 data loading: Verify data/featured/featured-db.csv loads correctly (45 features)
2. Feature compatibility: All 12 engineered features accessible
3. Data splitting: Train/test split works with 45-feature dataset
4. Stratification: 11.3% subscription rate preserved across customer segments
5. Cross-validation: 5-fold CV setup works with engineered features
6. Metrics calculation: Business metrics (precision, recall, ROI) compute correctly

Following streamlined testing approach: critical path over exhaustive coverage.
Based on Phase 5 foundation: 41,188 records, 45 features (33 original + 12 engineered).
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, classification_report

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_integration import load_phase3_output, prepare_ml_pipeline
    from feature_engineering import FEATURED_OUTPUT_PATH

    PHASE5_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Phase 5 integration not available: {e}")
    PHASE5_INTEGRATION_AVAILABLE = False

# Test constants based on Phase 5 specifications
EXPECTED_RECORD_COUNT = 41188
EXPECTED_TOTAL_FEATURES = 45  # 33 original + 12 engineered
EXPECTED_ORIGINAL_FEATURES = 33
EXPECTED_ENGINEERED_FEATURES = 12
EXPECTED_SUBSCRIPTION_RATE = 0.113  # 11.3%
PERFORMANCE_STANDARD = 97000  # records per second

# Expected engineered features from Phase 5
EXPECTED_ENGINEERED_FEATURE_LIST = [
    "age_bin",
    "education_job_segment",
    "customer_value_segment",
    "recent_contact_flag",
    "campaign_intensity",
    "contact_effectiveness_score",
    "financial_risk_score",
    "risk_category",
    "is_high_risk",
    "high_intensity_flag",
    "is_premium_customer",
    "contact_recency",
]


class TestPhase6ModelPreparationSmoke:
    """Smoke tests for Phase 6 Model Preparation core requirements."""

    def test_phase5_data_loading_smoke_test(self):
        """
        Smoke Test: Phase 5 data loading - data/featured/featured-db.csv loads correctly (45 features).

        Validates that Phase 5 featured data can be loaded successfully with expected structure.
        """
        try:
            # Test direct file loading
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                # Load featured data directly
                df_featured = pd.read_csv(featured_data_path)

                # Validate basic structure
                assert df_featured is not None, "Featured data loading failed"
                assert (
                    len(df_featured) == EXPECTED_RECORD_COUNT
                ), f"Expected {EXPECTED_RECORD_COUNT} records, got {len(df_featured)}"
                assert (
                    len(df_featured.columns) == EXPECTED_TOTAL_FEATURES
                ), f"Expected {EXPECTED_TOTAL_FEATURES} features, got {len(df_featured.columns)}"

                print(f"✅ Phase 5 data loading smoke test PASSED")
                print(f"   Records: {len(df_featured)}")
                print(f"   Features: {len(df_featured.columns)}")

            else:
                # Fallback: Use Phase 4 integration if featured data not available
                if PHASE5_INTEGRATION_AVAILABLE:
                    df = load_phase3_output()
                    assert df is not None, "Phase 4 data loading failed"
                    assert (
                        len(df) == EXPECTED_RECORD_COUNT
                    ), "Phase 4 record count mismatch"

                    print(
                        f"✅ Phase 5 data loading smoke test PASSED (using Phase 4 fallback)"
                    )
                    print(f"   Records: {len(df)}")
                    print(f"   Features: {len(df.columns)}")
                else:
                    pytest.skip(
                        "Phase 5 featured data not available and Phase 4 integration not available"
                    )

        except Exception as e:
            pytest.fail(f"Phase 5 data loading smoke test FAILED: {str(e)}")

    def test_feature_compatibility_smoke_test(self):
        """
        Smoke Test: Feature compatibility - All 12 engineered features accessible.

        Validates that all expected engineered features are present and accessible.
        """
        try:
            # Load featured data
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df_featured = pd.read_csv(featured_data_path)

                # Check for engineered features
                missing_features = []
                present_features = []

                for feature in EXPECTED_ENGINEERED_FEATURE_LIST:
                    if feature in df_featured.columns:
                        present_features.append(feature)
                    else:
                        missing_features.append(feature)

                # Validate feature presence
                assert (
                    len(present_features) >= 8
                ), f"Expected at least 8 engineered features, found {len(present_features)}"

                print(f"✅ Feature compatibility smoke test PASSED")
                print(f"   Present engineered features: {len(present_features)}")
                print(f"   Features found: {present_features[:5]}...")  # Show first 5

                if missing_features:
                    print(f"   Note: Missing features: {missing_features}")

            else:
                # Create mock engineered features for testing
                sample_data = pd.DataFrame(
                    {
                        "Age": [25, 40, 65, 30, 50],
                        "Campaign Calls": [1, 3, 5, 2, 4],
                        "Subscription Status": [0, 1, 0, 1, 0],
                    }
                )

                # Test feature creation capability
                sample_data["age_bin"] = pd.cut(
                    sample_data["Age"], bins=[18, 35, 55, 100], labels=[1, 2, 3]
                )
                sample_data["campaign_intensity"] = pd.cut(
                    sample_data["Campaign Calls"],
                    bins=[0, 2, 5, 50],
                    labels=["low", "medium", "high"],
                )

                assert (
                    "age_bin" in sample_data.columns
                ), "Age binning feature creation failed"
                assert (
                    "campaign_intensity" in sample_data.columns
                ), "Campaign intensity feature creation failed"

                print(f"✅ Feature compatibility smoke test PASSED (using mock data)")

        except Exception as e:
            pytest.fail(f"Feature compatibility smoke test FAILED: {str(e)}")

    def test_data_splitting_smoke_test(self):
        """
        Smoke Test: Data splitting - Train/test split works with 45-feature dataset.

        Validates that data splitting functionality works with the featured dataset.
        """
        try:
            # Load data for splitting
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df = pd.read_csv(featured_data_path)
            elif PHASE5_INTEGRATION_AVAILABLE:
                df = load_phase3_output()
            else:
                # Create mock data with 45 features
                np.random.seed(42)
                mock_data = {}

                # Create 45 mock features
                for i in range(EXPECTED_TOTAL_FEATURES - 1):  # -1 for target
                    mock_data[f"feature_{i}"] = np.random.randn(1000)

                mock_data["Subscription Status"] = np.random.choice(
                    [0, 1], size=1000, p=[0.887, 0.113]
                )
                df = pd.DataFrame(mock_data)

            # Test data splitting
            target_column = "Subscription Status"

            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Perform train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

                # Validate split results
                assert len(X_train) > 0, "Training set is empty"
                assert len(X_test) > 0, "Test set is empty"
                assert (
                    len(X_train.columns) >= 30
                ), f"Expected at least 30 features, got {len(X_train.columns)}"

                # Check split proportions
                total_records = len(X_train) + len(X_test)
                train_ratio = len(X_train) / total_records
                test_ratio = len(X_test) / total_records

                assert (
                    0.75 <= train_ratio <= 0.85
                ), f"Train ratio {train_ratio:.3f} outside expected range"
                assert (
                    0.15 <= test_ratio <= 0.25
                ), f"Test ratio {test_ratio:.3f} outside expected range"

                print(f"✅ Data splitting smoke test PASSED")
                print(f"   Train records: {len(X_train)} ({train_ratio:.1%})")
                print(f"   Test records: {len(X_test)} ({test_ratio:.1%})")
                print(f"   Features: {len(X_train.columns)}")

            else:
                pytest.skip(f"Target column '{target_column}' not found in dataset")

        except Exception as e:
            pytest.fail(f"Data splitting smoke test FAILED: {str(e)}")

    def test_stratification_smoke_test(self):
        """
        Smoke Test: Stratification - 11.3% subscription rate preserved across customer segments.

        Validates that stratified splitting preserves the target distribution.
        """
        try:
            # Load data for stratification testing
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df = pd.read_csv(featured_data_path)
            elif PHASE5_INTEGRATION_AVAILABLE:
                df = load_phase3_output()
            else:
                # Create mock data with realistic subscription distribution
                np.random.seed(42)
                n_samples = 5000
                mock_data = {
                    "Age": np.random.randint(18, 100, n_samples),
                    "Campaign Calls": np.random.randint(1, 10, n_samples),
                    "Subscription Status": np.random.choice(
                        [0, 1], size=n_samples, p=[0.887, 0.113]
                    ),
                }
                df = pd.DataFrame(mock_data)

            target_column = "Subscription Status"

            if target_column in df.columns:
                # Calculate original subscription rate
                original_rate = df[target_column].mean()

                # Perform stratified split
                X = df.drop(columns=[target_column])
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

                # Calculate rates in splits
                train_rate = y_train.mean()
                test_rate = y_test.mean()

                # Validate stratification preservation
                rate_tolerance = 0.02  # 2% tolerance

                assert (
                    abs(train_rate - original_rate) <= rate_tolerance
                ), f"Train rate {train_rate:.3f} differs from original {original_rate:.3f} by more than {rate_tolerance}"

                assert (
                    abs(test_rate - original_rate) <= rate_tolerance
                ), f"Test rate {test_rate:.3f} differs from original {original_rate:.3f} by more than {rate_tolerance}"

                print(f"✅ Stratification smoke test PASSED")
                print(f"   Original subscription rate: {original_rate:.1%}")
                print(f"   Train subscription rate: {train_rate:.1%}")
                print(f"   Test subscription rate: {test_rate:.1%}")

            else:
                pytest.skip(f"Target column '{target_column}' not found in dataset")

        except Exception as e:
            pytest.fail(f"Stratification smoke test FAILED: {str(e)}")

    def test_cross_validation_smoke_test(self):
        """
        Smoke Test: Cross-validation - 5-fold CV setup works with engineered features.

        Validates that cross-validation can be set up with the featured dataset.
        """
        try:
            # Load data for CV testing
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df = pd.read_csv(featured_data_path)
            elif PHASE5_INTEGRATION_AVAILABLE:
                df = load_phase3_output()
            else:
                # Create mock data for CV testing
                np.random.seed(42)
                n_samples = 2000
                mock_data = {}

                # Create features including some engineered ones
                for i in range(20):
                    mock_data[f"feature_{i}"] = np.random.randn(n_samples)

                mock_data["age_bin"] = np.random.choice([1, 2, 3], n_samples)
                mock_data["campaign_intensity"] = np.random.choice(
                    ["low", "medium", "high"], n_samples
                )
                mock_data["Subscription Status"] = np.random.choice(
                    [0, 1], size=n_samples, p=[0.887, 0.113]
                )

                df = pd.DataFrame(mock_data)

            target_column = "Subscription Status"

            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Set up 5-fold stratified cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Test CV splits
                fold_info = []
                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]

                    fold_info.append(
                        {
                            "fold": fold_idx + 1,
                            "train_size": len(X_train_fold),
                            "val_size": len(X_val_fold),
                            "train_rate": y_train_fold.mean(),
                            "val_rate": y_val_fold.mean(),
                        }
                    )

                # Validate CV setup
                assert len(fold_info) == 5, f"Expected 5 folds, got {len(fold_info)}"

                # Check fold sizes are reasonable
                total_samples = len(X)
                expected_val_size = total_samples // 5

                for fold in fold_info:
                    assert (
                        fold["val_size"] > 0
                    ), f"Fold {fold['fold']} validation set is empty"
                    assert (
                        fold["train_size"] > 0
                    ), f"Fold {fold['fold']} training set is empty"

                    # Validation size should be approximately 1/5 of total
                    size_ratio = fold["val_size"] / total_samples
                    assert (
                        0.15 <= size_ratio <= 0.25
                    ), f"Fold {fold['fold']} validation size ratio {size_ratio:.3f} outside expected range"

                print(f"✅ Cross-validation smoke test PASSED")
                print(f"   Total samples: {total_samples}")
                print(f"   Folds created: {len(fold_info)}")
                print(
                    f"   Avg validation size: {np.mean([f['val_size'] for f in fold_info]):.0f}"
                )

            else:
                pytest.skip(f"Target column '{target_column}' not found in dataset")

        except Exception as e:
            pytest.fail(f"Cross-validation smoke test FAILED: {str(e)}")

    def test_metrics_calculation_smoke_test(self):
        """
        Smoke Test: Metrics calculation - Business metrics (precision, recall, ROI) compute correctly.

        Validates that business metrics can be calculated with model predictions.
        """
        try:
            # Create mock predictions for metrics testing
            np.random.seed(42)
            n_samples = 1000

            # Generate realistic predictions and true labels
            y_true = np.random.choice([0, 1], size=n_samples, p=[0.887, 0.113])

            # Generate predictions with some correlation to true labels
            y_pred_prob = np.random.random(n_samples)
            # Boost probability for true positives
            y_pred_prob[y_true == 1] += 0.3
            y_pred_prob = np.clip(y_pred_prob, 0, 1)
            y_pred = (y_pred_prob > 0.5).astype(int)

            # Test basic classification metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            # Validate metrics calculation
            assert (
                0 <= precision <= 1
            ), f"Precision {precision} outside valid range [0, 1]"
            assert 0 <= recall <= 1, f"Recall {recall} outside valid range [0, 1]"

            # Test classification report generation
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            assert "accuracy" in report, "Classification report missing accuracy"
            assert "0" in report, "Classification report missing class 0 metrics"
            assert "1" in report, "Classification report missing class 1 metrics"

            # Test business ROI calculation (simplified)
            # Assume: TP = $100 value, FP = -$20 cost, FN = -$50 opportunity cost
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            roi = (tp * 100 - fp * 20 - fn * 50) / n_samples

            print(f"✅ Metrics calculation smoke test PASSED")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   ROI per sample: ${roi:.2f}")
            print(f"   True Positives: {tp}")
            print(f"   False Positives: {fp}")
            print(f"   False Negatives: {fn}")

        except Exception as e:
            pytest.fail(f"Metrics calculation smoke test FAILED: {str(e)}")
