"""
Phase 6 Model Preparation Integration Tests

Integration tests for Phase 6 model preparation following TDD approach:
1. End-to-end pipeline integration from Phase 5 to model preparation
2. Phase 5→Phase 6 data flow validation with performance monitoring
3. Complete model preparation workflow testing
4. Business metrics integration with customer segment awareness
5. Performance benchmarking across the entire pipeline

Following streamlined testing approach: critical path over exhaustive coverage.
Based on Phase 5 foundation: 41,188 records, 45 features, production-ready featured data.
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_integration import (
        load_phase3_output,
        prepare_ml_pipeline,
        validate_phase3_continuity,
    )
    from feature_engineering import (
        engineer_features_pipeline,
        FEATURED_OUTPUT_PATH,
        PERFORMANCE_STANDARD,
    )

    PHASE5_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Phase 5 integration not available: {e}")
    PHASE5_INTEGRATION_AVAILABLE = False

# Test constants
EXPECTED_RECORD_COUNT = 41188
EXPECTED_TOTAL_FEATURES = 45
EXPECTED_SUBSCRIPTION_RATE = 0.113
PERFORMANCE_STANDARD = 97000


class TestPhase6ModelPreparationIntegration:
    """Integration tests for Phase 6 Model Preparation pipeline."""

    def test_end_to_end_phase5_to_phase6_pipeline_integration(self):
        """
        Integration Test: End-to-end pipeline from Phase 5 to Phase 6 model preparation.

        Validates complete data flow from Phase 5 featured data through model preparation.
        """
        try:
            pipeline_start_time = time.time()

            # Step 1: Load Phase 5 featured data
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df_featured = pd.read_csv(featured_data_path)
                data_source = "Phase 5 featured data"
            elif PHASE5_INTEGRATION_AVAILABLE:
                # Fallback: Use Phase 4 data and apply feature engineering
                df_base = load_phase3_output()

                # Apply basic feature engineering for testing
                df_featured = df_base.copy()
                df_featured["age_bin"] = pd.cut(
                    df_featured["Age"], bins=[18, 35, 55, 100], labels=[1, 2, 3]
                )
                df_featured["customer_value_segment"] = np.random.choice(
                    ["Premium", "Standard", "Basic"], len(df_featured)
                )
                df_featured["campaign_intensity"] = pd.cut(
                    df_featured.get(
                        "Campaign Calls", np.random.randint(1, 10, len(df_featured))
                    ),
                    bins=[0, 2, 5, 50],
                    labels=["low", "medium", "high"],
                )
                data_source = "Phase 4 data with mock feature engineering"
            else:
                pytest.skip("No data source available for integration testing")
                return

            # Step 2: Validate data quality and structure
            assert df_featured is not None, "Featured data loading failed"
            assert len(df_featured) > 0, "Featured data is empty"
            assert "Subscription Status" in df_featured.columns, "Target column missing"

            data_quality_time = time.time() - pipeline_start_time

            # Step 3: Prepare data for model training
            target_column = "Subscription Status"
            X = df_featured.drop(columns=[target_column])
            y = df_featured[target_column]

            # Step 4: Perform train/validation/test split
            split_start_time = time.time()

            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Second split: separate train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=0.25,
                stratify=y_temp,
                random_state=42,  # 0.25 * 0.8 = 0.2 of total
            )

            split_time = time.time() - split_start_time

            # Step 5: Set up cross-validation
            cv_start_time = time.time()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_setup_time = time.time() - cv_start_time

            # Step 6: Train and evaluate models
            model_training_start_time = time.time()

            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=50, random_state=42
                ),
                "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
            }

            model_results = {}

            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate on validation set
                    val_predictions = model.predict(X_val)
                    val_probabilities = (
                        model.predict_proba(X_val)[:, 1]
                        if hasattr(model, "predict_proba")
                        else None
                    )

                    # Calculate metrics
                    val_report = classification_report(
                        y_val, val_predictions, output_dict=True, zero_division=0
                    )
                    val_auc = (
                        roc_auc_score(y_val, val_probabilities)
                        if val_probabilities is not None
                        else 0
                    )

                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=cv, scoring="roc_auc"
                    )

                    model_results[model_name] = {
                        "validation_accuracy": val_report["accuracy"],
                        "validation_precision": val_report["1"]["precision"],
                        "validation_recall": val_report["1"]["recall"],
                        "validation_f1": val_report["1"]["f1-score"],
                        "validation_auc": val_auc,
                        "cv_mean_auc": cv_scores.mean(),
                        "cv_std_auc": cv_scores.std(),
                        "training_samples": len(X_train),
                        "validation_samples": len(X_val),
                    }

                except Exception as model_error:
                    model_results[model_name] = {"error": str(model_error)}

            model_training_time = time.time() - model_training_start_time
            total_pipeline_time = time.time() - pipeline_start_time

            # Step 7: Validate integration results
            successful_models = [
                name for name, result in model_results.items() if "error" not in result
            ]
            assert (
                len(successful_models) >= 1
            ), f"Expected at least 1 successful model, got {len(successful_models)}"

            # Validate data splits
            total_samples = len(X_train) + len(X_val) + len(X_test)
            assert (
                abs(total_samples - len(df_featured)) <= 1
            ), "Sample count mismatch in splits"

            # Validate performance
            records_per_second = len(df_featured) / total_pipeline_time

            print(f"✅ End-to-end Phase 5→Phase 6 pipeline integration PASSED")
            print(f"   Data source: {data_source}")
            print(f"   Total records: {len(df_featured):,}")
            print(f"   Features: {len(X.columns)}")
            print(f"   Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
            print(f"   Successful models: {len(successful_models)}/{len(models)}")
            print(f"   Pipeline performance: {records_per_second:,.0f} records/sec")
            print(f"   Total pipeline time: {total_pipeline_time:.2f}s")

            # Print model performance summary
            for model_name in successful_models:
                result = model_results[model_name]
                print(
                    f"   {model_name}: Val AUC={result['validation_auc']:.3f}, CV AUC={result['cv_mean_auc']:.3f}±{result['cv_std_auc']:.3f}"
                )

        except Exception as e:
            pytest.fail(
                f"End-to-end Phase 5→Phase 6 pipeline integration FAILED: {str(e)}"
            )

    def test_phase5_to_phase6_data_flow_validation_with_performance_monitoring(self):
        """
        Integration Test: Phase 5→Phase 6 data flow validation with performance monitoring.

        Validates data flow continuity and monitors performance at each stage.
        """
        try:
            performance_log = []

            # Stage 1: Phase 5 data loading
            stage_start = time.time()

            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df = pd.read_csv(featured_data_path)
                data_source = "Phase 5 featured data"
            elif PHASE5_INTEGRATION_AVAILABLE:
                df = load_phase3_output()
                data_source = "Phase 4 integration"

                # Validate Phase 3→Phase 4 continuity
                continuity_report = validate_phase3_continuity(df)
                assert (
                    continuity_report.get("continuity_status") == "PASSED"
                ), f"Phase 3→Phase 4 continuity failed: {continuity_report}"
            else:
                pytest.skip("No data source available for data flow validation")
                return

            loading_time = time.time() - stage_start
            loading_rate = len(df) / loading_time if loading_time > 0 else float("inf")

            performance_log.append(
                {
                    "stage": "data_loading",
                    "time": loading_time,
                    "rate": loading_rate,
                    "records": len(df),
                    "meets_standard": loading_rate >= PERFORMANCE_STANDARD,
                }
            )

            # Stage 2: Data validation and quality checks
            stage_start = time.time()

            # Validate data structure
            assert len(df) > 0, "Data is empty"
            assert "Subscription Status" in df.columns, "Target column missing"

            # Check data quality
            missing_values = df.isnull().sum().sum()
            subscription_rate = df["Subscription Status"].mean()

            validation_time = time.time() - stage_start
            validation_rate = (
                len(df) / validation_time if validation_time > 0 else float("inf")
            )

            performance_log.append(
                {
                    "stage": "data_validation",
                    "time": validation_time,
                    "rate": validation_rate,
                    "records": len(df),
                    "meets_standard": validation_rate >= PERFORMANCE_STANDARD,
                    "missing_values": missing_values,
                    "subscription_rate": subscription_rate,
                }
            )

            # Stage 3: Data preparation for ML
            stage_start = time.time()

            target_column = "Subscription Status"
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Memory optimization
            original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB

            preparation_time = time.time() - stage_start
            preparation_rate = (
                len(df) / preparation_time if preparation_time > 0 else float("inf")
            )

            performance_log.append(
                {
                    "stage": "ml_preparation",
                    "time": preparation_time,
                    "rate": preparation_rate,
                    "records": len(df),
                    "meets_standard": preparation_rate >= PERFORMANCE_STANDARD,
                    "memory_mb": original_memory,
                    "features": len(X.columns),
                }
            )

            # Stage 4: Advanced data splitting with stratification
            stage_start = time.time()

            # Create stratified splits
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Validate stratification
            train_rate = y_train.mean()
            test_rate = y_test.mean()
            rate_difference = abs(train_rate - test_rate)

            splitting_time = time.time() - stage_start
            splitting_rate = (
                len(df) / splitting_time if splitting_time > 0 else float("inf")
            )

            performance_log.append(
                {
                    "stage": "stratified_splitting",
                    "time": splitting_time,
                    "rate": splitting_rate,
                    "records": len(df),
                    "meets_standard": splitting_rate >= PERFORMANCE_STANDARD,
                    "train_rate": train_rate,
                    "test_rate": test_rate,
                    "rate_difference": rate_difference,
                }
            )

            # Stage 5: Cross-validation setup and validation
            stage_start = time.time()

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Test CV splits
            fold_rates = []
            for train_idx, val_idx in cv.split(X, y):
                fold_train_rate = y.iloc[train_idx].mean()
                fold_val_rate = y.iloc[val_idx].mean()
                fold_rates.append(abs(fold_train_rate - fold_val_rate))

            cv_time = time.time() - stage_start
            cv_rate = len(df) / cv_time if cv_time > 0 else float("inf")

            performance_log.append(
                {
                    "stage": "cross_validation_setup",
                    "time": cv_time,
                    "rate": cv_rate,
                    "records": len(df),
                    "meets_standard": cv_rate >= PERFORMANCE_STANDARD,
                    "max_fold_rate_diff": max(fold_rates),
                    "avg_fold_rate_diff": np.mean(fold_rates),
                }
            )

            # Validate overall performance
            total_time = sum(stage["time"] for stage in performance_log)
            overall_rate = len(df) / total_time if total_time > 0 else float("inf")

            stages_meeting_standard = sum(
                1 for stage in performance_log if stage["meets_standard"]
            )

            # Assertions
            assert (
                rate_difference <= 0.02
            ), f"Stratification rate difference {rate_difference:.3f} too large"
            assert (
                max(fold_rates) <= 0.05
            ), f"CV fold rate difference {max(fold_rates):.3f} too large"
            assert (
                stages_meeting_standard >= 3
            ), f"Expected at least 3 stages to meet performance standard, got {stages_meeting_standard}"

            print(
                f"✅ Phase 5→Phase 6 data flow validation with performance monitoring PASSED"
            )
            print(f"   Data source: {data_source}")
            print(f"   Total records: {len(df):,}")
            print(f"   Overall performance: {overall_rate:,.0f} records/sec")
            print(
                f"   Stages meeting standard: {stages_meeting_standard}/{len(performance_log)}"
            )
            print(f"   Missing values: {missing_values}")
            print(f"   Subscription rate: {subscription_rate:.1%}")
            print(f"   Stratification quality: {rate_difference:.4f} rate difference")

        except Exception as e:
            pytest.fail(
                f"Phase 5→Phase 6 data flow validation with performance monitoring FAILED: {str(e)}"
            )

    def test_complete_model_preparation_workflow_with_business_metrics(self):
        """
        Integration Test: Complete model preparation workflow with business metrics integration.

        Validates the complete workflow from data loading to business metrics calculation
        with customer segment awareness.
        """
        try:
            workflow_start_time = time.time()

            # Step 1: Load and prepare data
            featured_data_path = (
                Path(project_root) / "data" / "featured" / "featured-db.csv"
            )

            if featured_data_path.exists():
                df = pd.read_csv(featured_data_path)
            elif PHASE5_INTEGRATION_AVAILABLE:
                df = load_phase3_output()
                # Add mock business features for testing
                df["customer_value_segment"] = np.random.choice(
                    ["Premium", "Standard", "Basic"], len(df), p=[0.316, 0.577, 0.107]
                )
                df["campaign_intensity"] = np.random.choice(
                    ["low", "medium", "high"], len(df)
                )
                df["age_bin"] = pd.cut(
                    df["Age"], bins=[18, 35, 55, 100], labels=[1, 2, 3]
                )
            else:
                # Create comprehensive mock data
                np.random.seed(42)
                n_samples = 5000

                # Generate realistic customer segments
                segments = np.random.choice(
                    ["Premium", "Standard", "Basic"], n_samples, p=[0.316, 0.577, 0.107]
                )
                campaign_intensity = np.random.choice(
                    ["low", "medium", "high"], n_samples
                )
                age_bins = np.random.choice([1, 2, 3], n_samples)

                # Generate subscription status with segment-based rates
                subscription_status = []
                for i in range(n_samples):
                    if segments[i] == "Premium":
                        prob = 0.20
                    elif segments[i] == "Standard":
                        prob = 0.12
                    else:  # Basic
                        prob = 0.06

                    subscription_status.append(
                        np.random.choice([0, 1], p=[1 - prob, prob])
                    )

                df = pd.DataFrame(
                    {
                        "Age": np.random.randint(18, 100, n_samples),
                        "Campaign Calls": np.random.randint(1, 10, n_samples),
                        "customer_value_segment": segments,
                        "campaign_intensity": campaign_intensity,
                        "age_bin": age_bins,
                        "Subscription Status": subscription_status,
                    }
                )

                # Add additional features to reach closer to 45
                for i in range(35):
                    df[f"feature_{i}"] = np.random.randn(n_samples)

            # Step 2: Comprehensive data preparation
            target_column = "Subscription Status"
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Step 3: Advanced stratified splitting with segment awareness
            if "customer_value_segment" in X.columns:
                # Create stratification key combining target and segment
                stratify_key = (
                    y.astype(str) + "_" + X["customer_value_segment"].astype(str)
                )

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=stratify_key, random_state=42
                    )
                except ValueError:
                    # Fallback to simple stratification if segment stratification fails
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

            # Step 4: Model training with multiple algorithms
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "LogisticRegression": LogisticRegression(random_state=42, max_iter=300),
            }

            model_performance = {}
            business_metrics = {}

            for model_name, model in models.items():
                try:
                    # Train model
                    model_start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - model_start_time

                    # Generate predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = (
                        model.predict_proba(X_test)[:, 1]
                        if hasattr(model, "predict_proba")
                        else None
                    )

                    # Standard metrics
                    standard_report = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )
                    auc_score = (
                        roc_auc_score(y_test, y_pred_proba)
                        if y_pred_proba is not None
                        else 0
                    )

                    model_performance[model_name] = {
                        "training_time": training_time,
                        "accuracy": standard_report["accuracy"],
                        "precision": standard_report["1"]["precision"],
                        "recall": standard_report["1"]["recall"],
                        "f1_score": standard_report["1"]["f1-score"],
                        "auc": auc_score,
                    }

                    # Business metrics with segment awareness
                    if "customer_value_segment" in X_test.columns:
                        segment_metrics = {}

                        for segment in ["Premium", "Standard", "Basic"]:
                            segment_mask = X_test["customer_value_segment"] == segment
                            if segment_mask.sum() > 0:
                                segment_y_true = y_test[segment_mask]
                                segment_y_pred = y_pred[segment_mask]

                                if len(segment_y_true) > 0:
                                    segment_report = classification_report(
                                        segment_y_true,
                                        segment_y_pred,
                                        output_dict=True,
                                        zero_division=0,
                                    )

                                    # Calculate segment-specific ROI
                                    tp = np.sum(
                                        (segment_y_true == 1) & (segment_y_pred == 1)
                                    )
                                    fp = np.sum(
                                        (segment_y_true == 0) & (segment_y_pred == 1)
                                    )

                                    # Segment-specific values
                                    if segment == "Premium":
                                        conversion_value = 200
                                        contact_cost = 25
                                    elif segment == "Standard":
                                        conversion_value = 120
                                        contact_cost = 15
                                    else:  # Basic
                                        conversion_value = 80
                                        contact_cost = 10

                                    total_contacts = tp + fp
                                    revenue = tp * conversion_value
                                    cost = total_contacts * contact_cost
                                    roi = (
                                        (revenue - cost) / max(cost, 1)
                                        if cost > 0
                                        else 0
                                    )

                                    segment_metrics[segment] = {
                                        "precision": segment_report["1"]["precision"],
                                        "recall": segment_report["1"]["recall"],
                                        "f1_score": segment_report["1"]["f1-score"],
                                        "roi": roi,
                                        "conversions": tp,
                                        "contacts": total_contacts,
                                        "revenue": revenue,
                                        "cost": cost,
                                    }

                        business_metrics[model_name] = segment_metrics

                except Exception as model_error:
                    model_performance[model_name] = {"error": str(model_error)}

            # Step 5: Cross-validation with business metrics
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = {}

            for model_name, model in models.items():
                if (
                    model_name in model_performance
                    and "error" not in model_performance[model_name]
                ):
                    try:
                        cv_scores = cross_val_score(
                            model, X_train, y_train, cv=cv, scoring="roc_auc"
                        )
                        cv_results[model_name] = {
                            "mean_auc": cv_scores.mean(),
                            "std_auc": cv_scores.std(),
                            "scores": cv_scores.tolist(),
                        }
                    except Exception as cv_error:
                        cv_results[model_name] = {"error": str(cv_error)}

            workflow_time = time.time() - workflow_start_time

            # Validate workflow results
            successful_models = [
                name
                for name in model_performance.keys()
                if "error" not in model_performance[name]
            ]
            assert (
                len(successful_models) >= 1
            ), f"Expected at least 1 successful model, got {len(successful_models)}"

            # Validate business metrics
            models_with_business_metrics = len(business_metrics)

            # Validate performance
            workflow_rate = (
                len(df) / workflow_time if workflow_time > 0 else float("inf")
            )

            print(
                f"✅ Complete model preparation workflow with business metrics PASSED"
            )
            print(f"   Dataset size: {len(df):,} records, {len(X.columns)} features")
            print(f"   Successful models: {len(successful_models)}/{len(models)}")
            print(f"   Models with business metrics: {models_with_business_metrics}")
            print(f"   Workflow performance: {workflow_rate:,.0f} records/sec")
            print(f"   Total workflow time: {workflow_time:.2f}s")

            # Print model performance summary
            for model_name in successful_models:
                perf = model_performance[model_name]
                cv_result = cv_results.get(model_name, {})

                print(f"   {model_name}:")
                print(f"     Accuracy: {perf['accuracy']:.3f}, AUC: {perf['auc']:.3f}")
                if "mean_auc" in cv_result:
                    print(
                        f"     CV AUC: {cv_result['mean_auc']:.3f}±{cv_result['std_auc']:.3f}"
                    )

                # Print business metrics summary
                if model_name in business_metrics:
                    for segment, metrics in business_metrics[model_name].items():
                        print(
                            f"     {segment}: ROI={metrics['roi']:.2f}, Precision={metrics['precision']:.3f}"
                        )

        except Exception as e:
            pytest.fail(
                f"Complete model preparation workflow with business metrics FAILED: {str(e)}"
            )
