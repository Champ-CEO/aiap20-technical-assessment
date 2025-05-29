"""
Evaluation Pipeline Module

Implements the main evaluate_models() function and EvaluationPipeline class.
Provides end-to-end model evaluation workflow with business metrics integration.

Key Features:
- Main evaluate_models() function as specified in Phase 8 Step 2
- End-to-end evaluation pipeline
- Integration with all evaluation components
- Output generation: model_evaluation_report.json, visualizations/, feature_importance_analysis.json
- Performance monitoring (>97K records/second)
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from .evaluator import ModelEvaluator
from .comparator import ModelComparator
from .business_calculator import BusinessMetricsCalculator
from .deployment_validator import ProductionDeploymentValidator
from .visualizer import ModelVisualizer
from .reporter import ReportGenerator
from .feature_analyzer import FeatureImportanceAnalyzer
from .performance_monitor import PerformanceMonitor
from .ensemble_evaluator import EnsembleEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRAINED_MODELS_DIR = "trained_models"
FEATURED_DATA_PATH = "data/featured/featured-db.csv"
OUTPUT_DIR = "specs/output"
VISUALIZATIONS_DIR = "visualizations"
PERFORMANCE_STANDARD = 97000  # >97K records/second


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for Phase 8 implementation.

    Orchestrates the complete model evaluation workflow including
    performance metrics, business analysis, and report generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EvaluationPipeline.

        Args:
            config (Dict, optional): Pipeline configuration
        """
        self.config = config or self._get_default_config()

        # Initialize components
        self.evaluator = ModelEvaluator(
            models_dir=self.config.get("models_directory", TRAINED_MODELS_DIR),
            data_path=self.config.get("data_path", FEATURED_DATA_PATH),
        )
        self.comparator = ModelComparator()
        self.business_calculator = BusinessMetricsCalculator()
        self.deployment_validator = ProductionDeploymentValidator()
        self.visualizer = ModelVisualizer(
            output_dir=self.config.get("visualizations_directory", VISUALIZATIONS_DIR)
        )
        self.reporter = ReportGenerator(
            output_dir=self.config.get("output_directory", OUTPUT_DIR)
        )
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.ensemble_evaluator = EnsembleEvaluator()

        # Results storage
        self.pipeline_results = {}
        self.performance_metrics = {}

        # Create output directories
        self._create_output_directories()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "models_directory": TRAINED_MODELS_DIR,
            "data_path": FEATURED_DATA_PATH,
            "output_directory": OUTPUT_DIR,
            "visualizations_directory": VISUALIZATIONS_DIR,
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1", "auc"],
            "business_analysis": True,
            "generate_visualizations": True,
            "save_reports": True,
        }

    def _create_output_directories(self):
        """Create necessary output directories."""
        output_dir = Path(self.config["output_directory"])
        viz_dir = Path(self.config["visualizations_directory"])

        output_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)

        logger.info(f"Output directories created: {output_dir}, {viz_dir}")

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        start_time = time.time()
        logger.info("Starting Phase 8 model evaluation pipeline...")

        try:
            # Step 1: Load models and evaluate performance
            logger.info("Step 1: Loading models and calculating performance metrics...")
            self.evaluator.load_models()
            evaluation_results = self.evaluator.evaluate_all_models()

            # Step 2: Compare models and generate rankings
            logger.info("Step 2: Comparing models and generating rankings...")
            comparison_results = self.comparator.compare_models(evaluation_results)

            # Step 3: Calculate business metrics
            logger.info("Step 3: Calculating business metrics and ROI analysis...")
            business_results = self._calculate_business_metrics(evaluation_results)

            # Step 4: Validate deployment strategy
            logger.info("Step 4: Validating production deployment strategy...")
            deployment_results = self.deployment_validator.validate_deployment_strategy(
                evaluation_results
            )

            # Step 5: Analyze feature importance
            logger.info("Step 5: Analyzing feature importance...")
            feature_results = self.feature_analyzer.analyze_feature_importance(
                self.evaluator.models, self.evaluator.X_test.columns.tolist()
            )

            # Step 6: Evaluate ensembles
            logger.info("Step 6: Evaluating ensemble models...")
            ensemble_results = self._evaluate_ensembles(evaluation_results)

            # Step 7: Generate visualizations
            logger.info("Step 7: Generating visualizations...")
            visualization_results = self._generate_visualizations(
                evaluation_results, feature_results, business_results
            )

            # Step 8: Generate comprehensive results
            logger.info("Step 8: Compiling comprehensive evaluation results...")
            pipeline_results = self._compile_results(
                evaluation_results,
                comparison_results,
                business_results,
                deployment_results,
                feature_results,
                ensemble_results,
                visualization_results,
            )

            # Step 9: Generate reports
            logger.info("Step 9: Generating evaluation reports...")
            report_results = self._generate_reports(pipeline_results)

            # Step 10: Save outputs
            logger.info("Step 10: Saving evaluation outputs...")
            self._save_outputs(pipeline_results)

            # Performance tracking
            total_time = time.time() - start_time
            total_records = (
                len(self.evaluator.X_test) if hasattr(self.evaluator, "X_test") else 0
            )
            records_per_second = total_records / total_time if total_time > 0 else 0

            pipeline_results["pipeline_performance"] = {
                "total_time": total_time,
                "total_records": total_records,
                "records_per_second": records_per_second,
                "meets_performance_standard": records_per_second
                >= PERFORMANCE_STANDARD,
            }

            self.pipeline_results = pipeline_results

            logger.info(
                f"Pipeline completed successfully in {total_time:.2f}s ({records_per_second:,.0f} records/sec)"
            )
            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _calculate_business_metrics(
        self, evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate business metrics for all models."""
        business_results = {}

        # Get test data
        X_test = self.evaluator.X_test
        y_test = self.evaluator.y_test

        for model_name, results in evaluation_results.items():
            if results is None or model_name not in self.evaluator.models:
                continue

            try:
                model = self.evaluator.models[model_name]

                # Get predictions and probabilities
                y_pred = model.predict(X_test)
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except:
                    y_pred_proba = None

                # Calculate business metrics
                model_business_results = {}

                # Marketing ROI
                if y_pred_proba is not None:
                    roi_results = self.business_calculator.calculate_marketing_roi(
                        y_test.values, y_pred, y_pred_proba
                    )
                    model_business_results["marketing_roi"] = roi_results

                    # Precision/recall trade-offs
                    tradeoff_results = (
                        self.business_calculator.analyze_precision_recall_tradeoffs(
                            y_test.values, y_pred_proba
                        )
                    )
                    model_business_results["precision_recall_tradeoffs"] = (
                        tradeoff_results
                    )

                    # Expected lift
                    lift_results = self.business_calculator.calculate_expected_lift(
                        y_test.values, y_pred_proba
                    )
                    model_business_results["expected_lift"] = lift_results

                business_results[model_name] = model_business_results

            except Exception as e:
                logger.error(
                    f"Error calculating business metrics for {model_name}: {str(e)}"
                )
                business_results[model_name] = None

        return business_results

    def _evaluate_ensembles(
        self, evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate ensemble models."""
        try:
            # Get model rankings from comparison results
            if (
                hasattr(self.comparator, "ranking_results")
                and "overall" in self.comparator.ranking_results
            ):
                model_rankings = self.comparator.ranking_results["overall"]
            else:
                # Create simple ranking based on accuracy
                model_rankings = []
                for model_name, results in evaluation_results.items():
                    if results is not None:
                        accuracy = results.get("accuracy", 0)
                        model_rankings.append((model_name, accuracy))
                model_rankings.sort(key=lambda x: x[1], reverse=True)

            # Evaluate ensembles
            ensemble_results = self.ensemble_evaluator.evaluate_all_ensembles(
                self.evaluator.models,
                model_rankings,
                self.evaluator.X_test,
                self.evaluator.y_test,
            )

            return ensemble_results

        except Exception as e:
            logger.error(f"Error evaluating ensembles: {str(e)}")
            return {"error": str(e)}

    def _generate_visualizations(
        self,
        evaluation_results: Dict[str, Any],
        feature_results: Dict[str, Any],
        business_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate all visualizations."""
        try:
            # Compile data for visualization
            viz_data = {
                "detailed_evaluation": evaluation_results,
                "feature_importance": feature_results,
                "business_analysis": business_results,
            }

            # Generate all charts
            generated_charts = self.visualizer.generate_all_charts(viz_data)

            return {
                "generated_charts": generated_charts,
                "visualization_status": "SUCCESS",
                "charts_created": len(generated_charts),
            }

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {"error": str(e), "visualization_status": "FAILED"}

    def _generate_reports(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation reports."""
        try:
            # Generate main evaluation report
            main_report_path = self.reporter.generate_evaluation_report(
                pipeline_results
            )

            # Generate executive summary
            exec_summary_path = self.reporter.generate_executive_summary(
                pipeline_results
            )

            return {
                "main_report": main_report_path,
                "executive_summary": exec_summary_path,
                "report_status": "SUCCESS",
            }

        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            return {"error": str(e), "report_status": "FAILED"}

    def _compile_results(
        self,
        evaluation_results: Dict[str, Any],
        comparison_results: Dict[str, Any],
        business_results: Dict[str, Any],
        deployment_results: Dict[str, Any],
        feature_results: Dict[str, Any],
        ensemble_results: Dict[str, Any],
        visualization_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile comprehensive evaluation results."""

        # Extract key metrics for summary
        model_summary = {}
        for model_name, results in evaluation_results.items():
            if results is not None:
                model_summary[model_name] = {
                    "accuracy": results.get("accuracy", 0),
                    "f1_score": results.get("f1_score", 0),
                    "auc_score": results.get("auc_score", 0),
                    "records_per_second": results.get("performance", {}).get(
                        "records_per_second", 0
                    ),
                }

        # Get top performers
        top_performers = {}
        if "rankings" in comparison_results:
            rankings = comparison_results["rankings"]
            for criterion, ranking in rankings.items():
                if ranking:
                    top_performers[criterion] = {
                        "model": ranking[0][0],
                        "score": ranking[0][1],
                    }

        # Compile comprehensive results
        compiled_results = {
            "evaluation_summary": {
                "total_models_evaluated": len(
                    [r for r in evaluation_results.values() if r is not None]
                ),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "top_performers": top_performers,
                "model_summary": model_summary,
            },
            "detailed_evaluation": evaluation_results,
            "model_comparison": comparison_results,
            "business_analysis": business_results,
            "deployment_validation": deployment_results,
            "feature_importance": feature_results,
            "ensemble_evaluation": ensemble_results,
            "visualizations": visualization_results,
            "production_recommendations": self._generate_production_recommendations(
                comparison_results
            ),
        }

        return compiled_results

    def _extract_feature_importance(self) -> Dict[str, Any]:
        """Extract feature importance analysis from models."""
        feature_importance = {}

        for model_name, model in self.evaluator.models.items():
            if model is None:
                continue

            try:
                # Try to get feature importance
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = self.evaluator.X_test.columns.tolist()

                    # Create feature importance dictionary
                    importance_dict = dict(zip(feature_names, importances))

                    # Sort by importance
                    sorted_importance = sorted(
                        importance_dict.items(), key=lambda x: x[1], reverse=True
                    )

                    feature_importance[model_name] = {
                        "feature_importance": dict(sorted_importance),
                        "top_5_features": sorted_importance[:5],
                    }
                elif hasattr(model, "coef_"):
                    # For linear models, use coefficient magnitude
                    coef = (
                        np.abs(model.coef_[0])
                        if len(model.coef_.shape) > 1
                        else np.abs(model.coef_)
                    )
                    feature_names = self.evaluator.X_test.columns.tolist()

                    importance_dict = dict(zip(feature_names, coef))
                    sorted_importance = sorted(
                        importance_dict.items(), key=lambda x: x[1], reverse=True
                    )

                    feature_importance[model_name] = {
                        "feature_importance": dict(sorted_importance),
                        "top_5_features": sorted_importance[:5],
                    }

            except Exception as e:
                logger.warning(
                    f"Could not extract feature importance for {model_name}: {str(e)}"
                )
                feature_importance[model_name] = None

        return feature_importance

    def _generate_production_recommendations(
        self, comparison_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate production deployment recommendations."""

        recommendations = {
            "primary_model": None,
            "secondary_model": None,
            "tertiary_model": None,
            "rationale": {},
            "deployment_strategy": "conservative",  # conservative, balanced, aggressive
        }

        if (
            "rankings" in comparison_results
            and "overall" in comparison_results["rankings"]
        ):
            overall_ranking = comparison_results["rankings"]["overall"]

            if len(overall_ranking) >= 1:
                recommendations["primary_model"] = overall_ranking[0][0]
                recommendations["rationale"][
                    "primary"
                ] = f"Best overall performance: {overall_ranking[0][1]:.4f}"

            if len(overall_ranking) >= 2:
                recommendations["secondary_model"] = overall_ranking[1][0]
                recommendations["rationale"][
                    "secondary"
                ] = f"Second best performance: {overall_ranking[1][1]:.4f}"

            if len(overall_ranking) >= 3:
                recommendations["tertiary_model"] = overall_ranking[2][0]
                recommendations["rationale"][
                    "tertiary"
                ] = f"Third best performance: {overall_ranking[2][1]:.4f}"

        return recommendations

    def _save_outputs(self, pipeline_results: Dict[str, Any]):
        """Save evaluation outputs to files."""
        output_dir = Path(self.config["output_directory"])

        # Save model evaluation report
        report_file = output_dir / "model_evaluation_report.json"
        with open(report_file, "w") as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        logger.info(f"Saved evaluation report: {report_file}")

        # Save feature importance analysis
        if "feature_importance" in pipeline_results:
            importance_file = output_dir / "feature_importance_analysis.json"
            with open(importance_file, "w") as f:
                json.dump(
                    pipeline_results["feature_importance"], f, indent=2, default=str
                )
            logger.info(f"Saved feature importance analysis: {importance_file}")


def evaluate_models() -> Dict[str, Any]:
    """
    Main evaluation function for Phase 8 Step 2 implementation.

    Input: trained_models/ + data/featured/featured-db.csv (test split)
    Features: 45 features including 12 engineered business features
    Output: model_evaluation_report.json + visualizations/ + feature_importance_analysis.json
    Business Purpose: Select best model for marketing campaign optimization using customer segments
    Performance: Maintain >97K records/second evaluation standard

    Returns:
        Dict[str, Any]: Comprehensive evaluation results
    """
    logger.info("Starting Phase 8 model evaluation...")

    # Initialize and run evaluation pipeline
    pipeline = EvaluationPipeline()
    results = pipeline.run_evaluation()

    logger.info("Phase 8 model evaluation completed successfully")
    return results
