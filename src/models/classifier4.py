"""
Classifier 4: Naive Bayes

Efficient probabilistic classifier, excellent for marketing probability estimates.
Provides fast training and prediction with strong probabilistic foundations for customer targeting.

Key Features:
- Fast training and prediction
- Strong probabilistic foundations
- Excellent for probability estimates
- Handles categorical features naturally
- Robust to irrelevant features
- Good baseline for comparison
"""

import numpy as np
import time
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.compose import ColumnTransformer
from .base_classifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    """
    Naive Bayes classifier for term deposit subscription prediction.

    Optimized for probabilistic estimates and fast prediction with mixed data types.
    Provides excellent probability calibration for marketing applications.
    """

    def __init__(self, var_smoothing=1e-9, alpha=1.0, fit_prior=True, class_prior=None):
        """
        Initialize Naive Bayes classifier.

        Args:
            var_smoothing (float): Smoothing parameter for Gaussian features
            alpha (float): Additive smoothing parameter for categorical features
            fit_prior (bool): Whether to learn class prior probabilities
            class_prior (array-like): Prior probabilities of classes
        """
        super().__init__("Naive Bayes")
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.categorical_features = []
        self.numerical_features = []

    def _create_model(self):
        """
        Create Naive Bayes model instance.
        Uses GaussianNB for numerical features and handles categorical separately.

        Returns:
            GaussianNB: Configured sklearn model
        """
        return GaussianNB(var_smoothing=self.var_smoothing, priors=self.class_prior)

    def _identify_feature_types(self, X):
        """
        Identify categorical and numerical features.

        Args:
            X (pd.DataFrame): Input features
        """
        self.categorical_features = list(X.select_dtypes(include=["object"]).columns)
        self.numerical_features = list(X.select_dtypes(exclude=["object"]).columns)

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier with proper feature type handling.

        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target

        Returns:
            self: Trained classifier
        """
        start_time = time.time()

        # Store feature names and identify types
        self.feature_names = list(X.columns)
        self._identify_feature_types(X)

        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=True)

        # Create and train model (using GaussianNB for all features after encoding)
        if self.model is None:
            self.model = self._create_model()

        # Train the model
        self.model.fit(X_encoded, y)

        # Record training time
        self.training_time = time.time() - start_time
        self.is_trained = True

        return self

    def get_class_probabilities(self):
        """
        Get learned class probabilities.

        Returns:
            dict: Class probability information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting class probabilities")

        class_priors = self.model.class_prior_
        classes = self.model.classes_

        return {
            "class_priors": dict(zip(classes, class_priors)),
            "class_counts": dict(zip(classes, self.model.class_count_)),
            "total_samples": self.model.class_count_.sum(),
        }

    def get_feature_likelihood_analysis(self):
        """
        Analyze feature likelihoods for each class.

        Returns:
            dict: Feature likelihood analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing likelihoods")

        # Get feature statistics for each class
        feature_stats = {}

        # For Gaussian NB, we have theta (mean) and sigma (variance) for each feature and class
        means = self.model.theta_  # Shape: (n_classes, n_features)
        variances = self.model.sigma_  # Shape: (n_classes, n_features)
        classes = self.model.classes_

        for i, feature in enumerate(self.feature_names):
            feature_stats[feature] = {}
            for j, class_label in enumerate(classes):
                feature_stats[feature][f"class_{class_label}"] = {
                    "mean": means[j, i],
                    "variance": variances[j, i],
                    "std": np.sqrt(variances[j, i]),
                }

        return feature_stats

    def get_feature_discriminative_power(self):
        """
        Calculate discriminative power of features based on class separation.

        Returns:
            dict: Feature discriminative power scores
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before calculating discriminative power"
            )

        feature_stats = self.get_feature_likelihood_analysis()
        discriminative_power = {}

        for feature, stats in feature_stats.items():
            # Calculate separation between classes
            class_means = [info["mean"] for info in stats.values()]
            class_vars = [info["variance"] for info in stats.values()]

            # Use coefficient of variation and mean separation as discriminative power
            mean_separation = np.std(class_means)
            avg_variance = np.mean(class_vars)

            # Discriminative power: higher mean separation, lower average variance
            if avg_variance > 0:
                discriminative_power[feature] = mean_separation / np.sqrt(avg_variance)
            else:
                discriminative_power[feature] = mean_separation

        return discriminative_power

    def get_probability_calibration_analysis(self, X, y):
        """
        Analyze probability calibration quality.

        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): Test target

        Returns:
            dict: Probability calibration analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing calibration")

        # Get predicted probabilities
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)

        # Analyze calibration by binning probabilities
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_analysis = {}

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Find predictions in this probability bin
            in_bin = (probabilities[:, 1] > bin_lower) & (
                probabilities[:, 1] <= bin_upper
            )

            if np.sum(in_bin) > 0:
                # Calculate actual positive rate in this bin
                actual_positive_rate = np.mean(y[in_bin])
                predicted_positive_rate = np.mean(probabilities[in_bin, 1])
                count = np.sum(in_bin)

                calibration_analysis[f"bin_{i+1}"] = {
                    "range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                    "count": count,
                    "predicted_rate": predicted_positive_rate,
                    "actual_rate": actual_positive_rate,
                    "calibration_error": abs(
                        predicted_positive_rate - actual_positive_rate
                    ),
                }

        # Calculate overall calibration metrics
        overall_calibration_error = np.mean(
            [
                bin_info["calibration_error"] * bin_info["count"]
                for bin_info in calibration_analysis.values()
            ]
        ) / len(y)

        return {
            "bin_analysis": calibration_analysis,
            "overall_calibration_error": overall_calibration_error,
            "probability_range": {
                "min": np.min(probabilities[:, 1]),
                "max": np.max(probabilities[:, 1]),
                "mean": np.mean(probabilities[:, 1]),
            },
        }

    def get_marketing_insights(self, X=None):
        """
        Generate marketing-specific insights from probability estimates.

        Args:
            X (pd.DataFrame): Optional features for customer segmentation

        Returns:
            dict: Marketing insights and targeting recommendations
        """
        if not self.is_trained:
            raise ValueError(
                "Model must be trained before generating marketing insights"
            )

        # Get class probabilities and feature analysis
        class_probs = self.get_class_probabilities()
        discriminative_power = self.get_feature_discriminative_power()

        # Identify top discriminative features
        top_features = sorted(
            discriminative_power.items(), key=lambda x: x[1], reverse=True
        )[:5]

        insights = {
            "baseline_conversion_rate": class_probs["class_priors"].get(1, 0),
            "top_discriminative_features": top_features,
            "targeting_recommendations": self._generate_targeting_recommendations(
                top_features
            ),
            "probability_thresholds": self._suggest_probability_thresholds(class_probs),
            "feature_insights": self._analyze_feature_business_impact(
                discriminative_power
            ),
        }

        # Add customer segmentation if data provided
        if X is not None:
            probabilities = self.predict_proba(X)
            insights["customer_segmentation"] = self._segment_customers_by_probability(
                probabilities
            )

        return insights

    def _generate_targeting_recommendations(self, top_features):
        """
        Generate targeting recommendations based on discriminative features.

        Args:
            top_features (list): Top discriminative features

        Returns:
            list: Targeting recommendations
        """
        recommendations = []

        for feature, power in top_features:
            if "age" in feature.lower():
                recommendations.append(
                    f"Age-based targeting is highly effective (discriminative power: {power:.3f})"
                )
            elif "education" in feature.lower():
                recommendations.append(
                    f"Education level drives subscription decisions (power: {power:.3f})"
                )
            elif "campaign" in feature.lower():
                recommendations.append(
                    f"Campaign strategy is critical for success (power: {power:.3f})"
                )
            elif "contact" in feature.lower():
                recommendations.append(
                    f"Contact method significantly impacts conversion (power: {power:.3f})"
                )
            elif "loan" in feature.lower():
                recommendations.append(
                    f"Loan status affects subscription likelihood (power: {power:.3f})"
                )

        return recommendations

    def _suggest_probability_thresholds(self, class_probs):
        """
        Suggest probability thresholds for different marketing strategies.

        Args:
            class_probs (dict): Class probability information

        Returns:
            dict: Suggested thresholds
        """
        baseline_rate = class_probs["class_priors"].get(1, 0.113)

        return {
            "conservative_threshold": 0.7,  # High confidence targeting
            "balanced_threshold": 0.5,  # Standard threshold
            "aggressive_threshold": 0.3,  # Broad targeting
            "baseline_rate": baseline_rate,
            "recommendations": {
                "high_value_campaigns": "Use conservative threshold (0.7+)",
                "standard_campaigns": "Use balanced threshold (0.5)",
                "awareness_campaigns": "Use aggressive threshold (0.3+)",
            },
        }

    def _analyze_feature_business_impact(self, discriminative_power):
        """
        Analyze business impact of discriminative features.

        Args:
            discriminative_power (dict): Feature discriminative power scores

        Returns:
            dict: Business impact analysis
        """
        # Categorize features by business domain
        feature_categories = {
            "demographic": ["age", "education", "marital"],
            "financial": ["loan", "default", "credit"],
            "campaign": ["campaign", "contact", "previous"],
            "engineered": ["age_bin", "customer_value", "intensity", "segment"],
        }

        category_impact = {}
        for category, keywords in feature_categories.items():
            category_features = [
                (feature, power)
                for feature, power in discriminative_power.items()
                if any(keyword in feature.lower() for keyword in keywords)
            ]

            if category_features:
                avg_impact = np.mean([power for _, power in category_features])
                category_impact[category] = {
                    "average_impact": avg_impact,
                    "top_feature": max(category_features, key=lambda x: x[1]),
                    "feature_count": len(category_features),
                }

        return category_impact

    def _segment_customers_by_probability(self, probabilities):
        """
        Segment customers based on subscription probabilities.

        Args:
            probabilities (np.array): Predicted probabilities

        Returns:
            dict: Customer segmentation
        """
        positive_probs = probabilities[:, 1]

        segments = {
            "high_probability": np.sum(positive_probs >= 0.7),
            "medium_probability": np.sum(
                (positive_probs >= 0.3) & (positive_probs < 0.7)
            ),
            "low_probability": np.sum(positive_probs < 0.3),
            "total_customers": len(positive_probs),
        }

        # Calculate percentages
        total = segments["total_customers"]
        segments["percentages"] = {
            "high_probability": segments["high_probability"] / total * 100,
            "medium_probability": segments["medium_probability"] / total * 100,
            "low_probability": segments["low_probability"] / total * 100,
        }

        return segments

    def get_business_insights(self):
        """
        Generate business insights from the Naive Bayes model.

        Returns:
            dict: Business insights and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating insights")

        class_probs = self.get_class_probabilities()
        discriminative_power = self.get_feature_discriminative_power()
        marketing_insights = self.get_marketing_insights()

        insights = {
            "model_type": "Naive Bayes",
            "interpretability": "High - probabilistic foundations with clear feature contributions",
            "class_distribution": class_probs,
            "top_discriminative_features": dict(
                sorted(discriminative_power.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
            "marketing_applications": marketing_insights,
            "business_recommendations": self._generate_nb_recommendations(
                discriminative_power, class_probs
            ),
        }

        return insights

    def _generate_nb_recommendations(self, discriminative_power, class_probs):
        """
        Generate business recommendations based on Naive Bayes insights.

        Args:
            discriminative_power (dict): Feature discriminative power
            class_probs (dict): Class probabilities

        Returns:
            list: Business recommendations
        """
        recommendations = []

        # Baseline conversion rate insights
        baseline_rate = class_probs["class_priors"].get(1, 0)
        if baseline_rate < 0.15:
            recommendations.append(
                f"Low baseline conversion rate ({baseline_rate:.1%}) - focus on high-probability segments"
            )

        # Feature-based recommendations
        top_features = sorted(
            discriminative_power.items(), key=lambda x: x[1], reverse=True
        )[:3]
        for feature, power in top_features:
            if power > 1.0:  # High discriminative power
                recommendations.append(
                    f"Feature '{feature}' has strong predictive power - prioritize in targeting"
                )

        # Model-specific recommendations
        recommendations.append(
            "Use probability estimates for customer scoring and campaign prioritization"
        )
        recommendations.append(
            "Naive Bayes provides well-calibrated probabilities - suitable for ROI calculations"
        )

        return recommendations

    def __str__(self):
        """String representation with model parameters."""
        params = f"var_smoothing={self.var_smoothing}, alpha={self.alpha}"
        status = "Trained" if self.is_trained else "Not Trained"
        return f"Naive Bayes ({params}) - {status}"
