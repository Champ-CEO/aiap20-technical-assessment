# Trained Models Directory

## Overview
This directory contains Phase 7 trained model artifacts and performance metrics.

## Model Files

### Individual Classifiers
- **gradientboosting_model.pkl** - Primary model (89.8% accuracy, 10,022 rec/sec)
- **naivebayes_model.pkl** - Secondary model (89.5% accuracy, 255,095 rec/sec)
- **randomforest_model.pkl** - Tertiary model (84.6% accuracy, 60,163 rec/sec)
- **logisticregression_model.pkl** - Interpretable model (71.5% accuracy, 17,064 rec/sec)
- **svm_model.pkl** - Support vector model (78.8% accuracy, 157 rec/sec)

### Ensemble Model
**Note:** Ensemble models are not currently generated. To create ensemble models, you would need to:

1. **For Phase 7 ensemble model:**
   ```bash
   python -c "from src.models.train_model import ModelTrainer; trainer = ModelTrainer(); trainer.train_ensemble_model()"
   ```

2. **For Phase 9 optimized ensemble model:**
   ```bash
   python -c "from src.model_optimization.ensemble_optimizer import EnsembleOptimizer; optimizer = EnsembleOptimizer(); optimizer.optimize_ensemble()"
   ```

## Performance Metrics

### model_performance.json
Comprehensive performance data for all trained models including:
- Training and validation metrics (accuracy, precision, recall, F1, AUC)
- Processing speed (records/second) and training time
- Performance standard compliance (>97K rec/sec target)
- Model comparison and status

## Model Performance Summary

| Model | Test Accuracy | F1 Score | Records/Second | Performance Standard | Status |
|-------|---------------|----------|----------------|---------------------|--------|
| **GradientBoosting** | **89.8%** | 87.3% | 10,022 | ❌ (10.3% of target) | ✅ Best Accuracy |
| **NaiveBayes** | **89.5%** | 87.1% | **255,095** | ✅ (263% of target) | ✅ Speed Champion |
| **RandomForest** | **84.6%** | 86.0% | 60,163 | ❌ (62% of target) | ✅ Balanced |
| **LogisticRegression** | **71.5%** | 76.5% | 17,064 | ❌ (18% of target) | ✅ Interpretable |
| **SVM** | **78.8%** | 81.8% | 157 | ❌ (0.2% of target) | ❌ Too Slow |

**Note:** Only NaiveBayes meets the >97K records/second performance standard.

## Production Recommendations

Based on actual performance metrics:

### **Recommended Architecture:**
- **Primary Model:** GradientBoosting (89.8% accuracy) - Best accuracy for critical decisions
- **High-Volume Model:** NaiveBayes (89.5% accuracy, 255K rec/sec) - Only model meeting speed requirements
- **Backup Model:** RandomForest (84.6% accuracy) - Good balance of accuracy and interpretability

### **Performance Reality:**
- **Speed Standard:** >97K records/second target
- **Models Meeting Standard:** 1 out of 5 (NaiveBayes only)
- **Best Accuracy:** GradientBoosting (89.8%)
- **Speed Champion:** NaiveBayes (255,095 rec/sec)

### **Business Considerations:**
- Ensemble models would need to be generated to achieve higher accuracy
- Current individual models provide good baseline performance
- NaiveBayes is the only production-ready model for high-volume scenarios
