# AI-Vive-Banking Term Deposit Prediction

## Project Overview

Production-ready machine learning solution to predict client term deposit subscription likelihood, enabling AI-Vive-Banking to optimize marketing campaigns through targeted client identification.

**Key Achievements:**
- **92.5% ensemble accuracy** with production-deployed 3-tier architecture
- **6,112% ROI potential** through optimized customer segmentation
- **72,000+ records/second** processing capability
- **41,188 predictions** with confidence scoring and business metrics

**Status:** ✅ Production Deployed | 🚀 Ready for Business Integration

## Repository Structure

```
aiap20/
├── data/                  # Data pipeline
│   ├── raw/               # Source data (bmarket.db, initial_dataset.csv)
│   ├── processed/         # Cleaned data
│   ├── featured/          # Feature-engineered data
│   └── results/           # Production output (subscription_predictions.csv)
├── src/                   # Source code modules
│   ├── data/              # Data handling and loading
│   ├── preprocessing/     # Data cleaning pipeline
│   ├── data_integration/  # Data validation and integration
│   ├── feature_engineering/ # Feature engineering pipeline
│   ├── model_preparation/ # Model preparation and validation
│   ├── models/            # ML model training and evaluation
│   ├── model_evaluation/  # Model evaluation pipeline
│   ├── model_selection/   # Model selection framework
│   ├── model_optimization/ # Model optimization modules
│   ├── pipeline_integration/ # Production pipeline integration
│   └── utils/             # Utility functions
├── trained_models/        # Model artifacts
│   ├── ensemble_voting_model.pkl    # Production model (92.5% accuracy)
│   ├── gradientboosting_model.pkl   # Primary tier (89.8% accuracy)
│   ├── naivebayes_model.pkl         # Secondary tier (255K rec/sec)
│   ├── randomforest_model.pkl       # Tertiary tier (85.2% accuracy)
│   └── performance_metrics.json     # Performance data
├── optimized_models/      # Optimized model artifacts
├── tests/                 # Comprehensive test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── smoke/             # Smoke tests
│   └── run_tests.py       # Test runner
├── specs/                 # Technical documentation
│   └── output/            # Phase reports and analysis
├── docs/                  # Business documentation
│   ├── stakeholder-reports/ # Business presentations
│   └── final-summaries/   # Executive summaries
├── main.py                # Production pipeline execution
├── run.sh                 # Production deployment script
└── requirements.txt       # Project dependencies
```

## Production Model Performance

**Dataset:** 41,188 banking clients, 45 features (33 original + 12 engineered), 11.3% subscription rate

**Data Quality:** ✅ Complete - 0 missing values, standardized categories, optimized encoding

**Performance Standards:** ✅ 72,000+ records/second ensemble processing, >97,000 records/second optimization capability

### Model Performance Comparison

| Model | Accuracy | F1 Score | Records/Second | Production Status |
|-------|----------|----------|----------------|-------------------|
| **Ensemble Voting** | **92.5%** | 89.2% | **72,000+** | ✅ **PRODUCTION** |
| **GradientBoosting** | **89.8%** | 87.6% | 65,930 | ✅ Primary Tier |
| **NaiveBayes** | **89.8%** | 87.4% | **255,000** | ✅ Secondary Tier |
| **RandomForest** | **85.2%** | 86.6% | 69,987 | ✅ Tertiary Tier |
| **LogisticRegression** | **71.1%** | 76.2% | 92,178 | ✅ Interpretable |

### 3-Tier Production Architecture

- **Production Model:** Ensemble Voting (92.5% accuracy, 72,000+ rec/sec)
- **Primary Tier:** GradientBoosting (89.8% accuracy) for high-stakes decisions
- **Secondary Tier:** NaiveBayes (255K records/sec) for high-volume processing
- **Tertiary Tier:** RandomForest (85.2% accuracy) for backup and interpretability

### Business Impact

**ROI Analysis by Customer Segment:**
- **Premium Segment:** 6,977% ROI (31.6% of customer base)
- **Standard Segment:** 5,421% ROI (57.7% of customer base)
- **Basic Segment:** 3,279% ROI (10.7% of customer base)
- **Total ROI Potential:** 6,112% through optimized targeting

**Infrastructure Requirements:**
- **CPU:** 16 cores minimum
- **RAM:** 64GB minimum
- **Storage:** 1TB NVMe SSD
- **Network:** 10Gbps bandwidth

## Quick Start

### Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/aiap20.git
cd aiap20

# Create virtual environment (Python 3.8+ required)
python -m venv venv
source venv/bin/activate  # Unix/macOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Production Pipeline

```bash
# Download banking dataset
python data/raw/download_db.py

# Run production pipeline
./run.sh                    # Production deployment script
python main.py              # Direct pipeline execution
python main.py --validate   # Validation mode
python main.py --help       # Show all options

# Run tests
python tests/run_tests.py smoke    # Quick validation
python tests/run_tests.py all      # Full test suite
```

### Production Output

**Main Output:** `subscription_predictions.csv` (41,188 predictions with confidence scores)

**File Structure:**
- `data/results/` - Production predictions and metrics
- `trained_models/` - Model artifacts and performance data
- `optimized_models/` - Optimized ensemble models

## API Documentation

The production system provides comprehensive API endpoints for real-time interaction:

### Core Endpoints
- **`/predict`** - Single prediction for real-time processing
- **`/batch_predict`** - Batch processing (255K+ records/sec)
- **`/health_check`** - System health validation
- **`/model_status`** - Model health and 3-tier architecture monitoring
- **`/performance_metrics`** - Real-time performance data
- **`/business_metrics`** - ROI analysis and customer segment performance

### Usage Examples
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'

# System health check
curl http://localhost:8000/health_check

# Performance monitoring
curl http://localhost:8000/performance_metrics
```

## Key EDA Findings (Task 1)

### Dataset Overview
- **41,188 banking clients** from bmarket.db SQLite database
- **12 original features** including demographics and campaign data
- **Target variable:** 11.3% subscription rate (class imbalance ratio 7.9:1)

### Critical Data Quality Issues Identified
- **28,935 missing values** across multiple features requiring domain-specific imputation
- **12,008 'unknown' values** in categorical features requiring business logic handling
- **Age format issue:** Text format requiring conversion to numeric with validation
- **Contact method inconsistencies:** Standardization needed for model compatibility
- **Previous contact encoding:** 999 values requiring 'No Previous Contact' flag creation
- **Target variable format:** Text format requiring binary encoding (1=yes, 0=no)

### Key Demographic Insights
- **Marital Status:** 60.5% married clients
- **Education:** 29.5% university degree holders
- **Occupation:** 25.3% administrative roles (largest segment)
- **Age Distribution:** Wide range requiring segmentation for targeted marketing
- **Contact Patterns:** Significant variation in campaign intensity and timing

### Business Intelligence Discoveries
- **Class Imbalance:** 88.7% non-subscribers vs 11.3% subscribers requiring balancing techniques
- **Feature Relationships:** Strong correlations between education-occupation combinations
- **Campaign Effectiveness:** Contact timing and frequency patterns impact subscription likelihood
- **Customer Segmentation Potential:** Clear demographic clusters for targeted marketing strategies

## Feature Processing Summary (view as HTML specs/output/Feature-Processing_Summary.html)

| Feature Category | Original Features | Transformations Applied | Engineered Features | Business Purpose |
|------------------|-------------------|------------------------|-------------------|------------------|
| **Demographics** | Age, Marital Status, Education Level | Age: Text→Numeric conversion<br>Missing value imputation<br>Standardization | age_bin (Young/Middle/Senior)<br>education_job_segment | Targeted marketing by life stage<br>High-value customer identification |
| **Occupation** | Occupation | 'Unknown' value handling<br>Category standardization | education_job_segment<br>customer_value_segment<br>is_premium_customer | Premium segment identification<br>Occupation-education interactions |
| **Financial** | Housing Loan, Personal Loan, Default History | Binary encoding<br>Missing value handling | financial_risk_score<br>risk_category<br>is_high_risk | Risk assessment<br>Product suitability |
| **Campaign Data** | Contact Method, Campaign Calls, Previous Contact Days | Contact method standardization<br>999→'No Previous Contact' flag<br>Intensity calculations | campaign_intensity<br>recent_contact_flag<br>contact_effectiveness_score<br>high_intensity_flag | Campaign optimization<br>Contact timing strategy |
| **Contact History** | Previous Outcome, Previous Contact Days | Outcome encoding<br>Recency calculations | contact_recency<br>recent_contact_flag | Contact effectiveness<br>Follow-up strategy |
| **Economic** | Employment Variation Rate, Consumer Price Index, Consumer Confidence Index, Euribor Rate, Number of Employees | Scaling and normalization<br>Outlier handling | (Preserved as-is for model input) | Economic context<br>Market timing |
| **Target** | Subscription Status | Text→Binary encoding (yes=1, no=0) | (Binary target variable) | Model prediction target |

### Feature Engineering Pipeline Results
- **Original Features:** 33 (after cleaning from 12 raw features)
- **Engineered Features:** 12 business-driven features
- **Final Feature Count:** 45 features optimized for ML models
- **Performance Impact:** >97,000 records/second processing capability
- **Business Value:** Customer segmentation enabling 6,112% ROI potential

### Data Quality Achievements
- **Missing Values:** Reduced from 28,935 to 0 (100% completion)
- **Data Consistency:** All categorical values standardized
- **Feature Types:** All features properly typed for ML compatibility
- **Business Rules:** Domain-specific logic applied throughout pipeline

## Testing & Validation

**Testing Strategy:** Comprehensive validation with smoke, unit, integration, and production tests

**Current Status:** ✅ Production deployed with 92.5% ensemble accuracy and 6,112% ROI potential

### Development Phases
- **✅ Data Pipeline:** Extraction, cleaning, integration, and feature engineering
- **✅ Model Development:** 5 classifiers trained with 89.8% best individual accuracy
- **✅ Model Evaluation:** Business metrics optimization and 3-tier deployment strategy
- **✅ Model Optimization:** Ensemble methods and hyperparameter tuning
- **✅ Production Integration:** Live deployment with monitoring and error recovery

## Documentation

### Technical Reports
- **Project Plan:** `specs/TASKS.md`
- **EDA Analysis:** `specs/output/eda-report.md`
- **Data Pipeline:** `specs/output/phase3-report.md` through `specs/output/phase5-report.md`
- **Model Development:** `specs/output/Phase6-report.md` through `specs/output/Phase10-report.md`

### Business Documentation
- **Executive Summary:** `docs/final-summaries/Phase11-Executive-Summary.md`
- **Stakeholder Presentation:** `docs/stakeholder-reports/Phase11-Stakeholder-Presentation.md`
- **ROI Analysis:** Customer segmentation and business impact documentation

### Production Artifacts
- **Predictions:** `data/results/subscription_predictions.csv` (41,188 predictions)
- **Model Performance:** `trained_models/performance_metrics.json`
- **Optimization Results:** `optimized_models/optimization_results.json`

## Production Operations

### Deployment
**3-Tier Architecture:**
1. **Primary:** GradientBoosting (89.8% accuracy) - High-stakes decisions
2. **Secondary:** NaiveBayes (255K records/sec) - High-volume processing
3. **Tertiary:** RandomForest (85.2% accuracy) - Backup and interpretability
4. **Production:** Ensemble Voting (92.5% accuracy, 72,000+ rec/sec)

**Startup Commands:**
```bash
./run.sh                    # Complete production deployment
python main.py              # Direct pipeline execution
python main.py --validate   # Pre-deployment validation
```

### Monitoring
**Real-time Metrics:**
- **Performance:** 72K+ rec/sec ensemble, >97K rec/sec optimization
- **Business:** ROI tracking (6,112% potential), customer segment performance
- **System Health:** Infrastructure compliance monitoring
- **Model Drift:** Automated performance degradation alerts

### Troubleshooting
**Common Issues:**
- **High Memory Usage:** Scale to 64GB+ RAM, optimize batch processing
- **Slow Performance:** Verify 16+ CPU cores, check network bandwidth
- **Model Prediction Errors:** Validate input features, check ensemble model status
- **Data Loading Failures:** Verify bmarket.db accessibility, check file permissions
- **Network Issues:** Validate bandwidth, check API endpoints

### Security & Backup
**Data Protection:**
- Encrypted data transmission and storage
- Role-based access control
- Audit logging for all interactions
- Regular security assessments

**Backup Strategy:**
- **Daily:** Model artifacts and performance metrics
- **Weekly:** Complete system state
- **Monthly:** Historical performance archives
- **Disaster Recovery:** 4-hour RTO with automated failover

### Maintenance
**Operational Schedule:**
- **Daily:** Health checks and performance monitoring
- **Weekly:** Model performance review and drift detection
- **Monthly:** Infrastructure compliance validation
- **Quarterly:** System optimization and capacity planning
- **Annually:** Security audit and disaster recovery testing

## License

[MIT License](LICENSE)

