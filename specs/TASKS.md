# AIAP 20 Technical Assessment - REVISED TASKS

## Phase 1: Project Setup and Environment Preparation
- [X] Create project directory structure following the required layout
  - **Business Value:** Establish organized workspace for efficient development
  - **Data Source:** N/A (Project structure setup)
  - **Output:** Complete directory structure matching planning specifications
- [X] Initialize Git repository with clear commit messages
  - **Business Value:** Version control for reproducible development
  - **AI Context:** Repository structure that enables AI to understand project flow
- [X] Set up virtual environment using Python 3.12
  - **Business Value:** Isolated, reproducible environment
  - **Output:** Working virtual environment with dependency isolation
- [X] Install essential libraries with version pinning:
  - **Core ML:** pandas, numpy, scikit-learn (for model pipeline)
  - **Visualization:** matplotlib, seaborn, plotly (for EDA insights)
  - **Database:** sqlite3 (for data retrieval as required)
  - **Development:** black, pytest (streamlined tooling)
  - **Business Value:** Reliable dependency management for consistent results
- [X] Create `requirements.txt` with exact versions for reproducibility
  - **AI Context:** Clear dependency documentation for AI-assisted development
- [X] Configure `.gitignore` to exclude data files and artifacts
  - **Data Pipeline:** Ensure `bmarket.db` and intermediate files are excluded
- [X] Create project structure with clear data flow indicators:
  ```
  data/
  ├── raw/           # Input: bmarket.db → raw dataset
  ├── processed/     # Output: cleaned_data.csv
  └── featured/      # Output: featured_data.csv
  notebooks/         # EDA with clear data source documentation
  src/              # Modular pipeline components
  tests/            # Streamlined testing strategy
  ```

### Streamlined Testing - Phase 1
- [X] Create simplified test structure:
  - `tests/unit/` (core function verification)
  - `tests/integration/` (component interactions)
  - `tests/smoke/` (quick pipeline verification)
  - **Testing Philosophy:** Focus on critical path, not exhaustive coverage
- [X] Create basic `conftest.py` with lightweight fixtures
  - **Efficiency:** Minimal setup, maximum utility
- [X] Write smoke test for project setup verification
  - **Business Value:** Ensure development environment works correctly

## Phase 2: Data Acquisition and Understanding
- [X] Implement SQLite database connection with clear error handling
  - **Input:** `data/bmarket.db` (provided database)
  - **Business Value:** Reliable data access for marketing analysis
  - **Output:** `src/data/data_loader.py` with robust connection handling
- [X] Create data loader with explicit data source documentation
  - **Data Pipeline:** Clear indication that source is `bmarket.db`
  - **AI Context:** Well-documented functions for AI understanding
- [X] Load banking dataset and create initial data snapshot
  - **Input:** SQLite database queries
  - **Output:** `data/raw/initial_dataset.csv` (for reference and backup)
  - **Business Value:** Baseline dataset for marketing campaign optimization
- [ ] Comprehensive EDA (`eda.py`):
  - **Data Source Documentation:** Clear headers indicating `bmarket.db` as source
  - **Business Focus:** Analysis directly tied to term deposit subscription prediction
  - **Key Sections:**
    - Data overview and quality assessment
    - Target variable distribution (subscription patterns)
    - Feature relationships to business outcomes
    - Data quality issues and their business impact
    - Actionable insights for model development
  - **Output:** Complete `eda.py` with business-relevant insights.
  - **Output:** Report from `eda.py` with findings, insights and recommendations for Phase 3.
- [ ] Manual creation of EDA in Jupyter Notebook (`eda.ipynb`) from interactive exploration

### Streamlined Testing - Phase 2
- [X] Create lightweight test fixtures in `conftest.py`
  - **Efficiency:** Small sample datasets (< 100 rows) for fast testing
  - **Implementation:** Added `small_sample_dataframe`, `expected_columns`, `expected_data_types`, `data_validation_rules` fixtures
- [X] Write essential data validation tests:
  - **Smoke Test:** Database connection works ✅
  - **Data Validation:** Expected columns and basic statistics ✅
  - **Sanity Check:** Data types and value ranges ✅
  - **Testing Philosophy:** Quick verification, not exhaustive validation
  - **Files:** `tests/unit/test_data_validation.py`, `tests/smoke/test_phase2_validation.py`
  - **Results:** 25/25 tests passed in 0.09 seconds

## Phase 3: Data Cleaning and Preprocessing
*Based on EDA findings: 28,935 missing values, 12,008 special values requiring cleaning*
**Output:** `data/processed/cleaned-db.csv` (cleaned dataset ready for feature engineering)

### 3.1 Critical Data Quality Issues (EDA Priority 1)
- [ ] **Age Data Type Conversion (CRITICAL)**
  - **Issue:** Age stored as text format ('57 years', '55 years', etc.)
  - **Input:** Text-formatted age column from raw data
  - **Output:** Numeric age values (18-100 range)
  - **Implementation:** Extract numeric values using regex patterns
  - **Business Value:** Enable demographic analysis and age-based segmentation
  - **Validation:** Ensure all ages fall within 18-100 years range

- [ ] **Missing Values Handling Strategy (28,935 total missing)**
  - **Housing Loan (24,789 missing - 60.2%):** Implement domain-specific imputation
    - Create 'Information Not Available' category for business analysis
    - Consider creating missing indicator flag for model features
  - **Personal Loan (4,146 missing - 10.1%):** Apply consistent imputation strategy
    - Align with Housing Loan approach for consistency
  - **Business Rationale:** Preserve information patterns that may indicate customer behavior

- [ ] **Special Values Cleaning (12,008 total requiring attention)**
  - **'Unknown' Categories by Priority:**
    - Credit Default: 8,597 unknown values (20.9%) - HIGH PRIORITY
    - Education Level: 1,731 unknown values (4.2%) - MEDIUM PRIORITY
    - Personal Loan: 877 unknown values (2.1%) - MEDIUM PRIORITY
    - Housing Loan: 393 unknown values (1.0%) - LOW PRIORITY
    - Occupation: 330 unknown values (0.8%) - LOW PRIORITY
    - Marital Status: 80 unknown values (0.2%) - LOW PRIORITY
  - **Strategy:** Retain as distinct business category (real customer information state)
  - **Implementation:** Create consistent 'unknown' handling across all categorical features

### 3.2 Data Standardization and Consistency (EDA Priority 2)
- [ ] **Contact Method Standardization**
  - **Issue:** Inconsistent contact method values ('Cell' vs 'cellular', 'Telephone' vs 'telephone')
  - **Current Distribution:** Cell: 31.8%, cellular: 31.7%, Telephone: 18.4%, telephone: 18.1%
  - **Implementation:** Standardize to consistent casing and terminology
  - **Business Value:** Accurate contact channel analysis for campaign optimization

- [ ] **Previous Contact Days Special Value Handling**
  - **Issue:** 39,673 rows (96.3%) have '999' indicating no previous contact
  - **Implementation:** Create 'No Previous Contact' binary flag
  - **Feature Engineering:** Convert 999 to meaningful business indicator
  - **Business Value:** Clear distinction between contacted and new prospects

- [ ] **Target Variable Binary Encoding**
  - **Current:** Text values ('yes', 'no') for Subscription Status
  - **Required:** Binary encoding (1=yes, 0=no) for model compatibility
  - **Validation:** Ensure consistent encoding across entire dataset
  - **Business Value:** Standardized target for prediction models

### 3.3 Data Validation and Quality Assurance (EDA Priority 3)
- [ ] **Range Validations (Business Rule Implementation)**
  - **Age Validation:** Ensure 18-100 years (flag outliers for review)
  - **Campaign Calls Validation:** Cap extreme values at 50 (business realistic limits)
    - Current range: -41 to 56 calls (negative values need investigation)
  - **Previous Contact Days:** Validate 0-999 range consistency
  - **Implementation:** Create validation functions with clear business rules

- [ ] **Consistency Checks and Business Logic**
  - **Education-Occupation Alignment:** Validate logical relationships
  - **Loan Status Consistency:** Check Housing/Personal loan combinations
  - **Campaign Timing Constraints:** Ensure realistic contact patterns
  - **Implementation:** Business rule validation framework

- [ ] **Data Quality Metrics Implementation**
  - **Target Metrics:**
    - 0 missing values (currently: 28,935)
    - 0 unhandled special values (currently: 12,008)
    - 100% data validation pass rate
    - All features properly typed and formatted
  - **Monitoring:** Create quality dashboard for ongoing validation

### 3.4 Feature Engineering Preparation (EDA-Driven)
- [ ] **Age Group Categorization**
  - **Business Logic:** Create meaningful age segments for marketing
  - **Implementation:** Age bins based on banking customer lifecycle
  - **Output:** Categorical age groups for improved model performance

- [ ] **Campaign Intensity Features**
  - **Campaign Calls Categorization:** Low/Medium/High intensity based on distribution
  - **Contact Recency Indicators:** Recent vs. historical contact patterns
  - **Business Value:** Optimize contact frequency for subscription likelihood

- [ ] **Interaction Feature Preparation**
  - **Education-Occupation Combinations:** High-value customer segments
  - **Loan Status Interactions:** Housing + Personal loan combinations
  - **Contact Method-Age Interactions:** Channel preferences by demographics

### 3.5 Data Pipeline Documentation and Implementation
- [ ] **Comprehensive Cleaning Pipeline**
  ```python
  def clean_banking_data():
      """
      Input: data/raw/initial_dataset.csv (from bmarket.db)
      Output: data/processed/cleaned-db.csv

      EDA-Based Transformations:
      1. Age: Text to numeric conversion with validation
      2. Missing Values: 28,935 total - domain-specific imputation
      3. Special Values: 12,008 'unknown' values - business category retention
      4. Contact Methods: Standardization of inconsistent values
      5. Previous Contact: 999 → 'No Previous Contact' flag
      6. Target Variable: Binary encoding (1=yes, 0=no)

      Quality Targets:
      - Zero missing values post-processing
      - All features properly typed
      - Business rules validated

      File Format: CSV (optimal for 41K records, direct ML integration)
      """
  ```

- [ ] **Error Handling and Logging**
  - **Data Quality Alerts:** Flag unexpected values or patterns
  - **Transformation Logging:** Track all cleaning operations
  - **Business Impact Reporting:** Document cleaning decisions and rationale

### Streamlined Testing - Phase 3 (EDA-Informed)
- [ ] **Critical Data Quality Tests (Priority 1)**
  - **Age conversion verification:** Text to numeric with edge cases
    - Test cases: '57 years' → 57, invalid formats, boundary values
  - **Missing value handling validation:** 28,935 missing values strategy
    - Housing Loan (60.2% missing), Personal Loan (10.1% missing)
  - **Special value cleaning tests:** 12,008 'unknown' values handling
    - Credit Default (20.9%), Education Level (4.2%), other categories
  - **Testing Strategy:** Use EDA-identified patterns for realistic test cases

- [ ] **Data Standardization Tests (Priority 2)**
  - **Contact method standardization:** Cell/cellular, Telephone/telephone consistency
  - **Previous Contact Days handling:** 999 → 'No Previous Contact' flag validation
  - **Target variable encoding:** 'yes'/'no' → 1/0 binary conversion
  - **Focus:** Ensure standardization maintains business meaning

- [ ] **Data Validation Tests (Priority 3)**
  - **Range validation:** Age (18-100), Campaign Calls (-41 to 56 investigation)
  - **Business rule validation:** Education-Occupation consistency
  - **Quality metrics verification:** Zero missing values post-processing
  - **Pipeline integration test:** End-to-end cleaning with EDA sample data

- [ ] **Performance and Quality Assurance**
  - **Data quality metrics:** Track cleaning success rates
  - **Transformation logging:** Verify all EDA-identified issues addressed
  - **Business impact validation:** Cleaned data supports marketing analysis

## Phase 4: Data Integration and Validation (Streamlined)
- [ ] **Data Integration Module (CSV-Based)**
  - **Input:** `data/processed/cleaned-db.csv` (from Phase 3)
  - **Business Value:** Efficient data access and validation for ML pipeline
  - **Implementation:** Direct CSV operations (optimal for 41K records)
  - **Rationale:** CSV format provides best performance and simplicity for this data size

- [ ] **Data Access Functions**
  - **Load and validate cleaned data:** Ensure Phase 3 cleaning was successful
  - **Data integrity checks:** Verify all transformations completed correctly
  - **Feature validation:** Confirm data types and ranges meet ML requirements
  - **Business Value:** Reliable data foundation for feature engineering

- [ ] **Pipeline Integration Utilities**
  - **Data splitting utilities:** Prepare for train/test splits
  - **Memory optimization:** Efficient data loading for downstream processes
  - **Error handling:** Graceful handling of data access issues
  - **AI Context:** Clear, reusable functions for data pipeline integration

### Essential Testing - Phase 4
- [ ] **Data Integration and Validation Testing**
  - **Data loading verification:** CSV file loads correctly with expected schema
  - **Data integrity tests:** All Phase 3 transformations preserved
  - **Performance tests:** Loading time acceptable for 41K records
  - **Error handling tests:** Graceful handling of missing or corrupted files
  - **Focus:** Reliable data access for downstream ML pipeline

## Phase 5: Feature Engineering with Business Context
- [ ] Create feature engineering module with clear business rationale
  - **Input:** `data/processed/cleaned-db.csv` (from Phase 3)
  - **Output:** `data/featured/featured-db.csv`
  - **Business Value:** Features that directly impact subscription prediction accuracy
- [ ] **Business-Driven Feature Creation:**
  - **Age binning:** Numerical age categories (1=young, 2=middle, 3=senior) for optimal model performance
  - **Education-occupation interactions:** High-value customer segments
  - **Contact recency:** Recent contact effect on subscription likelihood
  - **Campaign intensity:** Optimal contact frequency patterns
  - **Business Rationale:** Each feature tied to marketing strategy insights
- [ ] Feature transformations with clear purpose:
  - **Scaling:** Standardization for model performance
  - **Encoding:** One-hot encoding for categorical variables
  - **Dimensionality:** PCA if needed for computational efficiency
- [ ] **Data Pipeline Documentation:**
  ```python
  def engineer_features():
      """
      Input: data/processed/cleaned-db.csv (from Phase 3)
      Output: data/featured/featured-db.csv
      Business Purpose: Create features for subscription prediction
      Key Features: age_bin, education_job_segment, recent_contact_flag
      """
  ```

### Focused Testing - Phase 5
- [ ] Feature transformation verification:
  - **Sample-based testing:** Small datasets with known transformations
  - **Business validation:** Features make intuitive sense
  - **Integration test:** End-to-end feature pipeline

## Phase 6: Model Preparation (Streamlined)
- [ ] Essential model preparation components:
  - **Train/test splitting:** Stratified split preserving class distribution
  - **Cross-validation:** 5-fold stratified for reliable performance estimation
  - **Evaluation metrics:** Focus on business-relevant metrics (precision, recall, ROI)
  - **Business Value:** Robust model evaluation for marketing decision confidence
- [ ] Model utilities with clear interfaces:
  - **Model saving/loading:** Simple serialization for deployment
  - **Factory pattern:** Easy model selection and comparison
  - **AI Context:** Clear, extensible architecture for AI-assisted development

### Essential Testing - Phase 6
- [ ] Core functionality verification:
  - **Data splitting:** Stratification preservation
  - **Metrics calculation:** Known input/output validation
  - **Model serialization:** Save/load consistency

## Phase 7: Model Implementation (MVP-First Approach)
- [ ] **Working End-to-End Pipeline First:**
  - **Input:** `data/featured/featured_data.csv`
  - **Output:** Trained models with performance metrics
  - **Business Value:** Immediate feedback on subscription prediction capability
- [ ] Implement 5 classifiers with business justification:
  - **Logistic Regression (classifier1.py):** Interpretable baseline for marketing insights and coefficient analysis
  - **Random Forest (classifier2.py):** Robust performance with feature importance and handles mixed data types well
  - **Gradient Boosting(XGBoost) (classifier3.py):** High-performance gradient boosting for complex patterns
  - **Naive Bayes (classifier4.py):** Efficient probabilistic classifier, excellent for marketing probability estimates
  - **Support Vector Machine (classifier5.py):** Strong performance on structured data with clear decision boundaries
  - **Business Focus:** Models that provide actionable insights and handle the categorical-heavy banking dataset effectively
- [ ] **Model Training Pipeline:**
  ```python
  def train_model():
      """
      Input: data/featured/featured_data.csv
      Output: trained_models/model_v1.pkl + performance_metrics.json
      Business Purpose: Predict term deposit subscription likelihood
      """
  ```
- [ ] Essential model capabilities:
  - **Training:** Robust fitting with cross-validation
  - **Prediction:** Reliable inference pipeline
  - **Evaluation:** Business-relevant metrics
  - **Interpretability:** Feature importance for marketing insights

### Pragmatic Testing - Phase 7
- [ ] Model functionality verification:
  - **Smoke tests:** Models train and predict without errors
  - **Sanity checks:** Predictions in expected range [0,1]
  - **Performance baseline:** Models beat random guessing
  - **Testing Strategy:** Small synthetic datasets for speed

## Phase 8: Model Evaluation (Business-Focused)
- [ ] **Business-Relevant Evaluation:**
  - **Marketing ROI:** Expected value of targeted campaigns
  - **Precision/Recall trade-offs:** Balancing contact efficiency vs. coverage
  - **Threshold optimization:** Optimal decision boundaries for business outcomes
  - **Business Value:** Clear guidance for marketing strategy implementation
- [ ] Model comparison framework:
  - **Performance metrics:** Accuracy, precision, recall, F1, ROC-AUC
  - **Business metrics:** Expected lift, cost per acquisition
  - **Interpretability:** Feature importance rankings
- [ ] **Evaluation Pipeline:**
  ```python
  def evaluate_models():
      """
      Input: trained_models/ + data/featured/test_data.csv
      Output: model_evaluation_report.json + visualizations/
      Business Purpose: Select best model for marketing campaign optimization
      """
  ```

### Essential Testing - Phase 8
- [ ] Evaluation pipeline verification:
  - **Metrics calculation:** Correct computation with known datasets
  - **Comparison logic:** Proper model ranking
  - **Visualization generation:** Key charts render correctly

## Phase 9: Model Selection and Optimization (Results-Driven)
- [ ] **Business-Driven Model Selection:**
  - **Primary criterion:** Marketing ROI and business impact
  - **Secondary criteria:** Model interpretability and deployment feasibility
  - **Output:** Selected model with clear business justification
- [ ] Hyperparameter optimization:
  - **Approach:** Bayesian optimization for efficiency
  - **Focus:** Business-relevant metrics, not just accuracy
  - **Business Value:** Optimal model performance for marketing campaigns
- [ ] Model interpretability analysis:
  - **Feature importance:** Which customer attributes drive subscriptions
  - **Business insights:** Actionable recommendations for marketing strategy

### Focused Testing - Phase 9
- [ ] Selection and optimization verification:
  - **Selection criteria:** Business metrics properly calculated
  - **Optimization:** Hyperparameter tuning improves performance
  - **Interpretability:** Feature importance makes business sense

## Phase 10: Pipeline Integration (Deployment-Ready)
- [ ] **End-to-End Pipeline Implementation:**
  - **Input:** Raw data from database
  - **Output:** Subscription predictions with confidence scores
  - **Business Value:** Production-ready system for marketing campaigns
- [ ] Create main.py with clear execution flow:
  ```python
  def main():
      """
      Complete ML Pipeline: bmarket.db → subscription_predictions.csv
      Business Purpose: Automated term deposit prediction for marketing
      """
  ```
- [ ] Robust error handling and logging:
  - **Business Focus:** Clear error messages for operational teams
  - **Monitoring:** Key metrics for production monitoring
- [ ] Create run.sh for simple execution:
  - **One-command deployment:** Minimal setup for production use

### Integration Testing - Phase 10
- [ ] End-to-end pipeline testing:
  - **Full workflow:** Database to predictions
  - **Error handling:** Graceful failure modes
  - **Performance:** Acceptable execution time for business needs

## Phase 11: Documentation (Business-Focused)
- [ ] **Comprehensive README.md:**
  - **Business Context:** Clear problem statement and solution value
  - **Quick Start:** Simple instructions for immediate use
  - **Pipeline Flow:** Clear data transformation documentation
  - **Model Selection:** Business rationale for chosen approach
  - **Performance Results:** Business-relevant metrics and insights
- [ ] **Data Pipeline Documentation:**
  ```markdown
  ## Data Flow
  1. bmarket.db → data/raw/initial_dataset.csv (SQLite extraction)
  2. data/raw/ → data/processed/cleaned-db.csv (cleaning pipeline)
  3. data/processed/ → data/featured/featured_data.csv (feature engineering)
  4. data/featured/ → trained_models/ (model training)
  5. trained_models/ → predictions/ (inference pipeline)

  Format Decision: CSV-based pipeline (optimal for 41K records)
  ```
- [ ] Business impact documentation:
  - **Expected ROI:** Quantified marketing improvement
  - **Implementation guidance:** How to use predictions operationally
  - **Future enhancements:** Clear roadmap for improvements

### Documentation Testing - Phase 11
- [ ] Documentation validation:
  - **Examples work:** All code examples execute correctly
  - **Instructions complete:** Setup and execution steps are sufficient
  - **Business clarity:** Non-technical stakeholders can understand value