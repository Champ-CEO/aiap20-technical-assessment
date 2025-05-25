# AIAP 20 Technical Assessment - REVISED TASKS

## Phase 1: Project Setup and Environment Preparation
- [ ] Create project directory structure following the required layout
  - **Business Value:** Establish organized workspace for efficient development
  - **Data Source:** N/A (Project structure setup)
  - **Output:** Complete directory structure matching planning specifications
- [ ] Initialize Git repository with clear commit messages
  - **Business Value:** Version control for reproducible development
  - **AI Context:** Repository structure that enables AI to understand project flow
- [ ] Set up virtual environment using Python 3.12
  - **Business Value:** Isolated, reproducible environment
  - **Output:** Working virtual environment with dependency isolation
- [ ] Install essential libraries with version pinning:
  - **Core ML:** pandas, numpy, scikit-learn (for model pipeline)
  - **Visualization:** matplotlib, seaborn, plotly (for EDA insights)
  - **Database:** sqlite3 (for data retrieval as required)
  - **Development:** black, pytest (streamlined tooling)
  - **Business Value:** Reliable dependency management for consistent results
- [ ] Create `requirements.txt` with exact versions for reproducibility
  - **AI Context:** Clear dependency documentation for AI-assisted development
- [ ] Configure `.gitignore` to exclude data files and artifacts
  - **Data Pipeline:** Ensure `bmarket.db` and intermediate files are excluded
- [ ] Create project structure with clear data flow indicators:
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
- [ ] Create simplified test structure:
  - `tests/unit/` (core function verification)
  - `tests/integration/` (component interactions)
  - `tests/smoke/` (quick pipeline verification)
  - **Testing Philosophy:** Focus on critical path, not exhaustive coverage
- [ ] Create basic `conftest.py` with lightweight fixtures
  - **Efficiency:** Minimal setup, maximum utility
- [ ] Write smoke test for project setup verification
  - **Business Value:** Ensure development environment works correctly

## Phase 2: Data Acquisition and Understanding
- [ ] Implement SQLite database connection with clear error handling
  - **Input:** `data/bmarket.db` (provided database)
  - **Business Value:** Reliable data access for marketing analysis
  - **Output:** `src/data/data_loader.py` with robust connection handling
- [ ] Create data loader with explicit data source documentation
  - **Data Pipeline:** Clear indication that source is `bmarket.db`
  - **AI Context:** Well-documented functions for AI understanding
- [ ] Load banking dataset and create initial data snapshot
  - **Input:** SQLite database queries
  - **Output:** `data/raw/initial_dataset.csv` (for reference and backup)
  - **Business Value:** Baseline dataset for marketing campaign optimization
- [ ] Comprehensive EDA in Jupyter Notebook (`eda.ipynb`):
  - **Data Source Documentation:** Clear headers indicating `bmarket.db` as source
  - **Business Focus:** Analysis directly tied to term deposit subscription prediction
  - **Key Sections:**
    - Data overview and quality assessment
    - Target variable distribution (subscription patterns)
    - Feature relationships to business outcomes
    - Data quality issues and their business impact
    - Actionable insights for model development
  - **Output:** Complete `eda.ipynb` with business-relevant insights

### Streamlined Testing - Phase 2
- [ ] Create lightweight test fixtures in `conftest.py`
  - **Efficiency:** Small sample datasets (< 100 rows) for fast testing
- [ ] Write essential data validation tests:
  - **Smoke Test:** Database connection works
  - **Data Validation:** Expected columns and basic statistics
  - **Sanity Check:** Data types and value ranges
  - **Testing Philosophy:** Quick verification, not exhaustive validation

## Phase 3: Data Cleaning and Preprocessing
- [ ] Create data cleaning module with clear business rationale
  - **Input:** `data/raw/initial_dataset.csv`
  - **Output:** `data/processed/cleaned_data.csv`
  - **Business Value:** Clean data improves model reliability for marketing decisions
- [ ] Handle missing values with business-justified approaches:
  - **Housing/Personal Loans (NULLs):** Create 'Information Not Available' category
  - **'Unknown' values:** Retain as distinct business category (real customer state)
  - **Business Rationale:** Preserve information that may indicate customer behavior patterns
- [ ] Handle data quality issues:
  - **Age conversion:** TEXT to numeric (critical for demographic analysis)
  - **Contact Method standardization:** Resolve 'Cell'/'cellular' inconsistencies
  - **Outlier treatment:** Cap extreme campaign call values (business realistic limits)
  - **Special values:** Handle 999 in Previous Contact Days as 'no previous contact'
- [ ] Implement data validation with business rules:
  - Age within reasonable bounds (18-100)
  - Campaign calls within operational limits
  - Subscription status properly encoded
- [ ] **Data Pipeline Documentation:**
  ```python
  # Clear data flow indicators in each function
  def clean_data():
      """
      Input: data/raw/initial_dataset.csv (from bmarket.db)
      Output: data/processed/cleaned_data.csv
      Transformations: Age conversion, contact standardization, missing value handling
      """
  ```

### Pragmatic Testing - Phase 3
- [ ] Essential cleaning function tests:
  - **Age conversion verification** with sample data
  - **Contact method standardization** with known inputs
  - **Missing value handling** with synthetic examples
  - **Testing Strategy:** Use small, representative datasets for speed
- [ ] Integration test for cleaning pipeline:
  - **Input:** Small sample of raw data
  - **Output:** Verified cleaned data properties
  - **Focus:** Critical transformations work correctly

## Phase 4: Database Integration (Simplified)
- [ ] Create streamlined database operations
  - **Business Value:** Efficient data storage and retrieval for pipeline
  - **Implementation:** Focus on essential operations only
- [ ] Database schema for processed data
  - **Input:** `data/processed/cleaned_data.csv`
  - **Output:** SQLite tables with proper indexing
- [ ] Query functions for model pipeline
  - **AI Context:** Clear, reusable functions for data extraction

### Essential Testing - Phase 4
- [ ] Database connection and basic operations testing
  - **Smoke Tests:** Connection, insertion, querying
  - **Focus:** Core functionality verification only

## Phase 5: Feature Engineering with Business Context
- [ ] Create feature engineering module with clear business rationale
  - **Input:** `data/processed/cleaned_data.csv`
  - **Output:** `data/featured/featured_data.csv`
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
      Input: data/processed/cleaned_data.csv
      Output: data/featured/featured_data.csv
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
  2. data/raw/ → data/processed/cleaned_data.csv (cleaning pipeline)
  3. data/processed/ → data/featured/featured_data.csv (feature engineering)
  4. data/featured/ → trained_models/ (model training)
  5. trained_models/ → predictions/ (inference pipeline)
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