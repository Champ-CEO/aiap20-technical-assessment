# AIAP 20 Technical Assessment - PLANNING

## Project Overview

This document outlines the comprehensive plan for approaching the AI-Vive-Banking Term Deposit Prediction project as part of the AIAP Batch 20 Technical Assessment. The assessment consists of two main tasks:

1. **Exploratory Data Analysis (EDA)** in a Jupyter Notebook
2. **End-to-end Machine Learning Pipeline (MLP)** in Python scripts

The goal is to predict the likelihood of a client subscribing to a term deposit based on their information and data from direct marketing campaigns. This will help AI-Vive-Banking optimize its marketing strategies by identifying clients who are most likely to respond positively.

## Table of Contents

- [Project Timeline](#project-timeline)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Task 1: Exploratory Data Analysis](#task-1-exploratory-data-analysis)
- [Task 2: Machine Learning Pipeline](#task-2-machine-learning-pipeline)
- [Development Plan](#development-plan)
- [Testing Strategy](#testing-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Documentation Requirements](#documentation-requirements)
- [Submission Checklist](#submission-checklist)

## Project Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Environment Setup & Project Structure | 1 day | Pending |
| 2 | Data Acquisition & Initial Exploration | 1 day | Pending |
| 3 | Exploratory Data Analysis (Task 1) | 2-3 days | Pending |
| 4 | Design ML Pipeline Architecture | 1 day | Pending |
| 5 | Implement ML Pipeline Components | 3-4 days | Pending |
| 6 | Integrate & Test End-to-End Pipeline | 1-2 days | Pending |
| 7 | Documentation & README Preparation | 1 day | Pending |
| 8 | Final Review & Submission | 1 day | Pending |

## Project Structure

banking-term-deposit-prediction/
├── .github/                          # GitHub Actions (provided in template)
├── data/                             # Data directory
├── notebooks/
│   └── eda.ipynb                     # Task 1 - Exploratory Data Analysis
├── src/                              # Task 2 - Machine Learning Pipeline
│   ├── __init__.py
│   ├── config/                       # Configuration files
│   │   └── config.yaml               # Pipeline configuration
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py            # Database connection and queries
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Data cleaning
│   │   └── transformer.py            # Feature transformations
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── engineer.py               # Feature creation
│   │   └── selector.py               # Feature selection
│   ├── models/                       # ML models
│   │   ├── __init__.py
│   │   ├── base.py                   # Base model class
│   │   ├── classifier1.py            # First model implementation: Logistic Regression
│   │   ├── classifier2.py            # Second model implementation: Random Forest
│   │   └── classifier3.py            # Third model implementation: Gradient Boosting (XGBoost)
│   │   └── classifier4.py            # Fourth model implementation: Naive Bayes
   │    └── classifier5.py            # Fifth model implementation: Support Vector Machine
│   ├── evaluation/                   # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py                # Performance metrics
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       └── logger.py                 # Logging functionality
├── tests/                            # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py              # Test for pipeline components
├── main.py                           # Main entry point for pipeline
├── requirements.txt                  # Dependencies
├── run.sh                            # Execution script
└── README.md                         # Project documentation
```

## Technology Stack

Based on the technical context document, we'll use:

### Core Technologies
- **Python 3.12** - Primary development language
- **SQLite** - Database handling
- **Jupyter Notebooks** - For EDA

### Key Libraries
- **Data Processing**
  - pandas
  - numpy
  - sqlite3

- **Machine Learning**
  - scikit-learn
  - XGBoost (optional)
  - LightGBM (optional)

- **Visualization**
  - matplotlib
  - seaborn
  - plotly

### Development Tools
- **Version Control** - Git
- **Code Quality** - black, pylint
- **Testing** - pytest

## Task 1: Exploratory Data Analysis

The EDA should be conducted in a Jupyter Notebook (`eda.ipynb`) following these steps:

### EDA Plan

1. **Data Acquisition**
   - Connect to SQLite database
   - Load data using SQLAlchemy
   - Display initial information (shape, dtypes, etc.)

2. **Data Understanding**
   - Examine dataset structure
   - Check for missing values
   - Analyze data types and distributions
   - Generate descriptive statistics

3. **Data Cleaning**
   - Handle missing values
   - Identify and address outliers
   - Correct data types as needed
   - Identify and handle synthetic/contaminated data

4. **Univariate Analysis**
   - Distribution of target variable (Subscription Status)
   - Distribution of numerical features
   - Distribution of categorical features
   - Identify potential data imbalance

5. **Bivariate Analysis**
   - Correlation between features
   - Target vs. feature relationships
   - Identify key patterns and trends

6. **Multivariate Analysis**
   - Multi-feature interactions
   - Feature grouping patterns
   - Advanced relationship analysis

7. **Feature Engineering Ideas**
   - Identify potential new features
   - Test transformation techniques
   - Evaluate encoding strategies

8. **Key Findings and Insights**
   - Summarize important patterns
   - Document potential model approaches
   - Note data challenges and solutions

Each section should include:
- Clear explanations of the purpose
- Visual representations of the data
- Interpretations of the statistics
- Implications for modeling

## Task 2: Machine Learning Pipeline

The ML pipeline should be implemented as modular Python scripts following a logical flow:

### Pipeline Architecture

1. **Data Layer**
   - Database connection
   - Query execution
   - Data retrieval

2. **Processing Layer**
   - Data cleaning
   - Feature transformation
   - Data validation

3. **Model Layer**
   - Model selection
   - Training
   - Evaluation

4. **Output Layer**
   - Prediction generation
   - Results reporting
   - Visualization

### Implementation Plan

1. **Configuration System**
   - Create a YAML-based config system
   - Support command-line parameters
   - Enable easy experiment configuration

2. **Data Loading Module**
   - Implement SQLite connection
   - Create data retrieval functions
   - Implement data chunking for large datasets

3. **Preprocessing Module**
   - Implement cleaning functions
   - Create transformation pipelines
   - Build validation checks

4. **Feature Engineering Module**
   - Create feature generation functions
   - Implement feature selection methods
   - Build feature evaluation metrics

5. **Model Building Module**
   - Create base model class
   - Implement at least 3 different classifiers
   - Support hyperparameter tuning

6. **Evaluation Module**
   - Implement performance metrics
   - Create model comparison tools
   - Generate evaluation reports

7. **Pipeline Orchestration**
   - Create main pipeline flow
   - Implement error handling
   - Support logging and monitoring

### Design Patterns to Implement

Based on the System Patterns document, consider implementing:

1. **Factory Method** for model creation
2. **Strategy Pattern** for feature selection
3. **Facade Pattern** for pipeline simplification

## Development Plan

### Phase 1: Setup & Data Acquisition

1. Set up project structure
2. Install required dependencies
3. Create the GitHub repository
4. Setup SQLite connection
5. Download and inspect the dataset

### Phase 2: EDA Development

1. Create the EDA notebook structure
2. Implement data loading and initial inspection
3. Perform comprehensive data analysis
4. Document key findings
5. Identify feature engineering opportunities

### Phase 3: Pipeline Architecture

1. Design the overall pipeline structure
2. Implement the configuration system
3. Create base classes and interfaces
4. Implement key design patterns

### Phase 4: Feature Engineering

1. Implement data preprocessing
2. Create feature transformation pipeline
3. Develop feature selection methods
4. Test feature engineering effectiveness

### Phase 5: Model Development

1. Implement base model class
2. Create specific model implementations
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting / XGBoost
3. Implement hyperparameter optimization
4. Create model evaluation framework

### Phase 6: Pipeline Integration

1. Connect all pipeline components
2. Create main execution script
3. Implement error handling and logging
4. Test end-to-end workflow

### Phase 7: Documentation & Submission

1. Complete the README.md
2. Finalize the EDA notebook
3. Create requirement.txt
4. Create run.sh script
5. Test GitHub Actions workflow
6. Submit the assessment

## Testing Strategy

1. **Unit Tests**
   - Test individual components
   - Verify expected outputs
   - Check edge cases

2. **Integration Tests**
   - Test component interactions
   - Validate data flow between modules
   - Ensure proper error handling

3. **End-to-End Tests**
   - Validate full pipeline execution
   - Test with different configurations
   - Verify performance metrics

## Evaluation Metrics

For model evaluation, consider:

1. **Classification Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC

2. **Business Metrics**
   - Conversion rate improvement
   - Marketing cost reduction
   - Campaign effectiveness

## Documentation Requirements

### README.md Structure

1. **Personal Information**
   - Full name (as in NRIC)
   - Email address

2. **Project Overview**
   - Problem statement
   - Solution approach
   - Key findings

3. **Repository Structure**
   - Folder organization
   - File descriptions

4. **Pipeline Description**
   - Architecture overview
   - Component relationships
   - Flow diagram

5. **EDA Summary**
   - Key insights
   - Data characteristics
   - Feature importance findings

6. **Feature Processing**
   - Processing table
   - Transformation rationale
   - Engineering decisions

7. **Model Selection**
   - Models implemented
   - Selection rationale
   - Comparison of approaches

8. **Evaluation Results**
   - Performance metrics
   - Comparison analysis
   - Metric justification

9. **Deployment Considerations**
   - Productionization notes
   - Scaling recommendations
   - Future improvements

10. **Usage Instructions**
    - Pipeline execution steps
    - Configuration options
    - Parameter tuning guidance

## Submission Checklist

Before submission, ensure:

- [ ] Project structure follows requirements
- [ ] `eda.ipynb` is complete with explanations
- [ ] ML Pipeline is implemented in `.py` files
- [ ] `requirements.txt` is correctly formatted
- [ ] `run.sh` executes successfully
- [ ] `README.md` contains all required sections
- [ ] Repository is private
- [ ] AISG-AIAP is added as collaborator
- [ ] GitHub Actions workflow passes
- [ ] Google Form is completed

## Key Implementation Notes

1. **Data Handling**
   - Be cautious about synthetic/contaminated data
   - Document assumptions made during processing
   - Handle missing values appropriately

2. **Model Development**
   - Implement at least 3 models
   - Justify model selection based on data characteristics
   - Use appropriate hyperparameter tuning

3. **Code Quality**
   - Follow Python best practices
   - Use appropriate design patterns
   - Create modular, reusable components

4. **Documentation**
   - Explain all decisions clearly
   - Document assumptions and limitations
   - Provide comprehensive usage instructions