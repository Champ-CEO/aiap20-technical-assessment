# AI-Vive-Banking Term Deposit Prediction

## Project Overview

This project develops machine learning models to predict client term deposit subscription likelihood based on client information and direct marketing campaign data. The solution helps AI-Vive-Banking optimize marketing strategies by identifying clients most likely to respond positively to campaigns.

**Current Status:** Phase 2 Complete (EDA) | Phase 3 In Progress (Data Cleaning)

## Repository Structure

```
aiap20/
├── data/                  # Data directory
│   ├── raw/               # Raw data from bmarket.db
│   │   ├── bmarket.db     # Source SQLite database
│   │   ├── initial_dataset.csv  # Extracted raw dataset
│   │   └── download_db.py # Database download utility
│   ├── processed/         # Cleaned data (Phase 3 output)
│   └── featured/          # Feature-engineered data (Phase 5 output)
├── src/                   # Source code modules
│   ├── data/              # Data handling modules
│   │   └── data_loader.py # SQLite database connection
│   ├── preprocessing/     # Data cleaning and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML model training and evaluation
│   ├── evaluation/        # Model evaluation utilities
│   └── utils/             # Utility functions
├── tests/                 # Test suite (streamlined approach)
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── smoke/             # Smoke tests
│   ├── conftest.py        # Test fixtures
│   └── run_tests.py       # Test runner script
├── specs/                 # Project documentation
│   ├── output/            # EDA outputs and reports
│   │   ├── eda-report.md  # Complete EDA findings
│   │   └── eda-figures.md # EDA visualizations
│   └── TASKS.md           # Detailed project roadmap
├── eda.py                 # Standalone EDA script
├── pyproject.toml         # Project configuration and dependencies
└── requirements.txt       # Legacy dependency file
```

## Data Pipeline

**Current Progress:** ✅ Phase 1-2 Complete | 🔄 Phase 3 In Progress

1. **✅ Data Extraction (Phase 2)**: SQLite database (bmarket.db) → `data/raw/initial_dataset.csv`
2. **🔄 Data Cleaning (Phase 3)**: Handle missing values, standardize formats, fix data types → `data/processed/cleaned-db.csv`
3. **⏳ Feature Engineering (Phase 5)**: Create derived features, encode categorical variables → `data/featured/featured-db.csv`
4. **⏳ Model Training (Phase 7)**: Train and evaluate multiple prediction models
5. **⏳ Model Selection (Phase 9)**: Select best performing model based on business metrics

### Key Data Insights (from EDA)
- **Dataset Size**: 41,188 banking clients with 12 features
- **Target Distribution**: 11.3% subscription rate (class imbalance: 7.9:1)
- **Data Quality Issues**: 28,935 missing values, 12,008 special values requiring cleaning
- **Critical Cleaning Needs**: Age text-to-numeric conversion, standardize contact methods, handle 'unknown' categories

## Quick Start

### Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/aiap20.git
cd aiap20

# Create virtual environment (Python 3.12+ required)
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies using pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Download Data

```bash
# Download the banking dataset
python data/raw/download_db.py
```

### Run Tests

```bash
# Run smoke tests (quick verification)
python tests/run_tests.py smoke

# Run all tests
python tests/run_tests.py all

# Run with coverage report
python tests/run_tests.py coverage

# Current test status: 25/25 tests passing ✅
```

### Run EDA Analysis

```bash
# Run comprehensive EDA analysis
python eda.py

# View EDA results
cat specs/output/eda-report.md
```

### Current Available Components

```bash
# Test data loader functionality
python -c "from src.data.data_loader import BankingDataLoader; loader = BankingDataLoader(); print('Data loader working!')"

# Run Phase 2 validation
python tests/run_tests.py smoke
```

## EDA Key Findings

Based on comprehensive analysis of 41,188 banking clients:

### Target Variable Insights
- **Subscription Rate**: 11.3% of clients subscribe to term deposits
- **Class Imbalance**: 7.9:1 ratio (88.7% no, 11.3% yes)
- **Business Impact**: Significant opportunity for targeted marketing optimization

### Data Quality Assessment
- **Missing Values**: 28,935 total requiring attention
  - Housing Loan: 60.2% missing (24,789 records)
  - Personal Loan: 10.1% missing (4,146 records)
- **Special Values**: 12,008 'unknown' categories across features
  - Credit Default: 20.9% unknown (highest priority for cleaning)
  - Education Level: 4.2% unknown

### Feature Characteristics
- **Age Distribution**: Text format requiring conversion ('57 years' → 57)
- **Contact Methods**: Inconsistent formatting (Cell/cellular, Telephone/telephone)
- **Campaign Data**: Previous contact days (96.3% have '999' = no previous contact)
- **Demographics**: 60.5% married, 29.5% university degree, 25.3% admin occupation

## Business Impact Potential

Based on EDA findings, the ML solution will enable:

- **Targeted Marketing**: Focus on high-probability prospects (11.3% baseline)
- **Resource Optimization**: Reduce contact fatigue through better targeting
- **Campaign Efficiency**: Leverage demographic and behavioral patterns
- **Data-Driven Decisions**: Clear insights into customer subscription drivers

## Testing Philosophy

This project follows a streamlined testing approach focusing on critical path verification:

- **Smoke Tests**: Quick pipeline verification and project setup validation
- **Unit Tests**: Core function verification (data loading, validation)
- **Integration Tests**: Component interactions and data flow testing

**Current Test Status**: 25/25 tests passing ✅

### Test Commands
```bash
# Quick verification (recommended for development)
python tests/run_tests.py smoke

# Full test suite
python tests/run_tests.py all

# With coverage reporting
python tests/run_tests.py coverage
```

## Project Status & Next Steps

### ✅ Completed Phases
- **Phase 1**: Project setup, environment configuration, testing framework
- **Phase 2**: Data extraction, EDA analysis, data quality assessment

### 🔄 Current Phase
- **Phase 3**: Data cleaning and preprocessing implementation
  - Age text-to-numeric conversion
  - Missing value handling (28,935 records)
  - Special value cleaning (12,008 'unknown' values)
  - Target variable binary encoding

### ⏳ Upcoming Phases
- **Phase 4**: Data integration and validation
- **Phase 5**: Feature engineering with business context
- **Phase 7**: Model implementation (5 classifiers planned)
- **Phase 9**: Model selection and optimization

## Documentation

- **Detailed Project Plan**: `specs/TASKS.md`
- **EDA Complete Report**: `specs/output/eda-report.md`
- **EDA Visualizations**: `specs/output/eda-figures.md`
- **Test Results**: `tests/PHASE2_TESTING_SUMMARY.md`

## License

[MIT License](LICENSE)

