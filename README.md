# AI-Vive-Banking Term Deposit Prediction

## Project Overview

Machine learning solution to predict client term deposit subscription likelihood, enabling AI-Vive-Banking to optimize marketing campaigns through targeted client identification.

**Status:** Phase 2 Complete (EDA) | Phase 3 In Progress (Data Cleaning) | 25/25 Tests Passing ✅

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

## Data Pipeline & Key Insights

**Progress:** ✅ Phase 1-2 Complete | 🔄 Phase 3 In Progress

1. **✅ Data Extraction**: SQLite database → `data/raw/initial_dataset.csv`
2. **🔄 Data Cleaning**: Missing values, standardization → `data/processed/cleaned-db.csv`
3. **⏳ Feature Engineering**: Derived features → `data/featured/featured-db.csv`
4. **⏳ Model Training**: Multiple classifiers evaluation
5. **⏳ Model Selection**: Business metrics optimization

**Dataset:** 41,188 clients, 12 features, 11.3% subscription rate (7.9:1 imbalance)
**Data Issues:** 28,935 missing values, 12,008 'unknown' categories, age format conversion needed

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

### Run Analysis & Tests

```bash
# Download banking dataset
python data/raw/download_db.py

# Run EDA analysis
python eda.py

# Run tests (smoke/all/coverage)
python tests/run_tests.py smoke
python tests/run_tests.py all
python tests/run_tests.py coverage

# Test data loader
python -c "from src.data.data_loader import BankingDataLoader; loader = BankingDataLoader(); print('Data loader working!')"
```

## EDA Findings & Business Impact

**Key Insights from 41,188 banking clients:**

- **Target**: 11.3% subscription rate (7.9:1 class imbalance)
- **Missing Data**: 28,935 values (Housing: 60.2%, Personal Loan: 10.1%)
- **Data Quality**: 12,008 'unknown' categories, age format conversion needed
- **Demographics**: 60.5% married, 29.5% university degree, 25.3% admin occupation

**Business Value:**
- Targeted marketing optimization (11.3% baseline improvement potential)
- Resource efficiency through better prospect identification
- Data-driven campaign strategies leveraging demographic patterns

## Testing & Project Status

**Testing Approach:** Streamlined critical path verification (smoke/unit/integration tests)
**Current Status:** 25/25 tests passing ✅

### Phase Progress
- **✅ Phase 1-2**: Setup, EDA, data quality assessment
- **🔄 Phase 3**: Data cleaning (age conversion, missing values, 'unknown' handling)
- **⏳ Phase 4-5**: Data integration, feature engineering
- **⏳ Phase 7-9**: Model training, evaluation, selection

## Documentation

- **Project Plan**: `specs/TASKS.md`
- **EDA Report**: `specs/output/eda-report.md`
- **EDA Visualizations**: `specs/output/eda-figures.md`
- **Test Results**: `tests/PHASE2_TESTING_SUMMARY.md`

## License

[MIT License](LICENSE)

