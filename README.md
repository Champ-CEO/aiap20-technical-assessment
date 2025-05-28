# AI-Vive-Banking Term Deposit Prediction

## Project Overview

Machine learning solution to predict client term deposit subscription likelihood, enabling AI-Vive-Banking to optimize marketing campaigns through targeted client identification.

**Status:** Phase 5 Complete (Feature Engineering) | Phase 6 Ready (Model Preparation) | 67/70 Tests Passing ✅

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
│   ├── preprocessing/     # Data cleaning and preprocessing (Phase 3)
│   ├── data_integration/  # Data integration and validation (Phase 4)
│   ├── feature_engineering/ # Feature engineering (Phase 5)
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
│   ├── output/            # Phase outputs and reports
│   │   ├── eda-report.md  # Phase 2: Complete EDA findings
│   │   ├── eda-figures.md # Phase 2: EDA visualizations
│   │   ├── phase3-report.md # Phase 3: Data cleaning results
│   │   ├── phase4-report.md # Phase 4: Data integration results
│   │   └── phase5-report.md # Phase 5: Feature engineering results
│   └── TASKS.md           # Detailed project roadmap
├── eda.py                 # Standalone EDA script
├── pyproject.toml         # Project configuration and dependencies
└── requirements.txt       # Legacy dependency file
```

## Data Pipeline & Key Insights

**Progress:** ✅ Phase 1-5 Complete | 🔄 Phase 6 Ready

1. **✅ Data Extraction**: SQLite database → `data/raw/initial_dataset.csv`
2. **✅ Data Cleaning**: Missing values, standardization → `data/processed/cleaned-db.csv`
3. **✅ Data Integration**: Validation and pipeline → Phase 4 complete
4. **✅ Feature Engineering**: Derived features → `data/featured/featured-db.csv`
5. **⏳ Model Training**: Multiple classifiers evaluation
6. **⏳ Model Selection**: Business metrics optimization

**Dataset:** 41,188 clients, 33 features (after processing), 11.3% subscription rate (7.9:1 imbalance)
**Data Quality:** ✅ All issues resolved - 0 missing values, standardized categories, numeric age conversion complete

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

# Run EDA analysis (Phase 2)
python eda.py

# Run tests (smoke/all/coverage)
python tests/run_tests.py smoke    # 67/70 tests passing
python tests/run_tests.py all      # Full test suite
python tests/run_tests.py coverage # Coverage report

# Run specific phase tests
python tests/run_phase3_tests.py   # Data cleaning tests
python tests/run_phase4_tests.py   # Data integration tests
python tests/run_phase5_tests.py   # Feature engineering tests

# Test data loader
python -c "from src.data.data_loader import BankingDataLoader; loader = BankingDataLoader(); print('Data loader working!')"
```

## EDA Findings & Business Impact

**Key Insights from 41,188 banking clients:**

- **Target**: 11.3% subscription rate (7.9:1 class imbalance)
- **Data Quality**: ✅ Resolved - 0 missing values, standardized categories, numeric age
- **Demographics**: 60.5% married, 29.5% university degree, 25.3% admin occupation
- **Features**: 33 engineered features including age binning, contact recency, campaign intensity

**Business Value:**
- Targeted marketing optimization (11.3% baseline improvement potential)
- Resource efficiency through better prospect identification
- Data-driven campaign strategies leveraging demographic patterns

## Testing & Project Status

**Testing Approach:** Streamlined critical path verification (smoke/unit/integration tests)
**Current Status:** 67/70 tests passing ✅ (3 minor dependency issues)

### Phase Progress
- **✅ Phase 1-2**: Setup, EDA, data quality assessment
- **✅ Phase 3**: Data cleaning (age conversion, missing values, 'unknown' handling)
- **✅ Phase 4**: Data integration and validation pipeline
- **✅ Phase 5**: Feature engineering (age binning, contact recency, campaign intensity)
- **🔄 Phase 6**: Model preparation (ready to start)
- **⏳ Phase 7-9**: Model training, evaluation, selection

## Documentation

- **Project Plan**: `specs/TASKS.md`
- **Phase 2 - EDA Report**: `specs/output/eda-report.md`
- **Phase 2 - EDA Visualizations**: `specs/output/eda-figures.md`
- **Phase 3 - Data Cleaning**: `specs/output/phase3-report.md`
- **Phase 4 - Data Integration**: `specs/output/phase4-report.md`
- **Phase 5 - Feature Engineering**: `specs/output/phase5-report.md`
- **Test Results**: `tests/PHASE2_TESTING_SUMMARY.md`, `tests/PHASE3_TESTING_SUMMARY.md`, `tests/PHASE4_TESTING_SUMMARY.md`, `tests/PHASE5_TESTING_SUMMARY.md`

## License

[MIT License](LICENSE)

