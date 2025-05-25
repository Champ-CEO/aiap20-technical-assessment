# AI-Vive-Banking Term Deposit Prediction

## Project Overview

This project develops machine learning models to predict client term deposit subscription likelihood based on client information and direct marketing campaign data. The solution helps AI-Vive-Banking optimize marketing strategies by identifying clients most likely to respond positively to campaigns.

## Repository Structure

```
aiap20/
├── data/                  # Data directory
│   ├── raw/               # Raw data from bmarket.db
│   ├── processed/         # Cleaned data
│   └── featured/          # Feature-engineered data
├── notebooks/             # Jupyter notebooks
│   └── eda.ipynb          # Exploratory Data Analysis
├── src/                   # Source code
│   ├── data/              # Data handling modules
│   ├── preprocessing/     # Data cleaning and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML model training and evaluation
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── smoke/             # Smoke tests
│   ├── conftest.py        # Test fixtures
│   └── run_tests.py       # Test runner script
└── requirements.txt       # Project dependencies
```

## Data Pipeline

1. **Data Extraction**: SQLite database (bmarket.db) → raw dataset
2. **Data Cleaning**: Handle missing values, standardize formats, fix data types
3. **Feature Engineering**: Create derived features, encode categorical variables
4. **Model Training**: Train and evaluate multiple prediction models
5. **Model Selection**: Select best performing model based on business metrics

## Quick Start

### Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/aiap20.git
cd aiap20

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
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
```

### Run EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

### Run ML Pipeline

```bash
# Run the complete pipeline
./run.sh

# Or run specific components
python src/main.py --config config/default.yaml
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Model 1 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Model 2 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Model 3 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## Key Findings

- Finding 1: Description of important insight
- Finding 2: Description of important insight
- Finding 3: Description of important insight

## Business Impact

- Improved marketing campaign efficiency by X%
- Reduced customer contact fatigue
- Increased term deposit conversion rate
- Optimized resource allocation for marketing efforts

## Testing Philosophy

This project follows a streamlined testing approach focusing on critical path verification:
- **Smoke Tests**: Quick pipeline verification
- **Unit Tests**: Core function verification
- **Integration Tests**: Component interactions

Run tests with: `python tests/run_tests.py [test_type]`

## License

[MIT License](LICENSE)

