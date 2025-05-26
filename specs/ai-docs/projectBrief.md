# AI-Vive-Banking Term Deposit Prediction Project

## Project Overview
Development of machine learning models to predict client term deposit subscription likelihood based on client information and direct marketing campaign data.

## Objectives
1. Predict likelihood of client subscription to term deposits
2. Optimize marketing strategies through accurate client response prediction
3. Identify key features influencing subscription status
4. Develop and evaluate at least three suitable prediction models

## Project Structure
project/
├── data/ # Data directory (do not commit bmarket.db)
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis notebook
├── src/ # Source code for ML pipeline
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── run.sh # Execution script

## Dataset Details
- Source: bmarket.db (SQLite database)
- Location: https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db

### Attributes
| Attribute | Description |
|-----------|-------------|
| Client ID | Unique Identifier for the client |
| Age | The age of the client |
| Occupation | Type of job held by the client |
| Marital Status | Marital status of the client |
| Education Level | Highest education level attained |
| Credit Default | Indicates if client has credit in default |
| Housing Loan | Indicates if client has a housing loan |
| Personal Loan | Indicates if client has a personal loan |
| Contact Method | Communication type for last contact |
| Campaign Calls | Total contacts during campaign |
| Previous Contact Days | Days since last contact in previous campaign |
| Subscription Status | Term deposit subscription (yes/no) |

## Tasks

### 1. Exploratory Data Analysis (EDA)
- Create interactive Jupyter notebook
- Include visualizations and explanations
- Document analysis findings
- Interpret statistics and their impact

### 2. Machine Learning Pipeline
- Develop end-to-end ML pipeline in Python scripts
- Include configurable parameters
- Implement data fetching using SQLite
- Create comprehensive documentation

### Required Deliverables
1. EDA Jupyter Notebook (`eda.ipynb`)
2. Source Code:
   - Python modules/classes in `src` folder
   - Executable bash script (`run.sh`)
   - Requirements file (`requirements.txt`)
3. Documentation (`README.md`)

## Technical Requirements
- Python-based implementation
- SQLite/SQLAlchemy for database access
- Minimum three ML models
- Proper code structure and documentation
- Clear evaluation metrics

## Notes
- Dataset may contain synthetic/contaminated data
- Document all assumptions and justifications
- Focus on code quality, reusability, and readability
- Include proper error handling and logging