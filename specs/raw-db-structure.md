# Enhanced Database Structure Analysis

## Table: `bank_marketing`
- **Row Count:** 41,188

### Schema
| Column Name | Data Type | Nullable | Default | Primary Key |
|-------------|-----------|----------|---------|-------------|
| Client ID | INTEGER | True | None |  |
| Age | TEXT | True | None |  |
| Occupation | TEXT | True | None |  |
| Marital Status | TEXT | True | None |  |
| Education Level | TEXT | True | None |  |
| Credit Default | TEXT | True | None |  |
| Housing Loan | TEXT | True | None |  |
| Personal Loan | TEXT | True | None |  |
| Contact Method | TEXT | True | None |  |
| Campaign Calls | INTEGER | True | None |  |
| Previous Contact Days | INTEGER | True | None |  |
| Subscription Status | TEXT | True | None |  |

### Data Preview
|    |   Client ID | Age      | Occupation   | Marital Status   | Education Level   | Credit Default   | Housing Loan   | Personal Loan   | Contact Method   |   Campaign Calls |   Previous Contact Days | Subscription Status   |
|---:|------------:|:---------|:-------------|:-----------------|:------------------|:-----------------|:---------------|:----------------|:-----------------|-----------------:|------------------------:|:----------------------|
|  0 |       32885 | 57 years | technician   | married          | high.school       | no               | no             | yes             | Cell             |                1 |                     999 | no                    |
|  1 |        3170 | 55 years | unknown      | married          | unknown           | unknown          | yes            | no              | telephone        |                2 |                     999 | no                    |
|  2 |       32207 | 33 years | blue-collar  | married          | basic.9y          | no               | no             | no              | cellular         |                1 |                     999 | no                    |
|  3 |        9404 | 36 years | admin.       | married          | high.school       | no               | no             | no              | Telephone        |                4 |                     999 | no                    |
|  4 |       14021 | 27 years | housemaid    | married          | high.school       | no               |                | no              | Cell             |                2 |                     999 | no                    |

### Data Quality Analysis

#### Missing Values
| Column | Missing Count | Missing Percentage |
|--------|---------------|--------------------|
| Housing Loan | 24789 | 60.19% |
| Personal Loan | 4146 | 10.07% |

#### Data Types and Unique Values
| Column | Data Type | Unique Values | Unique Percentage |
|--------|-----------|---------------|-------------------|
| Client ID | int64 | 41188 | 100.0% |
| Age | object | 77 | 0.19% |
| Occupation | object | 12 | 0.03% |
| Marital Status | object | 4 | 0.01% |
| Education Level | object | 8 | 0.02% |
| Credit Default | object | 3 | 0.01% |
| Housing Loan | object | 3 | 0.01% |
| Personal Loan | object | 3 | 0.01% |
| Contact Method | object | 4 | 0.01% |
| Campaign Calls | int64 | 70 | 0.17% |
| Previous Contact Days | int64 | 27 | 0.07% |
| Subscription Status | object | 2 | 0.0% |

#### Categorical Distributions
**Occupation**
| Value | Count | Percentage |
|-------|-------|------------|
| admin. | 10422 | 25.3% |
| blue-collar | 9254 | 22.47% |
| technician | 6743 | 16.37% |
| services | 3969 | 9.64% |
| management | 2924 | 7.1% |
| retired | 1720 | 4.18% |
| entrepreneur | 1456 | 3.54% |
| self-employed | 1421 | 3.45% |
| housemaid | 1060 | 2.57% |
| unemployed | 1014 | 2.46% |
| student | 875 | 2.12% |
| unknown | 330 | 0.8% |

**Marital Status**
| Value | Count | Percentage |
|-------|-------|------------|
| married | 24928 | 60.52% |
| single | 11568 | 28.09% |
| divorced | 4612 | 11.2% |
| unknown | 80 | 0.19% |

**Education Level**
| Value | Count | Percentage |
|-------|-------|------------|
| university.degree | 12168 | 29.54% |
| high.school | 9515 | 23.1% |
| basic.9y | 6045 | 14.68% |
| professional.course | 5243 | 12.73% |
| basic.4y | 4176 | 10.14% |
| basic.6y | 2292 | 5.56% |
| unknown | 1731 | 4.2% |
| illiterate | 18 | 0.04% |

**Credit Default**
| Value | Count | Percentage |
|-------|-------|------------|
| no | 32588 | 79.12% |
| unknown | 8597 | 20.87% |
| yes | 3 | 0.01% |

**Housing Loan**
| Value | Count | Percentage |
|-------|-------|------------|
| yes | 8595 | 20.87% |
| no | 7411 | 17.99% |
| unknown | 393 | 0.95% |

**Personal Loan**
| Value | Count | Percentage |
|-------|-------|------------|
| no | 30532 | 74.13% |
| yes | 5633 | 13.68% |
| unknown | 877 | 2.13% |

**Contact Method**
| Value | Count | Percentage |
|-------|-------|------------|
| Cell | 13100 | 31.81% |
| cellular | 13044 | 31.67% |
| Telephone | 7585 | 18.42% |
| telephone | 7459 | 18.11% |

**Subscription Status**
| Value | Count | Percentage |
|-------|-------|------------|
| no | 36548 | 88.73% |
| yes | 4640 | 11.27% |

#### Special Values
| Column | Special Value | Count | Percentage |
|--------|---------------|-------|------------|
| Client ID | 999 | 1 | 0.0% |
| Occupation | unknown | 330 | 0.8% |
| Marital Status | unknown | 80 | 0.19% |
| Education Level | unknown | 1731 | 4.2% |
| Credit Default | unknown | 8597 | 20.87% |
| Housing Loan | unknown | 393 | 0.95% |
| Personal Loan | unknown | 877 | 2.13% |
| Previous Contact Days | 999 | 39673 | 96.32% |

#### Target Variable Distribution
| Subscription Status | Count | Percentage |
|---------------------|-------|------------|
| no | 36548 | 88.73% |
| yes | 4640 | 11.27% |

#### Numeric Column Statistics
| Column                |    count |     mean |      std |    min |      25% |      50% |      75% |      max |
|:----------------------|---------:|---------:|---------:|-------:|---------:|---------:|---------:|---------:|
| Client ID             | 41188.00 | 20594.50 | 11890.10 |   1.00 | 10297.75 | 20594.50 | 30891.25 | 41188.00 |
| Campaign Calls        | 41188.00 |     2.05 |     3.17 | -41.00 |     1.00 |     2.00 |     3.00 |    56.00 |
| Previous Contact Days | 41188.00 |   962.48 |   186.91 |   0.00 |   999.00 |   999.00 |   999.00 |   999.00 |

## Recommendations

### Data Cleaning
- High missing values (60.19%) in 'Housing Loan'. Consider dropping the column or using advanced imputation techniques.
- Missing values (10.07%) in 'Personal Loan'. Impute with appropriate strategy (mean/median for numeric, mode for categorical).
- Handle special value '999' in 'Client ID' (1 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Occupation' (330 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Marital Status' (80 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Education Level' (1731 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Credit Default' (8597 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Housing Loan' (393 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value 'unknown' in 'Personal Loan' (877 occurrences). Consider converting to NaN or a meaningful value.
- Handle special value '999' in 'Previous Contact Days' (39673 occurrences). Consider converting to NaN or a meaningful value.

### Data Preprocessing
- Encode categorical column 'Occupation'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Marital Status'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Education Level'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Credit Default'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Housing Loan'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Personal Loan'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Contact Method'. Consider one-hot encoding or label encoding based on cardinality.
- Encode categorical column 'Subscription Status'. Consider one-hot encoding or label encoding based on cardinality.
- Handle outliers in 'Campaign Calls' (3420 outliers, 8.3%). Consider capping, transformation, or binning.
- Handle outliers in 'Previous Contact Days' (1515 outliers, 3.68%). Consider capping, transformation, or binning.

### Feature Engineering
- Create age bins/categories from 'Age'.
- Create recency features from 'Previous Contact Days'.
- Create call frequency categories from 'Campaign Calls'.
- Create interaction features between demographic attributes (e.g., Age*Education).
- Create binary flags for key customer attributes (e.g., has_default, has_housing_loan).

### Exploratory Data Analysis
- Analyze class imbalance. Minor class represents 11.27% of data.
- Explore correlations between features and the target variable.
- Analyze feature distributions and their impact on subscription likelihood.
- Investigate feature relationships through cross-tabulations and visualization.

## Conclusion

The database contains banking client information and marketing campaign data with several quality issues that need addressing. Key challenges include handling special values like '999' and 'unknown', converting text fields to appropriate data types, and addressing potential class imbalance in the target variable.

The recommended approach is to first clean the data by handling missing and special values, then preprocess by standardizing data types and encoding categorical variables, followed by feature engineering to extract more predictive information from the existing features. The exploratory analysis should focus on understanding feature distributions and their relationship with the target variable.

