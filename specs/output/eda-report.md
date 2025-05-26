# EDA Report - 2025-05-26 23:09:38

This report contains the complete console output from the EDA process.

```
================================================================================
AI-VIVE-BANKING TERM DEPOSIT PREDICTION - EXPLORATORY DATA ANALYSIS
================================================================================
Data Source: bmarket.db (SQLite Database)
Business Objective: Predict term deposit subscription likelihood
Focus: Marketing campaign optimization and client targeting
ARTIFACT SAVING: Enabled - Saving to specs/output/
================================================================================

📊 LOADING DATA FROM: C:\Users\ellie\VSProjects\aisg\aiap20\data\raw\bmarket.db
--------------------------------------------------
✅ Connected to database successfully
📋 Using table: bank_marketing
✅ Data loaded successfully:
   • Rows: 41,188
   • Columns: 12
   • Memory usage: 19.62 MB
✅ Database connection closed

================================================================================
2. DATA OVERVIEW AND QUALITY ASSESSMENT
================================================================================

📋 DATASET OVERVIEW
------------------------------
Dataset Shape: 41,188 rows × 12 columns
Data Source: bmarket.db - bank_marketing table
Business Context: Banking marketing campaign data for term deposit prediction

📊 COLUMN INFORMATION
------------------------------
Column Names and Data Types:
 1. Client ID                 | int64     
 2. Age                       | object    
 3. Occupation                | object    
 4. Marital Status            | object    
 5. Education Level           | object    
 6. Credit Default            | object    
 7. Housing Loan              | object    
 8. Personal Loan             | object    
 9. Contact Method            | object    
10. Campaign Calls            | int64     
11. Previous Contact Days     | int64     
12. Subscription Status       | object    

🔍 DATA QUALITY ASSESSMENT
------------------------------
Missing Values Analysis:
               Missing Count  Missing %
Housing Loan           24789  60.185005
Personal Loan           4146  10.066039

Unique Values per Column:
  Subscription Status      :      2 unique values
  Housing Loan             :      3 unique values
  Credit Default           :      3 unique values
  Personal Loan            :      3 unique values
  Contact Method           :      4 unique values
  Marital Status           :      4 unique values
  Education Level          :      8 unique values
  Occupation               :     12 unique values
  Previous Contact Days    :     27 unique values
  Campaign Calls           :     70 unique values
  Age                      :     77 unique values
  Client ID                :  41188 unique values

🎯 TARGET VARIABLE ANALYSIS
------------------------------
Target Variable: Subscription Status
Distribution:
  no        : 36,548 ( 88.7%)
  yes       :  4,640 ( 11.3%)

Class Imbalance Ratio: 7.88:1
⚠️  Significant class imbalance detected - consider balancing techniques

📄 DATA SAMPLE
------------------------------
First 5 rows:
   Client ID       Age   Occupation Marital Status Education Level Credit Default Housing Loan Personal Loan Contact Method  Campaign Calls  Previous Contact Days Subscription Status
0      32885  57 years   technician        married     high.school             no           no           yes           Cell               1                    999                  no
1       3170  55 years      unknown        married         unknown        unknown          yes            no      telephone               2                    999                  no
2      32207  33 years  blue-collar        married        basic.9y             no           no            no       cellular               1                    999                  no
3       9404  36 years       admin.        married     high.school             no           no            no      Telephone               4                    999                  no
4      14021  27 years    housemaid        married     high.school             no         None            no           Cell               2                    999                  no

Last 5 rows:
       Client ID       Age  Occupation Marital Status      Education Level Credit Default Housing Loan Personal Loan Contact Method  Campaign Calls  Previous Contact Days Subscription Status
41183       6266  58 years     retired        married  professional.course        unknown           no            no      Telephone               2                    999                  no
41184      11285  37 years  management        married    university.degree             no           no            no      telephone               1                    999                  no
41185      38159  35 years      admin.        married          high.school             no         None            no       cellular               1                      4                 yes
41186        861  40 years  management        married    university.degree             no         None            no      telephone               2                    999                  no
41187      15796  29 years      admin.         single    university.degree             no          yes            no           Cell               2                    999                  no

================================================================================
3. RAW DATA QUALITY ASSESSMENT FOR PHASE 3 PREPARATION
================================================================================

🔍 DATA QUALITY ISSUES IDENTIFICATION
---------------------------------------------
Identifying special values that need Phase 3 cleaning:
  Occupation: unknown: 330
  Marital Status: unknown: 80
  Education Level: unknown: 1731
  Credit Default: unknown: 8597
  Housing Loan: unknown: 393
  Personal Loan: unknown: 877

Age column format analysis:
  • Data type: object
  • Sample values: ['57 years', '55 years', '33 years', '36 years', '27 years']
  ⚠️  Age stored as text - needs conversion to numeric in Phase 3

Campaign Calls special values analysis:
  ✅ No '999' values found in Campaign Calls

Previous Contact Days special values analysis:
  ⚠️  Found 39673 rows with '999' previous contact days
     → Phase 3 action: Create 'No Previous Contact' flag

Target variable format analysis:
  • Target variable: Subscription Status
  • Data type: object
  • Unique values: ['no', 'yes']
  → Phase 3 action: Convert to binary encoding (1=yes, 0=no)

📋 PHASE 3 CLEANING REQUIREMENTS IDENTIFIED
   → Age conversion from text to numeric
   → Special value handling (999, unknown)
   → Target variable binary encoding
   → Missing value strategy implementation

================================================================================
4. DESCRIPTIVE STATISTICS AND FEATURE DISTRIBUTIONS
================================================================================

📊 NUMERICAL FEATURES ANALYSIS (RAW DATA)
--------------------------------------------------
Descriptive Statistics for Numerical Features (Raw Data):
       Campaign Calls  Previous Contact Days
count        41188.00               41188.00
mean             2.05                 962.48
std              3.17                 186.91
min            -41.00                   0.00
25%              1.00                 999.00
50%              2.00                 999.00
75%              3.00                 999.00
max             56.00                 999.00

📊 CATEGORICAL FEATURES ANALYSIS (RAW DATA)
--------------------------------------------------
Categorical Features Summary (Raw Data):

Occupation:
  • Unique values: 12
  • Most common: 'admin.' (25.3%)
  • Top categories:
    - admin.: 10,422 (25.3%)
    - blue-collar: 9,254 (22.5%)
    - technician: 6,743 (16.4%)
    - services: 3,969 (9.6%)
    - management: 2,924 (7.1%)

Marital Status:
  • Unique values: 4
  • Most common: 'married' (60.5%)
  • Top categories:
    - married: 24,928 (60.5%)
    - single: 11,568 (28.1%)
    - divorced: 4,612 (11.2%)
    - unknown: 80 (0.2%)

Education Level:
  • Unique values: 8
  • Most common: 'university.degree' (29.5%)
  • Top categories:
    - university.degree: 12,168 (29.5%)
    - high.school: 9,515 (23.1%)
    - basic.9y: 6,045 (14.7%)
    - professional.course: 5,243 (12.7%)
    - basic.4y: 4,176 (10.1%)

Credit Default:
  • Unique values: 3
  • Most common: 'no' (79.1%)
  • Top categories:
    - no: 32,588 (79.1%)
    - unknown: 8,597 (20.9%)
    - yes: 3 (0.0%)

Housing Loan:
  • Unique values: 3
  • Most common: 'yes' (20.9%)
  • Top categories:
    - yes: 8,595 (20.9%)
    - no: 7,411 (18.0%)
    - unknown: 393 (1.0%)

Personal Loan:
  • Unique values: 3
  • Most common: 'no' (74.1%)
  • Top categories:
    - no: 30,532 (74.1%)
    - yes: 5,633 (13.7%)
    - unknown: 877 (2.1%)

Contact Method:
  • Unique values: 4
  • Most common: 'Cell' (31.8%)
  • Top categories:
    - Cell: 13,100 (31.8%)
    - cellular: 13,044 (31.7%)
    - Telephone: 7,585 (18.4%)
    - telephone: 7,459 (18.1%)

✅ Raw data feature analysis completed

================================================================================
5. RAW DATA VISUALIZATION AND PATTERNS (EDA FOCUS)
================================================================================

🎯 TARGET VARIABLE DISTRIBUTION (RAW DATA)
--------------------------------------------------
Target Variable Distribution:
  no: 36,548 (88.7%)
  yes: 4,640 (11.3%)

📊 CATEGORICAL FEATURES VISUALIZATION (RAW DATA)
-------------------------------------------------------

📊 NUMERICAL FEATURES DISTRIBUTION ANALYSIS
--------------------------------------------------

🔍 DATA QUALITY PATTERNS VISUALIZATION
---------------------------------------------
Special values requiring Phase 3 cleaning:
  Marital Status: 80 special values (0.2% of data)
  Occupation: 330 special values (0.8% of data)
  Housing Loan: 393 special values (1.0% of data)
  Personal Loan: 877 special values (2.1% of data)
  Education Level: 1,731 special values (4.2% of data)
  Credit Default: 8,597 special values (20.9% of data)

✅ Raw data visualization and pattern analysis completed

================================================================================
6. EDA INSIGHTS FOR PHASE 3 PREPARATION
================================================================================

📋 RAW DATA PATTERNS SUMMARY
-----------------------------------
Key findings from raw data exploration:

🎯 Target Variable Insights:
  • no: 88.7% of customers
  • yes: 11.3% of customers
  ⚠️  Class imbalance detected: 11.3% minority class
     → Phase 3 recommendation: Consider balancing techniques

📊 Categorical Features Insights:
  • Occupation: 12 categories, 25.3% in top category
  • Marital Status: 4 categories, 60.5% in top category
  • Education Level: 8 categories, 29.5% in top category
  • Credit Default: 3 categories, 79.1% in top category
  • Housing Loan: 3 categories, 20.9% in top category
  • Personal Loan: 3 categories, 74.1% in top category
  • Contact Method: 4 categories, 31.8% in top category

📈 Numerical Features Insights:
  • Campaign Calls: Mean=2.1, CV=154.6%
  • Previous Contact Days: Mean=962.5, CV=19.4%

🔍 DATA QUALITY ASSESSMENT FOR PHASE 3
---------------------------------------------
Data quality issues requiring Phase 3 attention:
  ⚠️  Missing values: 28,935 total
  ⚠️  Special values: 12,008 total requiring cleaning
     → Most affected columns:
       - Occupation: 330 (0.8%)
       - Marital Status: 80 (0.2%)
       - Education Level: 1,731 (4.2%)
       - Credit Default: 8,597 (20.9%)
       - Housing Loan: 393 (1.0%)
       - Personal Loan: 877 (2.1%)

📝 Data Type Issues:
  ⚠️  Age column stored as text - needs numeric conversion

✅ EDA insights summary completed

================================================================================
7. PHASE 3 RECOMMENDATIONS BASED ON EDA FINDINGS
================================================================================

🔧 DATA CLEANING RECOMMENDATIONS FOR PHASE 3
--------------------------------------------------
Based on EDA findings, Phase 3 should implement:

1. Missing Values Handling:
   • Total missing values: 28,935
   • Recommended approach: Domain-specific imputation
   • Consider creating 'missing' indicator flags

2. Special Values Cleaning:
   • Occupation: 330 special values
     → Handle 'unknown' as separate category or impute
   • Marital Status: 80 special values
     → Handle 'unknown' as separate category or impute
   • Education Level: 1,731 special values
     → Handle 'unknown' as separate category or impute
   • Credit Default: 8,597 special values
     → Handle 'unknown' as separate category or impute
   • Housing Loan: 393 special values
     → Handle 'unknown' as separate category or impute
   • Personal Loan: 877 special values
     → Handle 'unknown' as separate category or impute

3. Data Type Conversions:
   • Convert Age from text to numeric
   • Extract numeric values using regex patterns

4. Target Variable Encoding:
   • Convert 'Subscription Status' to binary (1=yes, 0=no)
   • Ensure consistent encoding across dataset

🛠️ FEATURE ENGINEERING RECOMMENDATIONS
---------------------------------------------
Recommended feature engineering for Phase 3:

1. Categorical Feature Engineering:
   • One-hot encode low-cardinality features
   • Label encode ordinal features (Education Level)
   • Create interaction features (Occupation × Education)

2. Numerical Feature Engineering:
   • Create age groups/bins for better segmentation
   • Normalize campaign calls and contact days
   • Create campaign intensity categories

3. Business Logic Features:
   • Create 'No Previous Contact' binary flag
   • Calculate days since last campaign
   • Create customer risk profiles

✅ DATA VALIDATION RECOMMENDATIONS
----------------------------------------
Implement these validation checks in Phase 3:

1. Range Validations:
   • Age: 18-100 years
   • Campaign Calls: 1-50 (cap extreme values)
   • Previous Contact Days: 0-999

2. Consistency Checks:
   • Standardize contact methods (cell/cellular)
   • Validate education level categories
   • Check occupation consistency

3. Business Rule Validations:
   • Ensure logical relationships between features
   • Validate campaign timing constraints
   • Check for impossible combinations

✅ Phase 3 recommendations completed

================================================================================
8. EDA SUMMARY AND NEXT STEPS FOR PHASE 3
================================================================================

🎯 KEY EDA FINDINGS (RAW DATA ANALYSIS)
---------------------------------------------
Summary of raw data exploration findings:

📊 Target Variable Insights:
  • Overall subscription rate: 11.3%
  • Class distribution: 88.7% / 11.3%
  • Imbalance ratio: 7.9:1

🔍 Data Quality Insights:
  • Dataset size: 41,188 records
  • Feature count: 12 columns
  • Missing values: 28,935 total
  • Special values: 12,008 requiring cleaning

📈 Feature Insights:
  • Categorical features: 7 identified
  • Numerical features: 2 identified
  • Age data type: Text (needs conversion)

✅ PHASE 3 PREPARATION CHECKLIST
----------------------------------------
Ready for Phase 3 implementation:
  ✅ Raw data structure understood
  ✅ Target variable distribution analyzed
  ✅ Data quality issues identified
  ✅ Special values catalogued
  ✅ Feature types classified
  ✅ Business patterns explored
  ✅ Cleaning requirements documented
  ✅ Visualization insights captured

🚀 CRITICAL PHASE 3 TASKS
------------------------------
1. Data Cleaning Pipeline:
   • Convert Age from text to numeric
   • Handle special values (999, unknown)
   • Create binary target encoding
   • Implement missing value strategies

2. Feature Engineering:
   • Create age group categories
   • Engineer campaign intensity features
   • Build interaction features
   • Generate business logic flags

3. Data Validation:
   • Implement range checks
   • Validate business rules
   • Ensure data consistency
   • Create quality metrics

4. Pipeline Integration:
   • Build reusable cleaning functions
   • Create data transformation pipeline
   • Implement error handling
   • Document transformation logic

📊 SUCCESS METRICS FOR PHASE 3
-----------------------------------
Data Quality Metrics:
  • Target: 0 missing values (currently: 28,935)
  • Target: 0 special values (currently: 12,008)
  • Target: All features properly typed
  • Target: 100% data validation pass rate

Feature Engineering Metrics:
  • Target: Age successfully converted to numeric
  • Target: All categorical features encoded
  • Target: Business logic features created
  • Target: Feature correlation analysis completed

Pipeline Metrics:
  • Target: End-to-end pipeline functional
  • Target: Reproducible transformations
  • Target: Comprehensive error handling
  • Target: Complete documentation

================================================================================
9. EDA CONCLUSION
================================================================================

🎯 EDA COMPLETION SUMMARY
------------------------------
✅ Analyzed 41,188 banking clients from bmarket.db
✅ Explored 12 raw features including demographics and campaign data
✅ Identified data quality issues requiring Phase 3 cleaning
✅ Documented special values and data type conversion needs
✅ Visualized raw data patterns and distributions
✅ Provided comprehensive Phase 3 preparation roadmap

📋 EDA DELIVERABLES
-------------------------
  ✅ Raw data structure analysis
  ✅ Target variable distribution insights
  ✅ Feature type classification
  ✅ Data quality assessment report
  ✅ Special values inventory
  ✅ Visualization insights
  ✅ Phase 3 cleaning requirements
  ✅ Feature engineering recommendations

🚀 READY FOR PHASE 3: DATA CLEANING AND PREPROCESSING
------------------------------------------------------------
Phase 3 Prerequisites Met:
  ✅ Raw data thoroughly understood
  ✅ Cleaning requirements documented
  ✅ Feature engineering strategy defined
  ✅ Data validation rules identified
  ✅ Business logic requirements captured

Phase 3 Implementation Ready:
  → Age text-to-numeric conversion pipeline
  → Special value handling strategies
  → Target variable binary encoding
  → Feature engineering transformations
  → Data validation framework
  → Quality metrics implementation

================================================================================
EDA PHASE COMPLETED SUCCESSFULLY
PROCEED TO PHASE 3: DATA CLEANING AND PREPROCESSING
================================================================================

🎉 Raw Data EDA Analysis Complete! 🎉
📊 All insights captured for Phase 3 implementation
🔧 Ready to build data cleaning and preprocessing pipeline

================================================================================
10. SAVING EDA ARTIFACTS AND GENERATING DOCUMENTATION
================================================================================

💾 SAVING DATA INSIGHTS TO CSV FILES
----------------------------------------
✅ Saved numerical features statistics: C:\Users\ellie\VSProjects\aisg\aiap20\specs\output\data\numerical_features_statistics.csv
✅ Saved categorical features summary: C:\Users\ellie\VSProjects\aisg\aiap20\specs\output\data\categorical_features_summary.csv
✅ Saved target variable distribution: C:\Users\ellie\VSProjects\aisg\aiap20\specs\output\data\target_variable_distribution.csv
✅ Saved data quality issues: C:\Users\ellie\VSProjects\aisg\aiap20\specs\output\data\data_quality_issues.csv

📄 GENERATING EDA REPORT
-------------------------

```
