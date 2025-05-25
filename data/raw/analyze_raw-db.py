import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def get_db_structure(db_path):
    """Analyze the SQLite database structure and return a DataFrame."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_data = {}
    
    for table in tables:
        table_name = table[0]
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Read full table data
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        table_data[table_name] = {
            'schema': columns,
            'dataframe': df
        }
    
    conn.close()
    return table_data

def analyze_data_quality(df):
    """Analyze data quality issues in the DataFrame."""
    total_rows = len(df)
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / total_rows * 100).round(2)
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    
    # Data type analysis
    data_types = df.dtypes
    
    # Value distributions and unique values
    unique_counts = df.nunique()
    unique_percentage = (unique_counts / total_rows * 100).round(2)
    
    # Check for potential categorical columns
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and unique_counts[col] < 50:
            categorical_cols.append(col)
    
    # Check for potential numeric columns stored as text
    potential_numeric = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column values look like numbers
            sample = df[col].dropna().head(100)
            numeric_count = sum(1 for val in sample if str(val).replace('.', '', 1).isdigit())
            if numeric_count > 0.5 * len(sample):
                potential_numeric.append(col)
    
    # Check for outliers in numeric columns
    outlier_info = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_percentage = (outlier_count / total_rows * 100).round(2)
        outlier_info[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage
        }
    
    # Special values check
    special_values = {}
    for col in df.columns:
        special_vals = {}
        if '999' in df[col].astype(str).values:
            special_vals['999'] = (df[col].astype(str) == '999').sum()
        if 'unknown' in df[col].astype(str).values:
            special_vals['unknown'] = (df[col].astype(str) == 'unknown').sum()
        if special_vals:
            special_values[col] = special_vals
    
    # Distribution of target variable (if present)
    target_distribution = None
    if 'Subscription Status' in df.columns:
        target_distribution = df['Subscription Status'].value_counts(normalize=True) * 100
    
    return {
        'total_rows': total_rows,
        'missing_values': missing_values,
        'missing_percentage': missing_percentage,
        'duplicate_rows': duplicate_rows,
        'data_types': data_types,
        'unique_counts': unique_counts,
        'unique_percentage': unique_percentage,
        'categorical_cols': categorical_cols,
        'potential_numeric': potential_numeric,
        'outlier_info': outlier_info,
        'special_values': special_values,
        'target_distribution': target_distribution
    }

def generate_recommendations(data_quality):
    """Generate recommendations for data preprocessing based on data quality analysis."""
    recommendations = {
        'data_cleaning': [],
        'data_preprocessing': [],
        'feature_engineering': [],
        'exploratory_analysis': []
    }
    
    # Data cleaning recommendations
    if data_quality['duplicate_rows'] > 0:
        recommendations['data_cleaning'].append(f"Remove {data_quality['duplicate_rows']} duplicate rows.")
    
    for col, pct in data_quality['missing_percentage'].items():
        if pct > 0:
            if pct > 20:
                recommendations['data_cleaning'].append(f"High missing values ({pct}%) in '{col}'. Consider dropping the column or using advanced imputation techniques.")
            else:
                recommendations['data_cleaning'].append(f"Missing values ({pct}%) in '{col}'. Impute with appropriate strategy (mean/median for numeric, mode for categorical).")
    
    for col, special_vals in data_quality['special_values'].items():
        for val, count in special_vals.items():
            recommendations['data_cleaning'].append(f"Handle special value '{val}' in '{col}' ({count} occurrences). Consider converting to NaN or a meaningful value.")
    
    # Data preprocessing recommendations
    for col in data_quality['potential_numeric']:
        recommendations['data_preprocessing'].append(f"Convert '{col}' from text to numeric. Extract numeric values if needed.")
    
    for col in data_quality['categorical_cols']:
        recommendations['data_preprocessing'].append(f"Encode categorical column '{col}'. Consider one-hot encoding or label encoding based on cardinality.")
    
    for col, outlier_info in data_quality['outlier_info'].items():
        if outlier_info['percentage'] > 1:
            recommendations['data_preprocessing'].append(f"Handle outliers in '{col}' ({outlier_info['count']} outliers, {outlier_info['percentage']}%). Consider capping, transformation, or binning.")
    
    # Feature engineering recommendations
    if 'Age' in data_quality['data_types']:
        recommendations['feature_engineering'].append("Create age bins/categories from 'Age'.")
    
    if 'Previous Contact Days' in data_quality['data_types']:
        recommendations['feature_engineering'].append("Create recency features from 'Previous Contact Days'.")
    
    if 'Campaign Calls' in data_quality['data_types']:
        recommendations['feature_engineering'].append("Create call frequency categories from 'Campaign Calls'.")
    
    # Add domain-specific feature engineering suggestions
    recommendations['feature_engineering'].append("Create interaction features between demographic attributes (e.g., Age*Education).")
    recommendations['feature_engineering'].append("Create binary flags for key customer attributes (e.g., has_default, has_housing_loan).")
    
    # Exploratory analysis recommendations
    if data_quality['target_distribution'] is not None:
        class_balance = data_quality['target_distribution'].to_dict()
        minor_class_pct = min(class_balance.values())
        recommendations['exploratory_analysis'].append(f"Analyze class imbalance. Minor class represents {minor_class_pct:.2f}% of data.")
    
    recommendations['exploratory_analysis'].append("Explore correlations between features and the target variable.")
    recommendations['exploratory_analysis'].append("Analyze feature distributions and their impact on subscription likelihood.")
    recommendations['exploratory_analysis'].append("Investigate feature relationships through cross-tabulations and visualization.")
    
    return recommendations

def generate_enhanced_report(table_data, recommendations):
    """Generate an enhanced markdown report with recommendations."""
    report = "# Enhanced Database Structure Analysis\n\n"
    
    for table_name, data in table_data.items():
        df = data['dataframe']
        schema = data['schema']
        
        report += f"## Table: `{table_name}`\n"
        report += f"- **Row Count:** {len(df):,}\n\n"
        
        # Schema section
        report += "### Schema\n"
        report += "| Column Name | Data Type | Nullable | Default | Primary Key |\n"
        report += "|-------------|-----------|----------|---------|-------------|\n"
        
        for col in schema:
            col_id, name, dtype, notnull, default, pk = col
            report += f"| {name} | {dtype} | {not bool(notnull)} | {default if default else 'None'} | {'✓' if pk else ''} |\n"
        
        # Data preview
        report += "\n### Data Preview\n"
        report += df.head().to_markdown() + "\n\n"
        
        # Data quality section
        report += "### Data Quality Analysis\n\n"
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            report += "#### Missing Values\n"
            report += "| Column | Missing Count | Missing Percentage |\n"
            report += "|--------|---------------|--------------------|\n"
            for col, count in missing.items():
                if count > 0:
                    pct = round(count / len(df) * 100, 2)
                    report += f"| {col} | {count} | {pct}% |\n"
            report += "\n"
        else:
            report += "No missing values found.\n\n"
        
        # Data types
        report += "#### Data Types and Unique Values\n"
        report += "| Column | Data Type | Unique Values | Unique Percentage |\n"
        report += "|--------|-----------|---------------|-------------------|\n"
        for col, dtype in df.dtypes.items():
            unique_count = df[col].nunique()
            unique_pct = round(unique_count / len(df) * 100, 2)
            report += f"| {col} | {dtype} | {unique_count} | {unique_pct}% |\n"
        report += "\n"
        
        # Value distributions for categorical columns
        report += "#### Categorical Distributions\n"
        for col in df.select_dtypes(include='object').columns:
            if df[col].nunique() < 15:  # Only show for columns with reasonable number of categories
                report += f"**{col}**\n"
                value_counts = df[col].value_counts()
                report += "| Value | Count | Percentage |\n"
                report += "|-------|-------|------------|\n"
                for val, count in value_counts.items():
                    pct = round(count / len(df) * 100, 2)
                    report += f"| {val} | {count} | {pct}% |\n"
                report += "\n"
        
        # Special values check
        special_values = {}
        for col in df.columns:
            if '999' in df[col].astype(str).values or 'unknown' in df[col].astype(str).values:
                special_values[col] = {}
                if '999' in df[col].astype(str).values:
                    special_values[col]['999'] = (df[col].astype(str) == '999').sum()
                if 'unknown' in df[col].astype(str).values:
                    special_values[col]['unknown'] = (df[col].astype(str) == 'unknown').sum()
        
        if special_values:
            report += "#### Special Values\n"
            report += "| Column | Special Value | Count | Percentage |\n"
            report += "|--------|---------------|-------|------------|\n"
            for col, values in special_values.items():
                for val, count in values.items():
                    pct = round(count / len(df) * 100, 2)
                    report += f"| {col} | {val} | {count} | {pct}% |\n"
            report += "\n"
        
        # Target variable distribution (if exists)
        if 'Subscription Status' in df.columns:
            report += "#### Target Variable Distribution\n"
            target_counts = df['Subscription Status'].value_counts()
            report += "| Subscription Status | Count | Percentage |\n"
            report += "|---------------------|-------|------------|\n"
            for val, count in target_counts.items():
                pct = round(count / len(df) * 100, 2)
                report += f"| {val} | {count} | {pct}% |\n"
            report += "\n"
        
        # Numeric column statistics
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            report += "#### Numeric Column Statistics\n"
            stats_df = df[num_cols].describe().T
            stats_df = stats_df.reset_index().rename(columns={'index': 'Column'})
            report += stats_df.to_markdown(index=False, floatfmt=".2f") + "\n\n"
    
    # Recommendations section
    report += "## Recommendations\n\n"
    
    report += "### Data Cleaning\n"
    for rec in recommendations['data_cleaning']:
        report += f"- {rec}\n"
    report += "\n"
    
    report += "### Data Preprocessing\n"
    for rec in recommendations['data_preprocessing']:
        report += f"- {rec}\n"
    report += "\n"
    
    report += "### Feature Engineering\n"
    for rec in recommendations['feature_engineering']:
        report += f"- {rec}\n"
    report += "\n"
    
    report += "### Exploratory Data Analysis\n"
    for rec in recommendations['exploratory_analysis']:
        report += f"- {rec}\n"
    report += "\n"
    
    # Add a conclusion summarizing key findings and next steps
    report += "## Conclusion\n\n"
    report += "The database contains banking client information and marketing campaign data with several quality issues that need addressing. "
    report += "Key challenges include handling special values like '999' and 'unknown', converting text fields to appropriate data types, "
    report += "and addressing potential class imbalance in the target variable.\n\n"
    report += "The recommended approach is to first clean the data by handling missing and special values, "
    report += "then preprocess by standardizing data types and encoding categorical variables, "
    report += "followed by feature engineering to extract more predictive information from the existing features. "
    report += "The exploratory analysis should focus on understanding feature distributions and their relationship with the target variable.\n\n"
    
    return report

if __name__ == "__main__":
    # Analyze database
    db_path = "data/raw/bmarket.db"
    table_data = get_db_structure(db_path)
    
    # Generate comprehensive analysis for the first table (assuming it's the only one)
    table_name = list(table_data.keys())[0]
    df = table_data[table_name]['dataframe']
    
    # Analyze data quality
    data_quality = analyze_data_quality(df)
    
    # Generate recommendations
    recommendations = generate_recommendations(data_quality)
    
    # Generate and save report
    report = generate_enhanced_report(table_data, recommendations)
    
    output_path = Path("specs/raw-db-structure.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Enhanced database structure analysis saved to {output_path}")
