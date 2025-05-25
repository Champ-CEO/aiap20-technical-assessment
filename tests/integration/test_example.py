"""
Example Integration Tests

This file demonstrates the integration testing structure and provides
templates for testing component interactions.

Focus: Testing how different modules work together.
"""

import pytest
import pandas as pd
import sqlite3


class TestDatabaseIntegration:
    """Example integration tests for database operations."""
    
    def test_database_connection_and_query(self, temp_database):
        """Test database connection and basic query operations."""
        conn = sqlite3.connect(temp_database)
        
        try:
            # Test connection
            cursor = conn.cursor()
            
            # Test table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            assert len(tables) > 0, "No tables found in test database"
            
            # Test data retrieval
            cursor.execute("SELECT * FROM bank_data LIMIT 5")
            rows = cursor.fetchall()
            assert len(rows) > 0, "No data retrieved from test database"
            
            # Test specific query
            cursor.execute("SELECT COUNT(*) FROM bank_data WHERE y = 'yes'")
            count = cursor.fetchone()[0]
            assert isinstance(count, int), "Count query did not return integer"
            
        finally:
            conn.close()
    
    def test_pandas_sqlite_integration(self, temp_database):
        """Test pandas and SQLite integration."""
        # Test reading data with pandas
        df = pd.read_sql_query("SELECT * FROM bank_data", sqlite3.connect(temp_database))
        
        assert not df.empty, "Pandas failed to read data from SQLite"
        assert len(df.columns) > 0, "No columns retrieved"
        assert 'y' in df.columns, "Target column 'y' not found"
        
        # Test data types
        assert df['age'].dtype in ['int64', 'int32'], "Age column should be integer"
        assert df['job'].dtype == 'object', "Job column should be object/string"


class TestDataProcessingPipeline:
    """Example integration tests for data processing pipeline."""
    
    def test_data_loading_and_basic_processing(self, sample_dataframe):
        """Test basic data loading and processing workflow."""
        # Simulate data loading (using fixture)
        df = sample_dataframe.copy()
        
        # Test basic processing steps
        initial_shape = df.shape
        assert initial_shape[0] > 0, "No rows in dataframe"
        assert initial_shape[1] > 0, "No columns in dataframe"
        
        # Test data cleaning simulation
        df_cleaned = df.dropna()
        assert len(df_cleaned) <= len(df), "Cleaned data should not have more rows"
        
        # Test feature selection simulation
        feature_columns = [col for col in df.columns if col != 'y']
        X = df[feature_columns]
        y = df['y']
        
        assert len(X.columns) > 0, "No feature columns selected"
        assert len(y) == len(df), "Target variable length mismatch"
    
    def test_train_test_split_integration(self, sample_dataframe):
        """Test train-test split integration."""
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        feature_columns = [col for col in sample_dataframe.columns if col != 'y']
        X = sample_dataframe[feature_columns]
        y = sample_dataframe['y']
        
        # Test train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Verify split
        assert len(X_train) + len(X_test) == len(X), "Split sizes don't add up"
        assert len(y_train) + len(y_test) == len(y), "Target split sizes don't add up"
        assert len(X_train) == len(y_train), "Training set size mismatch"
        assert len(X_test) == len(y_test), "Test set size mismatch"


class TestModelTrainingIntegration:
    """Example integration tests for model training workflow."""
    
    def test_basic_model_training_workflow(self, sample_dataframe):
        """Test basic model training integration."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        df = sample_dataframe.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['job', 'marital', 'education', 'housing', 'loan']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = le.fit_transform(df[col])
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'y']
        X = df[feature_columns]
        y = le.fit_transform(df['y'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test), "Prediction count mismatch"
        assert probabilities.shape[0] == len(X_test), "Probability shape mismatch"
        assert probabilities.shape[1] == 2, "Should have 2 classes for binary classification"


class TestConfigurationIntegration:
    """Example integration tests for configuration handling."""
    
    def test_config_and_model_integration(self, mock_config):
        """Test configuration integration with model parameters."""
        # Test that configuration can be used to initialize models
        model_config = mock_config['models']['random_forest']
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Test model initialization with config
        model = RandomForestClassifier(**model_config)
        
        assert model.n_estimators == model_config['n_estimators']
        assert model.random_state == model_config['random_state']
    
    def test_config_and_data_processing_integration(self, mock_config, sample_dataframe):
        """Test configuration integration with data processing."""
        data_config = mock_config['data']
        
        # Test using config for data processing
        target_column = data_config['target_column']
        test_size = data_config['test_size']
        random_state = data_config['random_state']
        
        assert target_column in sample_dataframe.columns, "Target column not found in data"
        assert 0 < test_size < 1, "Invalid test size in config"
        assert isinstance(random_state, int), "Random state should be integer"


# Template for future integration tests
class TestTemplateForFutureIntegrationTests:
    """
    Template class for future integration tests.
    
    When adding new module interactions, copy this template and modify for specific needs.
    """
    
    def test_placeholder_integration(self):
        """Placeholder integration test - replace with actual tests."""
        # This test should be replaced when actual integration scenarios are implemented
        assert True, "Replace this with actual integration tests"
    
    # Example integration test patterns:
    
    # def test_module_a_and_module_b_integration(self):
    #     """Test integration between module A and module B."""
    #     pass
    
    # def test_end_to_end_workflow(self):
    #     """Test complete workflow from start to finish."""
    #     pass
    
    # def test_error_propagation_between_modules(self):
    #     """Test how errors propagate between integrated modules."""
    #     pass
    
    # def test_data_flow_between_components(self):
    #     """Test data flow and transformations between components."""
    #     pass
