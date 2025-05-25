# AIAP20 Test Suite

Streamlined testing structure focusing on critical path verification.

## Testing Philosophy

**Focus on critical path, not exhaustive coverage**
- Minimal setup, maximum utility
- Business value: Ensure development environment works correctly
- Efficiency over completeness

## Test Structure

```
tests/
├── unit/           # Core function verification
├── integration/    # Component interactions  
├── smoke/          # Quick pipeline verification
├── conftest.py     # Lightweight fixtures
└── README.md       # This file
```

## Test Categories

### 🔧 Unit Tests (`tests/unit/`)
- **Purpose**: Test isolated functionality with minimal dependencies
- **Speed**: Fast (< 1 second per test)
- **Focus**: Individual functions and classes
- **Run with**: `pytest tests/unit/ -m unit`

### 🔗 Integration Tests (`tests/integration/`)
- **Purpose**: Test how different modules work together
- **Speed**: Medium (1-10 seconds per test)
- **Focus**: Component interactions and data flow
- **Run with**: `pytest tests/integration/ -m integration`

### 💨 Smoke Tests (`tests/smoke/`)
- **Purpose**: Quick pipeline verification and environment checks
- **Speed**: Fast (< 5 seconds total)
- **Focus**: Basic functionality and setup verification
- **Run with**: `pytest tests/smoke/ -m smoke`

## Quick Start

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Smoke tests (recommended first run)
pytest -m smoke

# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# Exclude slow tests
pytest -m "not slow"
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

## Test Fixtures

The `conftest.py` provides lightweight, reusable fixtures:

- `project_root`: Project root directory path
- `sample_dataframe`: Minimal test dataframe
- `temp_database`: Temporary SQLite database
- `mock_config`: Mock configuration for testing
- `sample_database_path`: Path to real database (if exists)

## Writing New Tests

### Unit Test Template
```python
class TestYourModule:
    def test_function_with_valid_input(self):
        """Test function behavior with valid input."""
        # Arrange
        input_data = "test_input"
        
        # Act
        result = your_function(input_data)
        
        # Assert
        assert result == expected_output
```

### Integration Test Template
```python
class TestModuleIntegration:
    def test_module_a_and_b_integration(self, sample_dataframe):
        """Test integration between modules."""
        # Test data flow between components
        processed_data = module_a.process(sample_dataframe)
        result = module_b.analyze(processed_data)
        
        assert result is not None
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_isolated_function():
    pass

@pytest.mark.integration  
def test_component_interaction():
    pass

@pytest.mark.smoke
def test_basic_functionality():
    pass

@pytest.mark.slow
def test_long_running_process():
    pass
```

## Best Practices

1. **Keep tests simple and focused**
2. **Use descriptive test names**
3. **Follow Arrange-Act-Assert pattern**
4. **Use fixtures for common setup**
5. **Test both happy path and edge cases**
6. **Mock external dependencies**
7. **Keep test data minimal but realistic**

## Troubleshooting

### Common Issues

**Import errors**: Ensure the project is installed in development mode:
```bash
pip install -e .
```

**Database tests failing**: Ensure the sample database exists:
```bash
python data/raw/download_db.py
```

**Slow tests**: Run only fast tests during development:
```bash
pytest -m "not slow"
```

### Getting Help

1. Check test output for specific error messages
2. Run tests with increased verbosity: `pytest -v`
3. Run a single test: `pytest tests/path/to/test.py::test_function`
4. Use pytest's debugging: `pytest --pdb`

## Business Value

This testing structure ensures:
- ✅ Development environment works correctly
- ✅ Core functionality is verified
- ✅ Component interactions are tested
- ✅ Quick feedback during development
- ✅ Confidence in code changes
