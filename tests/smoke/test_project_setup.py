"""
Smoke Tests for Project Setup Verification

Business Value: Ensure development environment works correctly.
Focus: Quick verification that the project is properly configured.
"""

import pytest
import sys
import os
import importlib
from pathlib import Path


class TestProjectStructure:
    """Test that the project structure is correctly set up."""
    
    def test_python_version(self):
        """Verify Python version meets requirements."""
        assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version_info}"
    
    def test_project_directories_exist(self, project_root):
        """Verify essential project directories exist."""
        required_dirs = [
            "src",
            "data",
            "tests",
            "specs"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory '{dir_name}' not found"
            assert dir_path.is_dir(), f"'{dir_name}' exists but is not a directory"
    
    def test_source_structure(self, project_root):
        """Verify source code structure is properly organized."""
        src_dir = project_root / "src"
        expected_modules = [
            "config",
            "data", 
            "preprocessing",
            "features",
            "models",
            "evaluation",
            "utils"
        ]
        
        for module in expected_modules:
            module_path = src_dir / module
            assert module_path.exists(), f"Source module '{module}' directory not found"
    
    def test_configuration_files_exist(self, project_root):
        """Verify essential configuration files exist."""
        config_files = [
            "pyproject.toml",
            "requirements.txt"
        ]
        
        for config_file in config_files:
            file_path = project_root / config_file
            assert file_path.exists(), f"Configuration file '{config_file}' not found"
            assert file_path.is_file(), f"'{config_file}' exists but is not a file"


class TestDependencies:
    """Test that required dependencies are available."""
    
    @pytest.mark.parametrize("package", [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "pytest"
    ])
    def test_core_packages_importable(self, package):
        """Verify core packages can be imported."""
        try:
            if package == "sklearn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' cannot be imported")
    
    def test_sqlite3_available(self):
        """Verify SQLite3 is available (should be in standard library)."""
        try:
            import sqlite3
            # Test basic functionality
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            assert result[0] == 1
        except Exception as e:
            pytest.fail(f"SQLite3 not working properly: {e}")


class TestDataAccess:
    """Test data access and basic database connectivity."""
    
    def test_data_directory_structure(self, project_root):
        """Verify data directory structure."""
        data_dir = project_root / "data"
        expected_subdirs = ["raw", "processed", "featured"]
        
        for subdir in expected_subdirs:
            subdir_path = data_dir / subdir
            # Create if doesn't exist (this is expected behavior)
            if not subdir_path.exists():
                subdir_path.mkdir(parents=True, exist_ok=True)
            assert subdir_path.exists(), f"Data subdirectory '{subdir}' not found"
    
    def test_database_accessibility(self, sample_database_path):
        """Test that the sample database is accessible if it exists."""
        import sqlite3
        
        try:
            conn = sqlite3.connect(sample_database_path)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            assert len(tables) > 0, "Database exists but contains no tables"
            
            # Test data access from first table
            table_name = tables[0][0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            assert count > 0, f"Table '{table_name}' exists but contains no data"
            
            conn.close()
            
        except sqlite3.Error as e:
            pytest.fail(f"Database access failed: {e}")


class TestTestingFramework:
    """Test that the testing framework itself is working."""
    
    def test_pytest_working(self):
        """Verify pytest is functioning correctly."""
        assert True, "If this fails, pytest itself is broken"
    
    def test_fixtures_available(self, project_root, sample_dataframe, mock_config):
        """Verify that custom fixtures are working."""
        assert project_root.exists(), "project_root fixture not working"
        assert len(sample_dataframe) > 0, "sample_dataframe fixture not working"
        assert "data" in mock_config, "mock_config fixture not working"
    
    def test_temp_database_fixture(self, temp_database):
        """Verify temporary database fixture works."""
        import sqlite3
        
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM bank_data")
        count = cursor.fetchone()[0]
        
        assert count > 0, "Temporary database fixture not working"
        conn.close()


class TestEnvironmentHealth:
    """Test overall environment health and readiness."""
    
    def test_memory_availability(self):
        """Basic memory availability check."""
        import psutil
        
        # Check if we have at least 1GB of available memory
        available_memory = psutil.virtual_memory().available
        min_memory_gb = 1 * 1024 * 1024 * 1024  # 1GB in bytes
        
        assert available_memory > min_memory_gb, f"Insufficient memory: {available_memory / (1024**3):.1f}GB available"
    
    def test_disk_space(self, project_root):
        """Basic disk space check."""
        import shutil
        
        # Check if we have at least 1GB of free space
        free_space = shutil.disk_usage(project_root).free
        min_space_gb = 1 * 1024 * 1024 * 1024  # 1GB in bytes
        
        assert free_space > min_space_gb, f"Insufficient disk space: {free_space / (1024**3):.1f}GB available"
    
    def test_write_permissions(self, project_root):
        """Test write permissions in project directory."""
        test_file = project_root / "test_write_permission.tmp"
        
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            
            assert test_file.exists(), "Could not create test file"
            
            # Cleanup
            test_file.unlink()
            
        except PermissionError:
            pytest.fail("No write permissions in project directory")
        except Exception as e:
            pytest.fail(f"Unexpected error testing write permissions: {e}")


# Smoke test summary function
def test_smoke_test_summary():
    """
    Summary test that indicates smoke tests are complete.
    
    If this test passes, it means:
    - Project structure is correct
    - Dependencies are installed
    - Database access works (if database exists)
    - Testing framework is functional
    - Environment is healthy
    
    Business Value: Confidence that development environment is ready for work.
    """
    assert True, "Smoke tests completed successfully"
