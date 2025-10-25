"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.load_data import (
    load_adult_data,
    generate_synthetic_data,
    save_processed_data,
    load_processed_data
)


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_adult_data(self):
        """Test Adult dataset loading."""
        X_train, X_test, y_train, y_test = load_adult_data(
            test_size=0.2,
            random_state=42,
            from_cache=False
        )
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check binary target
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            test_size=0.2,
            random_state=42
        )
        
        assert isinstance(X_train, pd.DataFrame)
        assert len(X_train) > 0
        assert X_train.shape[1] == 10
        
        assert len(y_train) == len(X_train)
        assert set(y_train.unique()) == {0, 1}
    
    def test_generate_synthetic_data_with_demographics(self):
        """Test synthetic data with demographic features."""
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=1000,
            n_features=5,
            add_demographics=True,
            random_state=42
        )
        
        # Check demographic features exist
        assert 'age' in X_train.columns
        assert 'gender' in X_train.columns
        assert 'race' in X_train.columns
    
    def test_data_split_sizes(self):
        """Test data split proportions."""
        test_size = 0.3
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=1000,
            test_size=test_size,
            random_state=42
        )
        
        total_samples = len(X_train) + len(X_test)
        actual_test_ratio = len(X_test) / total_samples
        
        # Allow small tolerance
        assert abs(actual_test_ratio - test_size) < 0.05


class TestDataSaving:
    """Test data saving and loading."""
    
    def test_save_and_load_processed_data(self, tmp_path):
        """Test saving and loading processed data."""
        # Generate test data
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            n_samples=100,
            random_state=42
        )
        
        # Temporarily change config path
        from src.utils import config as cfg
        original_path = cfg.config.data.processed_data_dir
        cfg.config.data.processed_data_dir = tmp_path
        
        try:
            # Save data
            save_processed_data(X_train, X_test, y_train, y_test, "test")
            
            # Load data
            X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = \
                load_processed_data("test")
            
            # Verify
            pd.testing.assert_frame_equal(X_train, X_train_loaded)
            pd.testing.assert_frame_equal(X_test, X_test_loaded)
            pd.testing.assert_series_equal(y_train, y_train_loaded)
            pd.testing.assert_series_equal(y_test, y_test_loaded)
        
        finally:
            # Restore original path
            cfg.config.data.processed_data_dir = original_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

