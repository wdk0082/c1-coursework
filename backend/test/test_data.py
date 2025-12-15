"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from fivedreg.data.load_data import load_data, _handle_missing_values, _standardize_features, _split_data


class TestLoadData:
    """Tests for the load_data function."""

    def test_load_npz_file(self, temp_npz_file):
        """Test loading data from .npz file without splits."""
        result = load_data(temp_npz_file, standardize=False)
        assert 'X' in result
        assert 'y' in result
        assert result['X'].shape[1] == 5

    def test_load_npz_with_splits(self, temp_npz_file):
        """Test loading .npz file with train/val/test splits."""
        result = load_data(temp_npz_file, standardize=False, split_ratios=(0.7, 0.15, 0.15))
        assert 'X_train' in result
        assert 'y_train' in result
        assert result['X_train'].shape[1] == 5

    def test_load_npz_alternative_keys(self, temp_npz_alt_keys):
        """Test loading .npz with alternative key names."""
        result = load_data(temp_npz_alt_keys, standardize=False)
        assert result['X'].shape[1] == 5

    def test_load_pkl_file(self, temp_pkl_file):
        """Test loading data from .pkl file."""
        result = load_data(temp_pkl_file, standardize=False)
        assert 'X' in result
        assert result['X'].shape[1] == 5

    def test_invalid_file_extension(self):
        """Test that invalid file extensions raise an error."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'invalid')
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_data(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_file(self):
        """Test that missing files raise an error."""
        with pytest.raises(FileNotFoundError):
            load_data('/nonexistent/path/file.npz')

    def test_wrong_feature_dimensions(self, tmp_path):
        """Test that wrong feature dimensions raise an error."""
        X = np.random.randn(100, 3).astype(np.float32)  # Wrong: 3 features
        y = np.random.randn(100).astype(np.float32)
        path = tmp_path / "test.npz"
        np.savez(path, X=X, y=y)
        with pytest.raises(ValueError, match="5 features"):
            load_data(str(path))

    def test_sample_count_mismatch(self, tmp_path):
        """Test that mismatched sample counts raise an error."""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)  # Wrong: different count
        path = tmp_path / "test.npz"
        np.savez(path, X=X, y=y)
        with pytest.raises(ValueError, match="mismatch"):
            load_data(str(path))

    def test_standardization_applied(self, temp_npz_file):
        """Test that standardization is applied correctly."""
        result = load_data(temp_npz_file, standardize=True, split_ratios=(0.7, 0.15, 0.15))
        assert np.abs(result['X_train'].mean()) < 0.5
        assert 'scaler' in result
        assert result['scaler'] is not None

    def test_split_ratios(self, temp_npz_file):
        """Test that split ratios are applied correctly."""
        result = load_data(temp_npz_file, split_ratios=(0.6, 0.2, 0.2), standardize=False)
        total = len(result['X_train']) + len(result['X_val']) + len(result['X_test'])
        assert total == 100
        assert len(result['X_train']) == 60
        assert len(result['X_val']) == 20
        assert len(result['X_test']) == 20


class TestMissingValues:
    """Tests for missing value handling."""

    def test_ignore_strategy(self, sample_data_with_nan):
        """Test that 'ignore' removes rows with NaN."""
        X, y = sample_data_with_nan
        X_clean, y_clean = _handle_missing_values(X, y, strategy='ignore')
        assert not np.isnan(X_clean).any()
        assert len(X_clean) == len(X) - 3

    def test_mean_strategy(self, sample_data_with_nan):
        """Test that 'mean' fills NaN with column means."""
        X, y = sample_data_with_nan
        X_clean, y_clean = _handle_missing_values(X, y, strategy='mean')
        assert not np.isnan(X_clean).any()
        assert len(X_clean) == len(X)

    def test_median_strategy(self, sample_data_with_nan):
        """Test that 'median' fills NaN with column medians."""
        X, y = sample_data_with_nan
        X_clean, y_clean = _handle_missing_values(X, y, strategy='median')
        assert not np.isnan(X_clean).any()
        assert len(X_clean) == len(X)

    def test_zero_strategy(self, sample_data_with_nan):
        """Test that 'zero' fills NaN with zeros."""
        X, y = sample_data_with_nan
        X_clean, y_clean = _handle_missing_values(X, y, strategy='zero')
        assert not np.isnan(X_clean).any()
        assert X_clean[5, 0] == 0.0

    def test_invalid_strategy(self, sample_data):
        """Test that invalid strategy raises an error."""
        X, y = sample_data
        with pytest.raises(ValueError, match="Unknown missing value strategy"):
            _handle_missing_values(X, y, strategy='invalid')


class TestStandardization:
    """Tests for feature standardization."""

    def test_standardize_features(self):
        """Test that standardization produces zero mean and unit variance."""
        X = np.random.randn(100, 5).astype(np.float32) * 10 + 5
        X_std, scaler = _standardize_features(X, fit=True)
        assert np.abs(X_std.mean(axis=0)).max() < 0.1
        assert np.abs(X_std.std(axis=0) - 1).max() < 0.1

    def test_scaler_reuse(self):
        """Test that scaler can be reused on new data."""
        X = np.random.randn(100, 5).astype(np.float32) * 10 + 5
        X_std, scaler = _standardize_features(X, fit=True)
        X_new = np.random.randn(20, 5).astype(np.float32) * 10 + 5
        X_new_std, _ = _standardize_features(X_new, fit=False, scaler=scaler)
        assert X_new_std.shape == (20, 5)


class TestDataSplitting:
    """Tests for data splitting."""

    def test_split_ratios(self):
        """Test that splits respect the given ratios."""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
            X, y, split_ratios=(0.7, 0.15, 0.15), random_seed=42
        )
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15

    def test_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        result1 = _split_data(X, y, split_ratios=(0.7, 0.15, 0.15), random_seed=42)
        result2 = _split_data(X, y, split_ratios=(0.7, 0.15, 0.15), random_seed=42)
        np.testing.assert_array_equal(result1[0], result2[0])

    def test_invalid_ratios(self, temp_npz_file):
        """Test that invalid ratios raise an error via load_data."""
        with pytest.raises(ValueError):
            load_data(temp_npz_file, split_ratios=(0.5, 0.5, 0.5))
