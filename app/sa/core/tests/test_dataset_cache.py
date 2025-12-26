import pytest
import pickle
import time
from pathlib import Path

from sa.core.data_manager import DatasetCache


@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def cache(temp_cache_dir):
    return DatasetCache(temp_cache_dir, "test_cache")


@pytest.fixture
def source_files(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"source_{i}.txt"
        f.write_text(f"content {i}")
        files.append(f)
    return files


class TestDatasetCacheInit:
    def test_creates_cache_path(self, temp_cache_dir):
        cache = DatasetCache(temp_cache_dir, "my_cache")
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_file == temp_cache_dir / "my_cache.pkl"


class TestDatasetCacheIsValid:
    def test_returns_false_when_cache_not_exists(self, cache, source_files):
        assert cache.is_valid(source_files) is False

    def test_returns_true_when_cache_newer_than_sources(self, cache, source_files):
        cache.save({"data": "test"})
        time.sleep(0.1)  # Ensure cache is newer
        assert cache.is_valid(source_files) is True

    def test_returns_false_when_source_newer_than_cache(self, cache, source_files):
        cache.save({"data": "test"})
        time.sleep(0.1)
        source_files[0].write_text("updated content")
        assert cache.is_valid(source_files) is False

    def test_returns_false_on_exception(self, cache):
        # Pass invalid source files that will cause an exception
        invalid_files = [Path("/nonexistent/path/file.txt")]
        assert cache.is_valid(invalid_files) is False


class TestDatasetCacheSaveLoad:
    def test_save_creates_cache_directory(self, cache):
        assert not cache.cache_dir.exists()
        cache.save({"key": "value"})
        assert cache.cache_dir.exists()
        assert cache.cache_file.exists()

    def test_load_returns_saved_data(self, cache):
        data = {"features": [1, 2, 3], "labels": ["a", "b", "c"]}
        cache.save(data)
        loaded = cache.load()
        assert loaded == data

    def test_roundtrip_preserves_complex_data(self, cache):
        import numpy as np

        data = {
            "array": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner": [1, 2, 3]},
            "string": "test",
        }
        cache.save(data)
        loaded = cache.load()

        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["nested"] == data["nested"]
        assert loaded["string"] == data["string"]
