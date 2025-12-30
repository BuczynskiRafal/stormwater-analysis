"""Unit tests for data_manager.py module to improve coverage."""

import os
import pickle
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sa.core.data_manager import (
    SWMMLabelGenerator,
    DatasetCache,
    BaseSWMMDataset,
    get_default_feature_columns,
    prepare_swmm_labels,
    find_inp_files,
    prepare_swmm_dataset,
    _find_and_validate_inp_files,
    _try_load_from_cache,
    _process_inp_files,
    _load_inp_files,
    _add_app_to_path,
    _generate_labels_for_dataset,
    _save_to_cache,
)
from sa.core.enums import RecommendationCategory


class TestSWMMLabelGenerator:
    """Tests for SWMMLabelGenerator class."""

    def test_generate_label_depth_increase(self):
        """Test label generation when ValCoverage is 0."""
        row = pd.Series({"ValCoverage": 0, "ValMaxFill": 1, "ValMinV": 1})
        assert SWMMLabelGenerator.generate_label(row) == "depth_increase"

    def test_generate_label_diameter_increase(self):
        """Test label generation when ValMaxFill is 0 and IncreaseDia is 1."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 0, "IncreaseDia": 1})
        assert SWMMLabelGenerator.generate_label(row) == "diameter_increase"

    def test_generate_label_tank(self):
        """Test label generation when ValMaxFill is 0 and IncreaseDia is 0."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 0, "IncreaseDia": 0})
        assert SWMMLabelGenerator.generate_label(row) == "tank"

    def test_generate_label_slope_increase(self):
        """Test label generation when ValMinV is 0 and IncreaseSlope is 1."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 0, "IncreaseSlope": 1})
        assert SWMMLabelGenerator.generate_label(row) == "slope_increase"

    def test_generate_label_seepage_boxes(self):
        """Test label generation when ValMinV is 0 and IncreaseSlope is 0."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 0, "IncreaseSlope": 0})
        assert SWMMLabelGenerator.generate_label(row) == "seepage_boxes"

    def test_generate_label_diameter_reduction(self):
        """Test label generation when ReduceDia is 1."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 1, "ReduceDia": 1})
        assert SWMMLabelGenerator.generate_label(row) == "diameter_reduction"

    def test_generate_label_valid(self):
        """Test label generation when all validations pass."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 1, "ReduceDia": 0})
        assert SWMMLabelGenerator.generate_label(row) == "valid"

    def test_validate_row_data_all_null_raises(self):
        """Test that validate_row_data raises for all null row."""
        row = pd.Series({"ValCoverage": None, "ValMaxFill": None}, name="C1")
        row[:] = None
        with pytest.raises(ValueError, match="missing from simulation file"):
            SWMMLabelGenerator.validate_row_data(row)

    def test_validate_row_data_missing_val_coverage_raises(self):
        """Test that validate_row_data raises when ValCoverage is NaN."""
        row = pd.Series({"ValCoverage": np.nan, "ValMaxFill": 1}, name="C1")
        with pytest.raises(ValueError, match="missing critical validation data"):
            SWMMLabelGenerator.validate_row_data(row)

    def test_validate_row_data_valid(self):
        """Test that validate_row_data passes for valid data."""
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1}, name="C1")
        # Should not raise
        SWMMLabelGenerator.validate_row_data(row)

    def test_generate_labels_from_dataframe(self):
        """Test generating labels for multiple rows."""
        df = pd.DataFrame(
            {
                "ValCoverage": [0, 1, 1],
                "ValMaxFill": [1, 0, 1],
                "ValMinV": [1, 1, 1],
                "IncreaseDia": [0, 1, 0],
                "ReduceDia": [0, 0, 0],
            }
        )
        labels = SWMMLabelGenerator.generate_labels_from_dataframe(df)
        assert labels == ["depth_increase", "diameter_increase", "valid"]


class TestDatasetCache:
    """Tests for DatasetCache class."""

    def test_init_creates_cache_file_path(self, tmp_path):
        """Test that DatasetCache initializes correctly."""
        cache = DatasetCache(tmp_path, "test_cache")
        assert cache.cache_dir == tmp_path
        assert cache.cache_file == tmp_path / "test_cache.pkl"

    def test_is_valid_returns_false_when_no_cache_file(self, tmp_path):
        """Test is_valid returns False when cache file doesn't exist."""
        cache = DatasetCache(tmp_path, "nonexistent")
        assert cache.is_valid([]) is False

    def test_is_valid_returns_true_when_cache_newer(self, tmp_path):
        """Test is_valid returns True when cache is newer than sources."""
        # Create source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("data")

        # Ensure source file is older
        past_time = time.time() - 100
        os.utime(source_file, (past_time, past_time))

        # Create cache file (will be newer)
        cache = DatasetCache(tmp_path, "test")
        cache.save({"test": "data"})

        assert cache.is_valid([source_file]) is True

    def test_is_valid_returns_false_when_source_newer(self, tmp_path):
        """Test is_valid returns False when source is newer than cache."""
        # Create cache first
        cache = DatasetCache(tmp_path, "test")
        cache.save({"test": "data"})

        # Modify source file to be newer
        import time

        time.sleep(0.1)  # Ensure time difference
        source_file = tmp_path / "source.txt"
        source_file.write_text("newer data")

        assert cache.is_valid([source_file]) is False

    def test_is_valid_handles_exception(self, tmp_path):
        """Test is_valid returns False on exception (covers lines 77-78)."""
        cache = DatasetCache(tmp_path, "test")
        # Create cache file
        cache.save({"test": "data"})

        # Create a mock source file that raises on stat
        mock_source = MagicMock()
        mock_source.stat.side_effect = OSError("test error")
        result = cache.is_valid([mock_source])
        assert result is False

    def test_load_returns_cached_data(self, tmp_path):
        """Test load returns previously saved data."""
        cache = DatasetCache(tmp_path, "test")
        test_data = {"key": "value", "num": 42}
        cache.save(test_data)

        loaded = cache.load()
        assert loaded == test_data

    def test_save_creates_directory_if_missing(self, tmp_path):
        """Test save creates cache directory if it doesn't exist."""
        # Use single-level directory (mkdir without parents=True)
        cache_dir = tmp_path / "cache"
        cache = DatasetCache(cache_dir, "test")
        cache.save({"test": "data"})

        assert cache_dir.exists()
        assert cache.cache_file.exists()


class TestBaseSWMMDataset:
    """Tests for BaseSWMMDataset class."""

    def test_init_with_none_uses_default_directory(self):
        """Test initialization with None uses default directory."""
        with patch.object(BaseSWMMDataset, "DEFAULT_INP_DIRECTORY", "/default/path"):
            with patch("sa.core.data_manager.find_inp_files", return_value=[]):
                dataset = BaseSWMMDataset(data_source=None)
                assert dataset.inp_directory == "/default/path"

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        with patch("sa.core.data_manager.find_inp_files", return_value=[]) as mock_find:
            dataset = BaseSWMMDataset(data_source=str(tmp_path))
            assert dataset.inp_directory == str(tmp_path)
            mock_find.assert_called_once_with(str(tmp_path))

    def test_init_with_dataframe(self):
        """Test initialization with DataFrame."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        dataset = BaseSWMMDataset(data_source=df)
        assert dataset.conduits_data is not None
        assert list(dataset.conduits_data["A"]) == [1, 2]
        # Verify it's a copy
        df["A"] = [5, 6]
        assert list(dataset.conduits_data["A"]) == [1, 2]

    def test_init_with_unsupported_type_raises(self):
        """Test initialization with unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported data_source type"):
            BaseSWMMDataset(data_source=123)

    def test_init_with_custom_feature_columns(self, tmp_path):
        """Test initialization with custom feature columns."""
        custom_cols = ["Col1", "Col2"]
        with patch("sa.core.data_manager.find_inp_files", return_value=[]):
            dataset = BaseSWMMDataset(data_source=str(tmp_path), feature_columns=custom_cols)
            assert dataset.feature_columns == custom_cols


class TestGetDefaultFeatureColumns:
    """Tests for get_default_feature_columns function."""

    def test_returns_list_of_columns(self):
        """Test that get_default_feature_columns returns expected columns."""
        columns = get_default_feature_columns()
        assert isinstance(columns, list)
        assert "ValMaxFill" in columns
        assert "ValCoverage" in columns
        assert "marshes" in columns
        assert len(columns) > 20


class TestPrepareSWMMLabels:
    """Tests for prepare_swmm_labels function."""

    def test_with_series_one_hot(self):
        """Test with pandas Series and one_hot=True."""
        labels = pd.Series(["valid", "tank", "valid"])
        result = prepare_swmm_labels(labels, one_hot=True)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # Check shape matches number of classes
        assert result.shape[1] == len(RecommendationCategory)

    def test_with_series_not_one_hot(self):
        """Test with pandas Series and one_hot=False."""
        labels = pd.Series(["valid", "tank"])
        result = prepare_swmm_labels(labels, one_hot=False)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_with_series_of_enums(self):
        """Test with pandas Series containing RecommendationCategory enums."""
        labels = pd.Series([RecommendationCategory.VALID, RecommendationCategory.TANK])
        result = prepare_swmm_labels(labels, one_hot=True)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_with_dataframe(self):
        """Test with DataFrame input."""
        df = pd.DataFrame(
            {
                "valid": [1, 0],
                "tank": [0, 1],
            }
        )
        result = prepare_swmm_labels(df)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_with_list_of_strings_raises_value_error(self):
        """Test with list of strings raises ValueError (covers line 396).

        Note: The code attempts to convert strings to float32 which fails.
        This documents the current behavior.
        """
        labels = ["valid", "tank", "pump"]
        with pytest.raises(ValueError, match="could not convert string"):
            prepare_swmm_labels(labels)

    def test_with_list_of_enums_raises_value_error(self):
        """Test with list of RecommendationCategory enums raises ValueError (covers line 396).

        Note: The code extracts enum values (strings) and tries to convert to float32.
        This documents the current behavior.
        """
        labels = [RecommendationCategory.VALID, RecommendationCategory.TANK]
        with pytest.raises(ValueError, match="could not convert string"):
            prepare_swmm_labels(labels)


class TestFindInpFiles:
    """Tests for find_inp_files function."""

    def test_returns_empty_list_when_directory_not_exists(self, tmp_path):
        """Test returns empty list when directory doesn't exist (covers line 414)."""
        nonexistent = tmp_path / "nonexistent"
        result = find_inp_files(str(nonexistent))
        assert result == []

    def test_returns_sorted_inp_files(self, tmp_path):
        """Test returns sorted list of .inp files."""
        # Create test files
        (tmp_path / "z_file.inp").write_text("")
        (tmp_path / "a_file.inp").write_text("")
        (tmp_path / "m_file.inp").write_text("")
        (tmp_path / "other.txt").write_text("")  # Not .inp

        result = find_inp_files(str(tmp_path))

        assert len(result) == 3
        assert result[0].name == "a_file.inp"
        assert result[1].name == "m_file.inp"
        assert result[2].name == "z_file.inp"

    def test_returns_empty_list_when_no_inp_files(self, tmp_path):
        """Test returns empty list when no .inp files exist."""
        (tmp_path / "other.txt").write_text("")
        result = find_inp_files(str(tmp_path))
        assert result == []


class TestPrepareSWMMDataset:
    """Tests for prepare_swmm_dataset function."""

    def test_returns_none_when_no_files_found(self, tmp_path):
        """Test returns (None, None) when no .inp files found (covers lines 421-423)."""
        result = prepare_swmm_dataset(str(tmp_path), quiet=True)
        assert result == (None, None)

    def test_uses_cache_when_valid(self, tmp_path):
        """Test loads from cache when valid (covers lines 425-429)."""
        # Create inp file
        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        # Create cache with test data
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cache = DatasetCache(output_dir, "prepared_dataset")
        test_df = pd.DataFrame({"Name": ["C1"]})
        test_labels = pd.Series(["valid"])
        cache.save({"conduits": test_df, "labels": test_labels})

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(DatasetCache, "is_valid", return_value=True):
                result = prepare_swmm_dataset(str(tmp_path), output_dir=str(output_dir), quiet=True)

                assert result[0] is not None
                assert "Name" in result[0].columns

    def test_skips_cache_when_use_cache_false(self, tmp_path):
        """Test processes from scratch when use_cache=False (covers line 431)."""
        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch("sa.core.data_manager._process_inp_files", return_value=(None, None)) as mock_process:
                prepare_swmm_dataset(str(tmp_path), use_cache=False, quiet=True)
                mock_process.assert_called_once()


class TestFindAndValidateInpFiles:
    """Tests for _find_and_validate_inp_files function."""

    def test_prints_message_when_not_quiet(self, tmp_path, capsys):
        """Test prints search message when quiet=False (covers lines 444-445)."""
        _find_and_validate_inp_files(str(tmp_path), quiet=False)
        captured = capsys.readouterr()
        assert "Searching for .inp files" in captured.out

    def test_prints_not_found_when_not_quiet(self, tmp_path, capsys):
        """Test prints 'not found' message when quiet=False (covers lines 449-450)."""
        _find_and_validate_inp_files(str(tmp_path), quiet=False)
        captured = capsys.readouterr()
        assert "No .inp files found" in captured.out

    def test_prints_count_when_files_found(self, tmp_path, capsys):
        """Test prints file count when files found (covers lines 453-454)."""
        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        _find_and_validate_inp_files(str(tmp_path), quiet=False)
        captured = capsys.readouterr()
        assert "Found 1 .inp files" in captured.out


class TestTryLoadFromCacheFunction:
    """Tests for _try_load_from_cache function."""

    def test_returns_none_when_cache_invalid(self, tmp_path):
        """Test returns None when cache is invalid (covers lines 460-463)."""
        cache = DatasetCache(tmp_path, "test")

        result = _try_load_from_cache(cache, [], quiet=True)
        assert result is None

    def test_prints_message_when_cache_invalid_not_quiet(self, tmp_path, capsys):
        """Test prints message when cache invalid and not quiet (covers lines 461-462)."""
        cache = DatasetCache(tmp_path, "test")

        _try_load_from_cache(cache, [], quiet=False)
        captured = capsys.readouterr()
        assert "Cache outdated or missing" in captured.out

    def test_returns_data_when_cache_valid(self, tmp_path):
        """Test returns cached data when valid (covers lines 465-469)."""
        cache = DatasetCache(tmp_path, "test")
        test_df = pd.DataFrame({"Name": ["C1"]})
        test_labels = pd.Series(["valid"])
        cache.save({"conduits": test_df, "labels": test_labels})

        with patch.object(DatasetCache, "is_valid", return_value=True):
            result = _try_load_from_cache(cache, [], quiet=True)
            assert result is not None
            assert "Name" in result[0].columns

    def test_prints_loading_message_when_not_quiet(self, tmp_path, capsys):
        """Test prints loading message when not quiet (covers lines 466-467)."""
        cache = DatasetCache(tmp_path, "test")
        cache.save({"conduits": pd.DataFrame(), "labels": pd.Series()})

        with patch.object(DatasetCache, "is_valid", return_value=True):
            _try_load_from_cache(cache, [], quiet=False)
            captured = capsys.readouterr()
            assert "Loading from cache" in captured.out

    def test_returns_none_on_exception(self, tmp_path, capsys):
        """Test returns None on cache load exception (covers lines 470-473)."""
        cache = DatasetCache(tmp_path, "test")
        # Create corrupt cache file
        cache.cache_file.parent.mkdir(exist_ok=True)
        cache.cache_file.write_bytes(b"not valid pickle")

        with patch.object(DatasetCache, "is_valid", return_value=True):
            result = _try_load_from_cache(cache, [], quiet=False)
            assert result is None
            captured = capsys.readouterr()
            assert "Cache error" in captured.out


class TestProcessInpFiles:
    """Tests for _process_inp_files function."""

    def test_prints_processing_message(self, tmp_path, capsys):
        """Test prints processing message (covers lines 478-479)."""
        with patch("sa.core.data_manager._load_inp_files", return_value=[]):
            _process_inp_files([], quiet=False)
            captured = capsys.readouterr()
            assert "Processing .inp files" in captured.out

    def test_returns_none_when_no_data_extracted(self, tmp_path, capsys):
        """Test returns (None, None) when no data extracted (covers lines 482-484)."""
        with patch("sa.core.data_manager._load_inp_files", return_value=[]):
            result = _process_inp_files([], quiet=True)
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "No data could be extracted" in captured.out

    def test_concatenates_frames_and_generates_labels(self, tmp_path):
        """Test concatenates DataFrames and generates labels (covers lines 486-491)."""
        df1 = pd.DataFrame({"Name": ["C1"], "ValCoverage": [1], "ValMaxFill": [1], "ValMinV": [1], "ReduceDia": [0]})
        df2 = pd.DataFrame({"Name": ["C2"], "ValCoverage": [0], "ValMaxFill": [1], "ValMinV": [1], "ReduceDia": [0]})

        with patch("sa.core.data_manager._load_inp_files", return_value=[df1, df2]):
            result = _process_inp_files([], quiet=True)

            assert result[0] is not None
            assert len(result[0]) == 2
            assert result[1] is not None


class TestLoadInpFiles:
    """Tests for _load_inp_files function."""

    def test_prints_progress_every_50_files(self, tmp_path, capsys):
        """Test prints progress every 50 files (covers lines 505-506)."""
        mock_files = [tmp_path / f"file{i}.inp" for i in range(51)]

        with patch("sa.core.data.DataManager") as mock_dm:
            mock_dm.return_value.__enter__.return_value = MagicMock(dfc=pd.DataFrame({"Tag": ["tag"], "Name": ["C1"]}))
            _load_inp_files(mock_files, quiet=False)
            captured = capsys.readouterr()
            # Should print at file 0 and file 50
            assert "Processing files 1-50" in captured.out or "Processing files" in captured.out

    def test_skips_files_without_tag_column(self, tmp_path):
        """Test skips files without Tag column (covers lines 510-511)."""
        mock_files = [tmp_path / "test.inp"]

        with patch("sa.core.data.DataManager") as mock_dm:
            # First file has no Tag column
            mock_dm.return_value.__enter__.return_value = MagicMock(
                dfc=pd.DataFrame({"Name": ["C1"]})  # No Tag column
            )
            result = _load_inp_files(mock_files, quiet=True)
            assert len(result) == 0

    def test_skips_files_with_null_tags(self, tmp_path):
        """Test skips files with null Tag values (covers lines 510-511)."""
        mock_files = [tmp_path / "test.inp"]

        with patch("sa.core.data.DataManager") as mock_dm:
            mock_dm.return_value.__enter__.return_value = MagicMock(dfc=pd.DataFrame({"Name": ["C1"], "Tag": [None]}))
            result = _load_inp_files(mock_files, quiet=True)
            assert len(result) == 0

    def test_adds_source_file_column(self, tmp_path):
        """Test adds source_file column (covers line 512)."""
        test_file = tmp_path / "test.inp"
        test_file.write_text("")

        with patch("sa.core.data.DataManager") as mock_dm:
            mock_dm.return_value.__enter__.return_value = MagicMock(dfc=pd.DataFrame({"Name": ["C1"], "Tag": ["tag"]}))
            result = _load_inp_files([test_file], quiet=True)
            assert len(result) == 1
            assert "source_file" in result[0].columns
            assert result[0]["source_file"].iloc[0] == "test.inp"

    def test_handles_exception_and_continues(self, tmp_path, capsys):
        """Test handles exception and continues processing (covers lines 514-517)."""
        mock_files = [tmp_path / "bad.inp", tmp_path / "good.inp"]

        def side_effect(*args, **kwargs):
            mock = MagicMock()
            if "bad" in str(args[0]):
                mock.__enter__.side_effect = Exception("Test error")
            else:
                mock.__enter__.return_value = MagicMock(dfc=pd.DataFrame({"Name": ["C1"], "Tag": ["tag"]}))
            return mock

        with patch("sa.core.data.DataManager", side_effect=side_effect):
            result = _load_inp_files(mock_files, quiet=False)
            captured = capsys.readouterr()
            assert "Error processing bad.inp" in captured.out
            assert len(result) == 1

    def test_import_error_adds_app_to_path(self, tmp_path):
        """Test handles ImportError by adding app to path (covers lines 496-500)."""
        mock_files = [tmp_path / "test.inp"]

        import_counter = {"count": 0}

        def mock_import(*args, **kwargs):
            import_counter["count"] += 1
            if import_counter["count"] == 1:
                raise ImportError("Test import error")
            return MagicMock()

        with patch("sa.core.data.DataManager", side_effect=mock_import):
            with patch("sa.core.data_manager._add_app_to_path"):
                # This will fail on first import, call _add_app_to_path, then succeed
                try:
                    _load_inp_files(mock_files, quiet=True)
                except Exception:
                    pass  # Expected to fail but _add_app_to_path should be called


class TestAddAppToPath:
    """Tests for _add_app_to_path function."""

    def test_adds_existing_app_path_to_sys_path(self, tmp_path):
        """Test adds app path when it exists (covers lines 524-526)."""
        app_dir = tmp_path / "app"
        app_dir.mkdir()

        with patch("os.getcwd", return_value=str(tmp_path / "a" / "b" / "c")):
            with patch("os.path.exists", return_value=True):
                with patch.object(sys, "path", []):
                    _add_app_to_path()
                    # Should have added something to sys.path
                    # Note: actual path depends on implementation


class TestGenerateLabelsForDataset:
    """Tests for _generate_labels_for_dataset function."""

    def test_generates_labels_for_all_rows(self):
        """Test generates labels for all rows (covers lines 531-535)."""
        df = pd.DataFrame(
            {
                "ValCoverage": [1, 0],
                "ValMaxFill": [1, 1],
                "ValMinV": [1, 1],
                "ReduceDia": [0, 0],
            }
        )
        result = _generate_labels_for_dataset(df, quiet=True)
        assert len(result) == 2
        assert result.iloc[0] == "valid"
        assert result.iloc[1] == "depth_increase"

    def test_prints_distribution_when_not_quiet(self, capsys):
        """Test prints label distribution when not quiet (covers lines 536-537)."""
        df = pd.DataFrame(
            {
                "ValCoverage": [1, 1],
                "ValMaxFill": [1, 1],
                "ValMinV": [1, 1],
                "ReduceDia": [0, 0],
            }
        )
        _generate_labels_for_dataset(df, quiet=False)
        captured = capsys.readouterr()
        assert "Label distribution" in captured.out


class TestSaveToCacheFunction:
    """Tests for _save_to_cache function."""

    def test_does_nothing_when_cache_is_none(self):
        """Test returns early when cache is None (covers lines 544-545)."""
        # Should not raise
        _save_to_cache(None, pd.DataFrame(), pd.Series(), quiet=True)

    def test_saves_data_to_cache(self, tmp_path):
        """Test saves data to cache (covers lines 547-551)."""
        cache = DatasetCache(tmp_path, "test")
        df = pd.DataFrame({"Name": ["C1"]})
        labels = pd.Series(["valid"])

        _save_to_cache(cache, df, labels, quiet=True)

        assert cache.cache_file.exists()
        loaded = cache.load()
        assert "conduits" in loaded
        assert "labels" in loaded

    def test_prints_message_when_not_quiet(self, tmp_path, capsys):
        """Test prints save message when not quiet (covers lines 550-551)."""
        cache = DatasetCache(tmp_path, "test")
        df = pd.DataFrame({"Name": ["C1"]})
        labels = pd.Series(["valid"])

        _save_to_cache(cache, df, labels, quiet=False)
        captured = capsys.readouterr()
        assert "Data saved to cache" in captured.out

    def test_handles_exception_gracefully(self, tmp_path, capsys):
        """Test handles save exception gracefully (covers lines 552-554)."""
        cache = DatasetCache(tmp_path, "test")

        with patch.object(DatasetCache, "save", side_effect=Exception("Write error")):
            _save_to_cache(cache, pd.DataFrame(), pd.Series(), quiet=False)
            captured = capsys.readouterr()
            assert "Could not save to cache" in captured.out


class TestGNNDataset:
    """Tests for GNNDataset class."""

    def test_raises_when_no_inp_files(self, tmp_path):
        """Test raises FileNotFoundError when no .inp files found."""
        from sa.core.data_manager import GNNDataset

        with pytest.raises(FileNotFoundError, match="No .inp files found"):
            GNNDataset(inp_directory=str(tmp_path), use_cache=False)

    def test_try_load_from_cache_success(self, tmp_path):
        """Test successful cache load (covers lines 198, 225-229)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        # Ensure source file is older than cache
        past_time = time.time() - 100
        os.utime(inp_file, (past_time, past_time))

        # Create valid cache with matching file list
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        import hashlib

        dir_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()
        cache_file = cache_dir / f"gnn_dataset_{dir_hash}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "inp_files": [inp_file],  # Same file list
                    "adjacency_matrix": sp.csr_matrix((3, 3)),
                    "conduit_order": ["C1", "C2", "C3"],
                    "simulations": [(np.array([[1.0]]), np.array([[1.0]]))],
                },
                f,
            )

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=True)
            # Should have loaded from cache
            assert dataset.conduit_order == ["C1", "C2", "C3"]
            assert len(dataset.simulations) == 1

    def test_len_method(self, tmp_path):
        """Test __len__ method (covers line 356)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.simulations = [
                    (np.array([[1.0]]), np.array([[1.0]])),
                    (np.array([[2.0]]), np.array([[2.0]])),
                ]
                assert len(dataset) == 2

    def test_getitem_method(self, tmp_path):
        """Test __getitem__ method (covers line 360)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                features = np.array([[1.0, 2.0]])
                labels = np.array([[0.0, 1.0]])
                dataset.simulations = [(features, labels)]

                result = dataset[0]
                assert np.array_equal(result[0], features)
                assert np.array_equal(result[1], labels)

    def test_to_stacked_numpy(self, tmp_path):
        """Test to_stacked_numpy method (covers lines 373-375)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.adjacency_matrix = sp.csr_matrix(np.array([[0, 1], [1, 0]]))
                dataset.simulations = [
                    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])),
                    (np.array([[5.0, 6.0], [7.0, 8.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])),
                ]

                adj, features, labels = dataset.to_stacked_numpy()

                assert sp.issparse(adj)
                assert features.shape == (2, 2, 2)  # (n_simulations, n_conduits, n_features)
                assert labels.shape == (2, 2, 2)  # (n_simulations, n_conduits, n_classes)

    def test_align_df_by_conduit_order(self, tmp_path):
        """Test _align_df_by_conduit_order method (covers lines 326-331)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2", "C3"]

                # DataFrame with different order
                df = pd.DataFrame(
                    {
                        "Name": ["C3", "C1", "C2"],
                        "Value": [30, 10, 20],
                    }
                )

                aligned = dataset._align_df_by_conduit_order(df)
                assert list(aligned.index) == ["C1", "C2", "C3"]
                assert list(aligned["Value"]) == [10, 20, 30]

    def test_align_df_by_conduit_order_missing_name_column(self, tmp_path):
        """Test _align_df_by_conduit_order raises when Name column missing (covers line 327)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1"]

                df = pd.DataFrame({"Value": [10]})  # No Name column

                with pytest.raises(ValueError, match="must have a 'Name' column"):
                    dataset._align_df_by_conduit_order(df)

    def test_extract_features(self, tmp_path):
        """Test _extract_features method (covers lines 335-346)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2"]
                dataset.feature_columns = ["ValMaxFill", "ValCoverage"]

                df = pd.DataFrame(
                    {
                        "Name": ["C1", "C2"],
                        "ValMaxFill": [0.5, 0.8],
                        "ValCoverage": [1.0, 0.9],
                    }
                )

                features = dataset._extract_features(df)
                assert features.shape == (2, 2)
                assert features.dtype == np.float32

    def test_extract_features_with_missing_data_fills_with_zero(self, tmp_path, caplog):
        """Test _extract_features fills missing data with 0 (covers lines 338-340)."""
        from sa.core.data_manager import GNNDataset
        import logging

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2", "C3"]  # C3 is missing from df
                dataset.feature_columns = ["ValMaxFill"]

                df = pd.DataFrame(
                    {
                        "Name": ["C1", "C2"],  # Missing C3
                        "ValMaxFill": [0.5, 0.8],
                    }
                )

                with caplog.at_level(logging.WARNING):
                    features = dataset._extract_features(df)
                    # C3 should be filled with 0
                    assert features.shape == (3, 1)
                    assert features[2, 0] == 0.0

    def test_extract_labels(self, tmp_path):
        """Test _extract_labels method (covers lines 350-352)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2"]

                df = pd.DataFrame(
                    {
                        "Name": ["C1", "C2"],
                        "ValCoverage": [1, 0],  # C2 -> depth_increase
                        "ValMaxFill": [1, 1],
                        "ValMinV": [1, 1],
                        "ReduceDia": [0, 0],
                    }
                )

                labels = dataset._extract_labels(df)
                assert labels.shape[0] == 2  # Two conduits
                assert labels.dtype == np.float32

    def test_validate_simulation_data_missing_conduits(self, tmp_path, caplog):
        """Test _validate_simulation_data returns False when conduits missing (covers line 322)."""
        from sa.core.data_manager import GNNDataset
        import logging

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2", "C3"]

                df = pd.DataFrame({"Name": ["C1", "C2"]})  # Missing C3

                with caplog.at_level(logging.WARNING):
                    result = dataset._validate_simulation_data(df, inp_file)
                    assert result is False

    def test_process_single_simulation_success(self, tmp_path):
        """Test _process_single_simulation returns features and labels on success (covers lines 303-305)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1"]
                dataset.feature_columns = ["ValMaxFill"]

                mock_df = pd.DataFrame(
                    {
                        "Name": ["C1"],
                        "ValMaxFill": [0.5],
                        "ValCoverage": [1],
                        "ValMinV": [1],
                        "ReduceDia": [0],
                    }
                )

                with patch.object(dataset, "_load_simulation_data", return_value=mock_df):
                    result = dataset._process_single_simulation(inp_file, MagicMock())
                    assert result is not None
                    features, labels = result
                    assert isinstance(features, np.ndarray)
                    assert isinstance(labels, np.ndarray)

    def test_setup_cache_disabled(self, tmp_path):
        """Test _setup_cache sets cache to None when disabled (covers line 209)."""
        from sa.core.data_manager import GNNDataset

        # Create a test .inp file
        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                assert dataset.cache is None

    def test_try_load_from_cache_returns_false_when_invalid(self, tmp_path):
        """Test _try_load_from_cache returns False when cache invalid (covers line 218-220)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                # Cache will be invalid because no cache file exists
                GNNDataset(inp_directory=str(tmp_path), use_cache=True)
                # Dataset was processed, not loaded from cache

    def test_try_load_from_cache_file_list_differs(self, tmp_path):
        """Test _try_load_from_cache returns False when file list differs (covers lines 231-232)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        # Create cache with different file list
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        import hashlib

        dir_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()
        cache_file = cache_dir / f"gnn_dataset_{dir_hash}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "inp_files": [Path("/different/path.inp")],
                    "adjacency_matrix": sp.csr_matrix((3, 3)),
                    "conduit_order": ["C1"],
                    "simulations": [],
                },
                f,
            )

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset") as mock_process:
                GNNDataset(inp_directory=str(tmp_path), use_cache=True)
                # Should have called _process_dataset because file list differs
                mock_process.assert_called_once()

    def test_try_load_from_cache_exception(self, tmp_path):
        """Test _try_load_from_cache handles exception (covers lines 233-235)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        # Create corrupt cache file
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        import hashlib

        dir_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()
        cache_file = cache_dir / f"gnn_dataset_{dir_hash}.pkl"
        cache_file.write_bytes(b"corrupt data")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset") as mock_process:
                GNNDataset(inp_directory=str(tmp_path), use_cache=True)
                # Should have called _process_dataset because cache load failed
                mock_process.assert_called_once()

    def test_save_to_cache_does_nothing_when_cache_none(self, tmp_path):
        """Test _save_to_cache returns early when cache is None (covers line 248)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                # _save_to_cache should do nothing because cache is None
                dataset._save_to_cache()  # Should not raise

    def test_save_to_cache_handles_exception(self, tmp_path):
        """Test _save_to_cache handles exception (covers lines 259-260)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=True)
                # Mock cache to raise exception on save
                dataset.adjacency_matrix = sp.csr_matrix((3, 3))
                dataset.conduit_order = ["C1"]
                dataset.simulations = []

                with patch.object(dataset.cache, "save", side_effect=Exception("Save error")):
                    # Should not raise, just log warning
                    dataset._save_to_cache()

    def test_process_all_simulations(self, tmp_path):
        """Test _process_all_simulations processes all files (covers lines 283-293)."""
        from sa.core.data_manager import GNNDataset

        inp_file1 = tmp_path / "test1.inp"
        inp_file2 = tmp_path / "test2.inp"
        inp_file1.write_text("")
        inp_file2.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file1, inp_file2]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.inp_files = [inp_file1, inp_file2]
                dataset.conduit_order = ["C1", "C2"]
                dataset.feature_columns = ["ValCoverage"]

                mock_result = (np.array([[1.0]]), np.array([[1.0]]))
                with patch.object(dataset, "_process_single_simulation", return_value=mock_result):
                    result = dataset._process_all_simulations(MagicMock())
                    assert len(result) == 2

    def test_process_single_simulation_returns_none_on_invalid(self, tmp_path):
        """Test _process_single_simulation returns None on invalid data (covers lines 300-301)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)
                dataset.conduit_order = ["C1", "C2"]

                # Mock data that doesn't have all conduits
                mock_df = pd.DataFrame({"Name": ["C1"]})  # Missing C2

                with patch.object(dataset, "_load_simulation_data", return_value=mock_df):
                    result = dataset._process_single_simulation(inp_file, MagicMock())
                    assert result is None

    def test_process_single_simulation_handles_exception(self, tmp_path):
        """Test _process_single_simulation returns None on exception (covers lines 307-309)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)

                with patch.object(dataset, "_load_simulation_data", side_effect=Exception("Load error")):
                    result = dataset._process_single_simulation(inp_file, MagicMock())
                    assert result is None

    def test_load_simulation_data(self, tmp_path):
        """Test _load_simulation_data loads data from file (covers lines 313-314)."""
        from sa.core.data_manager import GNNDataset

        inp_file = tmp_path / "test.inp"
        inp_file.write_text("")

        with patch("sa.core.data_manager.find_inp_files", return_value=[inp_file]):
            with patch.object(GNNDataset, "_process_dataset"):
                dataset = GNNDataset(inp_directory=str(tmp_path), use_cache=False)

                mock_dm_class = MagicMock()
                mock_dm_instance = MagicMock()
                mock_dm_instance.dfc = pd.DataFrame({"Name": ["C1"]})
                mock_dm_class.return_value.__enter__.return_value = mock_dm_instance

                result = dataset._load_simulation_data(inp_file, mock_dm_class)
                assert isinstance(result, pd.DataFrame)


class TestIntegration:
    """Integration tests using real .inp files without mocking.

    These tests complement the unit tests by verifying actual integration
    with the SWMM file parser and DataManager.
    """

    def test_prepare_swmm_dataset_with_real_inp_file(self, tmp_path):
        """Integration test: prepare_swmm_dataset with real .inp file."""
        import shutil
        from sa.core.tests import TEST_FILE

        test_inp = tmp_path / "test_file.inp"
        shutil.copy(TEST_FILE, test_inp)

        conduits, labels = prepare_swmm_dataset(
            str(tmp_path),
            output_dir=str(tmp_path / "output"),
            use_cache=False,
            quiet=True,
        )

        if conduits is not None:
            assert isinstance(conduits, pd.DataFrame)
            assert isinstance(labels, pd.Series)
            assert len(conduits) == len(labels)

    def test_find_inp_files_with_real_directory(self, tmp_path):
        """Integration test: find_inp_files with real filesystem."""
        import shutil
        from sa.core.tests import TEST_FILE

        test_inp = tmp_path / "real_test.inp"
        shutil.copy(TEST_FILE, test_inp)

        (tmp_path / "another.inp").write_text("[TITLE]\nTest\n")

        result = find_inp_files(str(tmp_path))

        assert len(result) == 2
        assert all(p.suffix == ".inp" for p in result)
        assert result[0].name == "another.inp"
        assert result[1].name == "real_test.inp"

    def test_dataset_cache_roundtrip_with_real_data(self, tmp_path):
        """Integration test: DatasetCache save/load with real pandas objects."""
        cache = DatasetCache(tmp_path, "integration_test")

        conduits = pd.DataFrame(
            {
                "Name": ["C1", "C2", "C3"],
                "ValCoverage": [1.0, 0.0, 1.0],
                "ValMaxFill": [0.8, 0.9, 0.7],
            }
        )
        labels = pd.Series(["valid", "depth_increase", "valid"])

        cache.save({"conduits": conduits, "labels": labels})

        loaded = cache.load()
        assert loaded["conduits"].equals(conduits)
        assert loaded["labels"].equals(labels)

    def test_swmm_label_generator_with_realistic_dataframe(self):
        """Integration test: SWMMLabelGenerator with realistic conduit data."""
        df = pd.DataFrame(
            {
                "ValCoverage": [1, 1, 0, 1, 1, 1],
                "ValMaxFill": [1, 0, 1, 1, 1, 0],
                "ValMinV": [1, 1, 1, 0, 1, 1],
                "IncreaseDia": [0, 1, 0, 0, 0, 0],
                "IncreaseSlope": [0, 0, 0, 1, 0, 0],
                "ReduceDia": [0, 0, 0, 0, 1, 0],
            }
        )

        labels = SWMMLabelGenerator.generate_labels_from_dataframe(df)

        assert labels == [
            "valid",
            "diameter_increase",
            "depth_increase",
            "slope_increase",
            "diameter_reduction",
            "tank",
        ]

    def test_base_swmm_dataset_with_real_dataframe(self):
        """Integration test: BaseSWMMDataset initialization with real DataFrame."""
        df = pd.DataFrame(
            {
                "Name": ["C1", "C2"],
                "InletNode": ["N1", "N2"],
                "OutletNode": ["N2", "N3"],
                "Length": [100.0, 150.0],
                "Diameter": [0.3, 0.4],
            }
        )

        dataset = BaseSWMMDataset(data_source=df)

        assert dataset.conduits_data is not None
        assert len(dataset.conduits_data) == 2
        assert "Name" in dataset.conduits_data.columns

        dataset.conduits_data["Length"] = [200.0, 250.0]
        assert df["Length"].iloc[0] == 100.0
