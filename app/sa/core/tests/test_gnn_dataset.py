"""Tests for GNNDataset class in data_manager.py."""

import pytest
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from unittest.mock import patch

from sa.core.data_manager import GNNDataset, BaseSWMMDataset


@pytest.fixture
def sample_conduits_df():
    """Create sample conduits DataFrame for testing."""
    return pd.DataFrame(
        {
            "Name": ["C1", "C2", "C3"],
            "InletNode": ["N1", "N2", "N3"],
            "OutletNode": ["N2", "N3", "N4"],
            "ValCoverage": [1, 1, 0],
            "ValMaxFill": [1, 0, 1],
            "ValMinV": [1, 1, 1],
            "IncreaseDia": [0, 1, 0],
            "ReduceDia": [0, 0, 0],
            "IncreaseSlope": [0, 0, 0],
            "ReduceSlope": [0, 0, 0],
            "NRoughness": [0.5, 0.6, 0.7],
            "NMaxV": [1.0, 1.2, 1.1],
            "NFilling": [0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def mock_data_manager_class(sample_conduits_df):
    """Create mock DataManager class."""

    class MockDataManager:
        def __init__(self, path, zone=None):
            self.path = path
            self.zone = zone
            self.dfc = sample_conduits_df.copy()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return MockDataManager


@pytest.fixture
def mock_graph_constructor_class():
    """Create mock SWMMGraphConstructor class."""

    class MockSWMMGraphConstructor:
        def __init__(self, conduits_data):
            self.conduits_data = conduits_data
            self.idx_to_conduit = {0: "C1", 1: "C2", 2: "C3"}
            self.conduit_to_idx = {"C1": 0, "C2": 1, "C3": 2}

        def build_conduit_graph(self):
            # Return sparse adjacency matrix (C1 -> C2 -> C3)
            adj = sp.lil_matrix((3, 3), dtype=np.float32)
            adj[0, 1] = 1.0
            adj[1, 2] = 1.0
            return adj.tocsr(), None

    return MockSWMMGraphConstructor


@pytest.fixture
def temp_inp_directory(tmp_path):
    """Create temporary directory with mock .inp files."""
    inp_dir = tmp_path / "inp_files"
    inp_dir.mkdir()

    # Create dummy .inp files
    for i in range(3):
        (inp_dir / f"sim_{i}.inp").write_text(f"[TITLE]\nSimulation {i}")

    return inp_dir


class TestBaseSWMMDataset:
    """Tests for BaseSWMMDataset base class."""

    def test_init_with_directory(self, temp_inp_directory):
        dataset = BaseSWMMDataset(data_source=str(temp_inp_directory))

        assert dataset.inp_directory == str(temp_inp_directory)
        assert len(dataset.inp_files) == 3
        assert dataset.conduits_data is None

    def test_init_with_dataframe(self, sample_conduits_df):
        dataset = BaseSWMMDataset(data_source=sample_conduits_df)

        assert dataset.conduits_data is not None
        assert len(dataset.conduits_data) == 3
        assert dataset.inp_files == []

    def test_init_with_none_uses_default_directory(self):
        with patch("sa.core.data_manager.find_inp_files", return_value=[]):
            dataset = BaseSWMMDataset(data_source=None)
            assert dataset.inp_directory == BaseSWMMDataset.DEFAULT_INP_DIRECTORY

    def test_init_with_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="Unsupported data_source type"):
            BaseSWMMDataset(data_source=12345)

    def test_custom_feature_columns(self, sample_conduits_df):
        custom_cols = ["ValCoverage", "ValMaxFill"]
        dataset = BaseSWMMDataset(data_source=sample_conduits_df, feature_columns=custom_cols)

        assert dataset.feature_columns == custom_cols


class TestGNNDatasetInit:
    """Tests for GNNDataset initialization."""

    def test_raises_file_not_found_when_no_inp_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No .inp files found"):
            GNNDataset(inp_directory=str(empty_dir))

    def test_init_with_cache_disabled(self, temp_inp_directory, mock_data_manager_class, mock_graph_constructor_class):
        with patch("sa.core.data_manager.GNNDataset._process_dataset") as mock_process:
            # Make _process_dataset set required attributes
            def set_attrs(self_ref=None):
                pass

            mock_process.side_effect = set_attrs

            with patch.object(GNNDataset, "_process_dataset") as mock_proc:

                def init_attrs(obj):
                    obj.adjacency_matrix = sp.csr_matrix((3, 3))
                    obj.conduit_order = ["C1", "C2", "C3"]
                    obj.simulations = []

                mock_proc.side_effect = lambda: init_attrs(GNNDataset.__new__(GNNDataset))

                # Test that cache is None when disabled
                dataset = GNNDataset.__new__(GNNDataset)
                dataset.inp_directory = str(temp_inp_directory)
                dataset.inp_files = list(temp_inp_directory.glob("*.inp"))
                dataset.feature_columns = ["ValCoverage"]
                dataset._setup_cache(use_cache=False)

                assert dataset.cache is None


class TestGNNDatasetBuildBaseGraph:
    """Tests for GNNDataset._build_base_graph()."""

    def test_build_base_graph_returns_adjacency_and_order(
        self, temp_inp_directory, mock_data_manager_class, mock_graph_constructor_class
    ):
        # Create dataset instance without full init
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.inp_directory = str(temp_inp_directory)
        dataset.inp_files = list(temp_inp_directory.glob("*.inp"))
        dataset.feature_columns = ["ValCoverage", "NRoughness"]

        # Call _build_base_graph
        adj, order = dataset._build_base_graph(dataset.inp_files[0], mock_graph_constructor_class, mock_data_manager_class)

        assert isinstance(adj, sp.csr_matrix)
        assert adj.shape == (3, 3)
        assert order == ["C1", "C2", "C3"]

    def test_build_base_graph_propagates_exceptions(self, temp_inp_directory, mock_data_manager_class):
        class FailingConstructor:
            def __init__(self, data):
                raise RuntimeError("Graph construction failed")

        dataset = GNNDataset.__new__(GNNDataset)
        dataset.inp_directory = str(temp_inp_directory)
        dataset.inp_files = list(temp_inp_directory.glob("*.inp"))

        with pytest.raises(RuntimeError, match="Graph construction failed"):
            dataset._build_base_graph(dataset.inp_files[0], FailingConstructor, mock_data_manager_class)


class TestGNNDatasetAlignDfByConduitOrder:
    """Tests for GNNDataset._align_df_by_conduit_order()."""

    def test_aligns_dataframe_to_conduit_order(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C3", "C1", "C2"]  # Different order

        aligned = dataset._align_df_by_conduit_order(sample_conduits_df)

        assert list(aligned.index) == ["C3", "C1", "C2"]
        assert aligned.loc["C1", "NRoughness"] == 0.5
        assert aligned.loc["C3", "NRoughness"] == 0.7

    def test_raises_error_when_name_column_missing(self):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2"]

        df_without_name = pd.DataFrame({"OtherCol": [1, 2]})

        with pytest.raises(ValueError, match="must have a 'Name' column"):
            dataset._align_df_by_conduit_order(df_without_name)

    def test_handles_missing_conduits_with_nan(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3", "C4"]  # C4 doesn't exist

        aligned = dataset._align_df_by_conduit_order(sample_conduits_df)

        assert len(aligned) == 4
        assert pd.isna(aligned.loc["C4", "NRoughness"])


class TestGNNDatasetExtractFeatures:
    """Tests for GNNDataset._extract_features()."""

    def test_extracts_features_in_correct_order(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3"]
        dataset.feature_columns = ["NRoughness", "NMaxV"]

        features = dataset._extract_features(sample_conduits_df)

        assert features.shape == (3, 2)
        assert features.dtype == np.float32
        # C1 has NRoughness=0.5, NMaxV=1.0
        np.testing.assert_array_almost_equal(features[0], [0.5, 1.0])

    def test_fills_missing_features_with_zero(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3"]
        dataset.feature_columns = ["NRoughness", "NonExistentCol"]

        features = dataset._extract_features(sample_conduits_df)

        # Only NRoughness should be extracted (NonExistentCol is ignored)
        assert features.shape[1] == 1

    def test_handles_nan_values_in_features(self):
        df_with_nan = pd.DataFrame(
            {
                "Name": ["C1", "C2"],
                "NRoughness": [0.5, np.nan],
                "NMaxV": [np.nan, 1.0],
            }
        )

        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2"]
        dataset.feature_columns = ["NRoughness", "NMaxV"]

        features = dataset._extract_features(df_with_nan)

        # NaN should be filled with 0
        assert not np.isnan(features).any()
        np.testing.assert_array_almost_equal(features[0], [0.5, 0.0])
        np.testing.assert_array_almost_equal(features[1], [0.0, 1.0])


class TestGNNDatasetExtractLabels:
    """Tests for GNNDataset._extract_labels()."""

    def test_extracts_one_hot_labels(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3"]

        labels = dataset._extract_labels(sample_conduits_df)

        # Should be one-hot encoded with 9 classes (RecommendationCategory)
        assert labels.shape[0] == 3
        assert labels.dtype == np.float32
        # Each row should sum to 1 (one-hot)
        np.testing.assert_array_almost_equal(labels.sum(axis=1), [1.0, 1.0, 1.0])

    def test_labels_match_expected_categories(self, sample_conduits_df):
        """
        C1: ValCoverage=1, ValMaxFill=1, ValMinV=1 -> valid
        C2: ValCoverage=1, ValMaxFill=0, IncreaseDia=1 -> diameter_increase
        C3: ValCoverage=0 -> depth_increase
        """
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3"]

        labels = dataset._extract_labels(sample_conduits_df)

        # Get the index of max value for each row (predicted class)
        from sa.core.enums import RecommendationCategory

        all_classes = [cat.value for cat in RecommendationCategory]

        predicted_classes = [all_classes[np.argmax(labels[i])] for i in range(3)]

        assert predicted_classes[0] == "valid"
        assert predicted_classes[1] == "diameter_increase"
        assert predicted_classes[2] == "depth_increase"


class TestGNNDatasetValidateSimulationData:
    """Tests for GNNDataset._validate_simulation_data()."""

    def test_returns_true_when_all_conduits_present(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3"]

        result = dataset._validate_simulation_data(sample_conduits_df, Path("test.inp"))

        assert result is True

    def test_returns_false_when_conduits_missing(self, sample_conduits_df):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.conduit_order = ["C1", "C2", "C3", "C4"]  # C4 doesn't exist

        result = dataset._validate_simulation_data(sample_conduits_df, Path("test.inp"))

        assert result is False


class TestGNNDatasetToStackedNumpy:
    """Tests for GNNDataset.to_stacked_numpy()."""

    def test_returns_stacked_arrays(self):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.adjacency_matrix = sp.csr_matrix(np.eye(3))
        dataset.simulations = [
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 0], [0, 1], [1, 0]])),
            (np.array([[7, 8], [9, 10], [11, 12]]), np.array([[0, 1], [1, 0], [0, 1]])),
        ]

        adj, features, labels = dataset.to_stacked_numpy()

        assert isinstance(adj, sp.csr_matrix)
        assert features.shape == (2, 3, 2)  # 2 sims, 3 conduits, 2 features
        assert labels.shape == (2, 3, 2)  # 2 sims, 3 conduits, 2 classes


class TestGNNDatasetLenAndGetitem:
    """Tests for GNNDataset.__len__() and __getitem__()."""

    def test_len_returns_simulation_count(self):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.simulations = [(None, None), (None, None), (None, None)]

        assert len(dataset) == 3

    def test_getitem_returns_features_and_labels(self):
        features = np.array([[1, 2], [3, 4]])
        labels = np.array([[1, 0], [0, 1]])

        dataset = GNNDataset.__new__(GNNDataset)
        dataset.simulations = [(features, labels)]

        result_features, result_labels = dataset[0]

        np.testing.assert_array_equal(result_features, features)
        np.testing.assert_array_equal(result_labels, labels)


class TestGNNDatasetCache:
    """Tests for GNNDataset caching functionality."""

    def test_setup_cache_creates_cache_object(self, temp_inp_directory):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.inp_directory = str(temp_inp_directory)
        dataset.inp_files = list(temp_inp_directory.glob("*.inp"))

        dataset._setup_cache(use_cache=True)

        assert dataset.cache is not None
        assert "gnn_dataset_" in str(dataset.cache.cache_file)

    def test_try_load_from_cache_returns_false_when_no_cache(self, temp_inp_directory):
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.inp_directory = str(temp_inp_directory)
        dataset.inp_files = list(temp_inp_directory.glob("*.inp"))
        dataset._setup_cache(use_cache=True)

        result = dataset._try_load_from_cache()

        assert result is False

    def test_save_and_load_cache_roundtrip(self, temp_inp_directory):
        # Create dataset and save to cache
        dataset = GNNDataset.__new__(GNNDataset)
        dataset.inp_directory = str(temp_inp_directory)
        dataset.inp_files = list(temp_inp_directory.glob("*.inp"))
        dataset._setup_cache(use_cache=True)

        dataset.adjacency_matrix = sp.csr_matrix(np.eye(3))
        dataset.conduit_order = ["C1", "C2", "C3"]
        dataset.simulations = [(np.array([[1, 2]]), np.array([[1, 0]]))]

        dataset._save_to_cache()

        # Create new dataset and try to load from cache
        dataset2 = GNNDataset.__new__(GNNDataset)
        dataset2.inp_directory = str(temp_inp_directory)
        dataset2.inp_files = list(temp_inp_directory.glob("*.inp"))
        dataset2._setup_cache(use_cache=True)

        result = dataset2._try_load_from_cache()

        assert result is True
        assert dataset2.conduit_order == ["C1", "C2", "C3"]
        assert len(dataset2.simulations) == 1
