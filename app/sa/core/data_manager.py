"""
Data management utilities for SWMM conduit networks.
Handles graph construction, feature preparation, and dataset creation.
Prepare real SWMM data for GNN training.
This script loads your SWMM .inp files and prepares them for the GNN training pipeline.

Requirements: pandas, numpy, scipy, openpyxl (for Excel file support)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import scipy.sparse as sp
from collections import defaultdict
from typing import List, Tuple, Optional
from .enums import RecommendationCategory
from scipy.spatial.distance import cdist
import pickle
import hashlib

logger = logging.getLogger(__name__)


class SWMMLabelGenerator:
    """Generates SWMM recommendation labels from conduit data."""
    
    @staticmethod
    def generate_label(row: pd.Series) -> str:
        """Generate a single label from conduit row data."""
        if row.get('ValCoverage', 1) == 0:
            return 'depth_increase'
        elif row.get('ValMaxFill', 1) == 0:
            return 'diameter_increase' if row.get('IncreaseDia', 0) == 1 else 'tank'
        elif row.get('ValMinV', 1) == 0:
            return 'slope_increase' if row.get('IncreaseSlope', 0) == 1 else 'seepage_boxes'
        elif row.get('ReduceDia', 0) == 1:
            return 'diameter_reduction'
        else:
            return 'valid'
    
    @staticmethod
    def validate_row_data(row: pd.Series):
        """Validate that row has required data for label generation."""
        if row.isnull().all():
            raise ValueError(
                f"Conduit '{row.name}' is present in base topology but missing from simulation file"
            )
        
        if pd.isna(row.get('ValCoverage')):
            raise ValueError(
                f"Conduit '{row.name}' missing critical validation data ('ValCoverage')"
            )
    
    @classmethod
    def generate_labels_from_dataframe(cls, df: pd.DataFrame) -> List[str]:
        """Generate labels for all rows in dataframe."""
        labels = []
        for _, row in df.iterrows():
            cls.validate_row_data(row)
            labels.append(cls.generate_label(row))
        return labels


class DatasetCache:
    """Handles caching logic for SWMM datasets."""
    
    def __init__(self, cache_dir: Path, cache_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{cache_name}.pkl"
    
    def is_valid(self, source_files: List[Path]) -> bool:
        """Check if cache exists and is newer than source files."""
        if not self.cache_file.exists():
            return False
            
        try:
            cache_time = self.cache_file.stat().st_mtime
            newest_source_time = max(f.stat().st_mtime for f in source_files)
            return cache_time > newest_source_time
        except Exception:
            return False
    
    def load(self) -> dict:
        """Load data from cache."""
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)
    
    def save(self, data: dict):
        """Save data to cache."""
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)


def get_default_feature_columns():
    """Get default feature columns for SWMM conduits."""
    return [
        "ValMaxFill", "ValMaxV", "ValMinV", "ValMaxSlope", "ValMinSlope",
        "ValDepth", "ValCoverage", "isMinDiameter", "IncreaseDia", "ReduceDia",
        "IncreaseSlope", "ReduceSlope", "NRoughness", "NMaxV", "NInletDepth", 
        "NOutletDepth", "NFilling", "NMaxQ", "NInletGroundCover", "NOutletGroundCover", 
        "NSlope", "marshes", "suburban_highly_impervious", "suburban_weakly_impervious",
        "arable", "meadows", "forests", "rural", "urban_weakly_impervious",
        "urban_moderately_impervious", "urban_highly_impervious",
        "mountains_rocky", "mountains_vegetated"
    ]

class BaseSWMMDataset:
    """
    Base class for SWMM datasets with common data handling functionality.
    
    Handles initialization from various data sources (directories, DataFrames)
    and provides standard feature column management.
    """
    DEFAULT_INP_DIRECTORY = "models/recomendations/dataset/proba/generated"

    def __init__(self, data_source: object = None, feature_columns: list[str] = None):
        """
        Initializes the dataset from a data source.

        Args:
            data_source (Union[str, pd.DataFrame, None]): Path to a directory of .inp files or a pandas DataFrame.
                                                          If None, uses the default directory.
            feature_columns (List[str], optional): List of columns to use as features. 
                                                   Defaults to get_default_feature_columns().
        """
        if data_source is None:
            data_source = self.DEFAULT_INP_DIRECTORY

        self.feature_columns = feature_columns if feature_columns is not None else get_default_feature_columns()
        self.inp_files = []
        self.conduits_data = None

        if isinstance(data_source, str):
            self.inp_directory = data_source
            self.inp_files = find_inp_files(self.inp_directory)
            if not self.inp_files:
                logger.warning(f"No .inp files found in the specified directory: {self.inp_directory}")
        elif isinstance(data_source, pd.DataFrame):
            self.conduits_data = data_source.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported data_source type: {type(data_source)}")



class GNNDataset(BaseSWMMDataset):
    """
    Prepares SWMM data for GNN training based on a "one graph, many features" approach.

    This approach assumes:
    - The graph topology (connections between conduits) is constant across all simulations.
    - The features of the conduits (nodes in our graph) change with each simulation.

    The class builds the graph from the first .inp file found in the directory and then
    extracts feature and label sets for each .inp file, ensuring they align with the
    base graph's topology.
    """

    def __init__(self, inp_directory: str = None, feature_columns: List[str] = None, use_cache=True):
        """
        Args:
            inp_directory (str): Path to the directory containing .inp files.
            feature_columns (List[str], optional): List of columns to use as features.
            use_cache (bool): Whether to use a cached version of the dataset if available.
        """
        super().__init__(data_source=inp_directory, feature_columns=feature_columns)
        logger.info(f"Initializing GNN Dataset from directory: {self.inp_directory}")

        if not self.inp_files:
            raise FileNotFoundError(f"No .inp files found in {self.inp_directory}")

        self._setup_cache(use_cache)
        
        if use_cache and self._try_load_from_cache():
            return
            
        logger.info(f"Found {len(self.inp_files)} files for GNN dataset. Processing from scratch...")
        self._process_dataset()
        
        if use_cache:
            self._save_to_cache()
    
    def _setup_cache(self, use_cache: bool):
        """Setup caching system."""
        if not use_cache:
            self.cache = None
            return
            
        dir_hash = hashlib.md5(str(self.inp_directory).encode()).hexdigest()
        cache_dir = Path(self.inp_directory) / '.cache'
        self.cache = DatasetCache(cache_dir, f'gnn_dataset_{dir_hash}')
    
    def _try_load_from_cache(self) -> bool:
        """Try to load dataset from cache. Returns True if successful."""
        if not self.cache or not self.cache.is_valid(self.inp_files):
            logger.info("Cache invalid or missing, processing from scratch...")
            return False
            
        try:
            cached_data = self.cache.load()
            if [str(p) for p in cached_data.get('inp_files', [])] == [str(p) for p in self.inp_files]:
                logger.info("✅ Loading dataset from cache...")
                self.adjacency_matrix = cached_data['adjacency_matrix']
                self.conduit_order = cached_data['conduit_order']
                self.simulations = cached_data['simulations']
                return True
            else:
                logger.info("⚠️ Cache file list differs, re-processing...")
                return False
        except Exception as e:
            logger.warning(f"Cache load failed: {e}, re-processing...")
            return False
    
    def _process_dataset(self):
        """Process dataset from scratch."""
        from graph_constructor import SWMMGraphConstructor
        from sa.core.data import DataManager

        self.adjacency_matrix, self.conduit_order = self._build_base_graph(
            self.inp_files[0], SWMMGraphConstructor, DataManager
        )
        self.simulations = self._process_all_simulations(DataManager)
    
    def _save_to_cache(self):
        """Save processed dataset to cache."""
        if not self.cache:
            return
            
        try:
            data_to_cache = {
                'adjacency_matrix': self.adjacency_matrix,
                'conduit_order': self.conduit_order,
                'simulations': self.simulations,
                'inp_files': self.inp_files
            }
            self.cache.save(data_to_cache)
            logger.info(f"✅ Dataset saved to cache: {self.cache.cache_file}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _build_base_graph(self, first_file_path: Path, constructor_cls, data_manager_cls) -> Tuple[sp.csr_matrix, List[str]]:
        """Builds the base graph topology and gets the canonical conduit order from the first file."""
        logger.info(f"Building base graph topology from: {first_file_path.name}")
        try:
            with data_manager_cls(str(first_file_path), zone=1.6) as model:
                first_file_conduits = model.dfc.copy().sort_values(by="Name").reset_index()

            constructor = constructor_cls(first_file_conduits)
            adjacency_matrix, _ = constructor.build_conduit_graph()
            
            # The canonical order of conduits is determined by the constructor's mapping
            conduit_order = [name for idx, name in sorted(constructor.idx_to_conduit.items())]

            logger.info(f"Base graph created with {adjacency_matrix.shape[0]} nodes and {adjacency_matrix.nnz} edges.")
            return adjacency_matrix, conduit_order
        except Exception as e:
            logger.error(f"Failed to build base graph from {first_file_path.name}: {e}")
            raise

    def _process_all_simulations(self, data_manager_cls) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Processes each .inp file to extract features and labels aligned with the base graph."""
        simulations_data = []

        for i, inp_file in enumerate(self.inp_files):
            logger.info(f"Processing simulation {i+1}/{len(self.inp_files)}: {inp_file.name}")
            
            simulation_data = self._process_single_simulation(inp_file, data_manager_cls)
            if simulation_data is not None:
                simulations_data.append(simulation_data)

        logger.info(f"Successfully processed {len(simulations_data)} simulations.")
        return simulations_data
    
    def _process_single_simulation(self, inp_file: Path, data_manager_cls) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Process a single simulation file."""
        try:
            conduits_df = self._load_simulation_data(inp_file, data_manager_cls)
            
            if not self._validate_simulation_data(conduits_df, inp_file):
                return None
                
            features = self._extract_features(conduits_df)
            labels = self._extract_labels(conduits_df)
            return (features, labels)
            
        except Exception as e:
            logger.warning(f"Could not process file {inp_file.name}. Error: {e}")
            return None
    
    def _load_simulation_data(self, inp_file: Path, data_manager_cls) -> pd.DataFrame:
        """Load conduits data from a simulation file."""
        with data_manager_cls(str(inp_file), zone=1.6) as model:
            return model.dfc.copy()
    
    def _validate_simulation_data(self, conduits_df: pd.DataFrame, inp_file: Path) -> bool:
        """Validate that simulation data can be aligned with base graph."""
        missing_conduits = set(self.conduit_order) - set(conduits_df['Name'])
        if missing_conduits:
            logger.warning(f"Skipping {inp_file.name}: missing conduits from base topology")
            return False
        return True

    def _align_df_by_conduit_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns a DataFrame's rows to match the canonical conduit order."""
        if "Name" not in df.columns:
            raise ValueError("DataFrame must have a 'Name' column for alignment.")
        
        df_indexed = df.set_index("Name")
        aligned_df = df_indexed.reindex(self.conduit_order)
        return aligned_df

    def _extract_features(self, conduits_df: pd.DataFrame) -> np.ndarray:
        """Extracts and prepares a feature matrix from conduit data, ensuring consistent order."""
        aligned_df = self._align_df_by_conduit_order(conduits_df)
        
        # Check for missing conduits after reindexing, which indicates data inconsistency
        if aligned_df.isnull().values.any():
            logger.warning("Found missing conduit data when aligning features. Filling with 0.")
            aligned_df = aligned_df.fillna(0.0)
            
        available_features = [col for col in self.feature_columns if col in aligned_df.columns]
        features_df = aligned_df[available_features]

        conduit_features = (
            features_df.apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values.astype(np.float32)
        )
        return conduit_features

    def _extract_labels(self, conduits_df: pd.DataFrame) -> np.ndarray:
        """Generates labels for each conduit, ensuring consistent order."""
        aligned_df = self._align_df_by_conduit_order(conduits_df)
        labels = SWMMLabelGenerator.generate_labels_from_dataframe(aligned_df)
        return prepare_swmm_labels(pd.Series(labels), one_hot=True)

    def __len__(self):
        """Returns the number of simulations (i.e., .inp files processed)."""
        return len(self.simulations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the features and labels for a given simulation index."""
        return self.simulations[idx]

    def to_stacked_numpy(self) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Returns the data as a tuple of stacked NumPy arrays.
        This is useful for training loops that expect all data in memory.
        
        Returns:
            Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
                - adjacency_matrix: The single graph topology.
                - all_features: Stacked features (n_simulations, n_conduits, n_features).
                - all_labels: Stacked labels (n_simulations, n_conduits, n_classes).
        """
        all_features = np.stack([features for features, labels in self.simulations])
        all_labels = np.stack([labels for features, labels in self.simulations])
        return self.adjacency_matrix, all_features, all_labels


def prepare_swmm_labels(labels, one_hot=True):
    """Prepare SWMM recommendation labels."""
    all_classes = [cat.value for cat in RecommendationCategory]

    if isinstance(labels, pd.Series) and isinstance(labels.iloc[0], RecommendationCategory):
        labels = labels.apply(lambda x: x.value)

    if isinstance(labels, pd.Series):
        if one_hot:
            y_onehot = pd.get_dummies(labels)
            y_onehot = y_onehot.reindex(columns=all_classes, fill_value=0)
            return y_onehot.values.astype(np.float32)
        else:
            label_to_idx = {label: idx for idx, label in enumerate(all_classes)}
            return np.array([label_to_idx.get(label, 8) for label in labels])
    elif isinstance(labels, pd.DataFrame):
        return labels.values.astype(np.float32)
    else:
        return np.array([l.value if isinstance(l, RecommendationCategory) else l for l in labels]).astype(np.float32)


def find_inp_files(directory):
    """
    Find all .inp files in the specified directory.
    
    Args:
        directory (str): Path to the directory to search
        
    Returns:
        list: List of paths to .inp files
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        return []
    
    return sorted(directory_path.glob("*.inp"))


def prepare_swmm_dataset(inp_directory, output_dir="real_data", use_cache=True, quiet=False):
    """Prepare SWMM dataset from .inp files for GNN training with pickle caching."""
    files = _find_and_validate_inp_files(inp_directory, quiet)
    if not files:
        return None, None
    
    if use_cache and output_dir:
        cache = DatasetCache(Path(output_dir), "prepared_dataset")
        cached_data = _try_load_from_cache(cache, files, quiet)
        if cached_data:
            return cached_data
    else:
        cache = None
    
    # Process from scratch
    conduits_data, labels = _process_inp_files(files, quiet)
    if conduits_data is None:
        return None, None
    
    _save_to_cache(cache, conduits_data, labels, quiet)
    return conduits_data, labels


def _find_and_validate_inp_files(inp_directory: str, quiet: bool) -> List[Path]:
    """Find and validate .inp files in directory."""
    if not quiet:
        print(f"🔍 Searching for .inp files in: {inp_directory}")
    
    files = find_inp_files(inp_directory)
    if not files:
        if not quiet:
            print("❌ No .inp files found!")
        return []
    
    if not quiet:
        print(f"✅ Found {len(files)} .inp files")
    return files


def _try_load_from_cache(cache: DatasetCache, files: List[Path], quiet: bool) -> Optional[Tuple]:
    """Try to load dataset from cache."""
    if not cache.is_valid(files):
        if not quiet:
            print("⚠️ Cache outdated or missing, processing...")
        return None
    
    try:
        if not quiet:
            print(f"✅ Loading from cache: {cache.cache_file}")
        data = cache.load()
        return data['conduits'], data['labels']
    except Exception as e:
        if not quiet:
            print(f"⚠️ Cache error: {e}, processing...")
        return None

def _process_inp_files(files: List[Path], quiet: bool) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Process .inp files and extract conduits data and labels."""
    if not quiet:
        print("📂 Processing .inp files...")
    
    frames = _load_inp_files(files, quiet)
    if not frames:
        print("❌ No data could be extracted from .inp files!")
        return None, None

    conduits_data = pd.concat(frames, ignore_index=True)
    if not quiet:
        print(f"📊 Combined dataset shape: {conduits_data.shape}")

    labels = _generate_labels_for_dataset(conduits_data, quiet)
    return conduits_data, labels


def _load_inp_files(files: List[Path], quiet: bool) -> List[pd.DataFrame]:
    """Load and process individual .inp files."""
    try:
        from sa.core.data import DataManager
    except ImportError:
        _add_app_to_path()
        from sa.core.data import DataManager

    frames = []
    for i, inp_path in enumerate(files):
        try:
            if not quiet and (i % 50 == 0 or i == len(files) - 1):
                print(f"📂 Processing files {i+1}-{min(i+50, len(files))}/{len(files)}...")
                
            with DataManager(str(inp_path), zone=1.6) as model:
                df = model.dfc.copy()
                if 'Tag' not in df.columns or df['Tag'].isnull().any():
                    continue
                df['source_file'] = inp_path.name
                frames.append(df)
        except Exception as e:
            if not quiet:
                print(f"\n⚠️ Error processing {inp_path.name}: {e}")
            continue
    
    return frames


def _add_app_to_path():
    """Add app directory to Python path for DataManager import."""
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'app')
    if os.path.exists(app_path):
        sys.path.insert(0, app_path)


def _generate_labels_for_dataset(conduits_data: pd.DataFrame, quiet: bool) -> pd.Series:
    """Generate labels for the entire dataset."""
    labels = []
    for _, row in conduits_data.iterrows():
        labels.append(SWMMLabelGenerator.generate_label(row))
    
    labels_series = pd.Series(labels)
    if not quiet:
        print(f"📈 Label distribution: {labels_series.value_counts().to_dict()}")
    
    return labels_series


def _save_to_cache(cache: Optional[DatasetCache], conduits_data: pd.DataFrame, labels: pd.Series, quiet: bool):
    """Save processed data to cache."""
    if not cache:
        return
        
    try:
        data_to_cache = {'conduits': conduits_data, 'labels': labels}
        cache.save(data_to_cache)
        if not quiet:
            print(f"💾 Data saved to cache: {cache.cache_file}")
    except Exception as e:
        if not quiet:
            print(f"Could not save to cache: {e}")
