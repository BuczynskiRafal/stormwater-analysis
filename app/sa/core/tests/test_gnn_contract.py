"""Contract tests: training/serving single source of truth (T9) and the
GNN dataset-cache schema contract (T10, D9).

Import-root note (binding, see CONTRACT.md): pytest puts app/ on sys.path,
so app code imports as `sa.core...`. The training package
(models/recomendations/gnn/) is NOT importable as a normal package in this
env (its __init__.py eagerly imports training.py -> sklearn, which is not
installed), so its re-export modules are loaded directly by file path with
the repo root added to sys.path (needed for their own
`from app.sa.core... import ...` statements to resolve). This produces a
module tree parented under `app.*` that is a genuinely distinct object
graph from the `sa.core...` tree pytest already has loaded - so `is`
comparisons across them are False *by construction*, not a sign of
duplication. T9 therefore checks (1) numerical equivalence and (2) that the
training source files textually re-export from app rather than redefine.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_tf = pytest.mark.skipif(not HAS_TF, reason="requires tensorflow")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GNN_MODELS_PATH = os.path.join(REPO_ROOT, "models", "recomendations", "gnn", "gnn_models.py")
GRAPH_CONSTRUCTOR_PATH = os.path.join(REPO_ROOT, "models", "recomendations", "gnn", "graph_constructor.py")

# The training re-export modules live under the git-ignored models/ tree, so they
# are absent on a fresh CI checkout. T9 verifies the single-source-of-truth
# contract wherever that training code exists (local dev); it skips (not errors)
# when the files are absent.
TRAINING_MODULES_AVAILABLE = os.path.exists(GNN_MODELS_PATH) and os.path.exists(GRAPH_CONSTRUCTOR_PATH)
requires_training_modules = pytest.mark.skipif(
    not TRAINING_MODULES_AVAILABLE,
    reason="training re-export modules (models/recomendations/gnn/*.py) are git-ignored / absent",
)


def _load_module_by_path(name: str, path: str):
    """Load a module directly by file path, bypassing its package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_root_on_path():
    """T9 mechanics: add repo root to sys.path so the training module's
    `from app.sa.core...` absolute imports resolve. Removed again afterward
    to avoid leaking import-root state into other tests."""
    inserted = REPO_ROOT not in sys.path
    if inserted:
        sys.path.insert(0, REPO_ROOT)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(REPO_ROOT)
            except ValueError:
                pass


@requires_training_modules
class TestPreprocessAdjacencySingleSourceOfTruth:
    """T9 (preprocess_adjacency half)."""

    def test_source_reexports_rather_than_redefines(self):
        """Text-level contract guard: no TF needed. Confirms gnn_models.py
        imports preprocess_adjacency from app instead of defining its own
        copy (the K3/D1 duplication this test suite guards against)."""
        source = open(GNN_MODELS_PATH, encoding="utf-8").read()
        assert "from app.sa.core.gnn.preprocessing import preprocess_adjacency" in source
        assert "def preprocess_adjacency(" not in source

    @requires_tf
    def test_not_object_identical_by_construction(self, repo_root_on_path):
        """Documents the dual-import-root fact: this MUST be False here, and
        that is not a regression - see module docstring."""
        from sa.core.gnn import preprocess_adjacency as app_preprocess_adjacency

        training_module = _load_module_by_path("t9_training_gnn_models", GNN_MODELS_PATH)

        assert training_module.preprocess_adjacency is not app_preprocess_adjacency

    @requires_tf
    def test_numerically_equivalent_to_app_implementation(self, repo_root_on_path):
        from sa.core.gnn import preprocess_adjacency as app_preprocess_adjacency

        training_module = _load_module_by_path("t9_training_gnn_models_2", GNN_MODELS_PATH)

        rng = np.random.default_rng(0)
        dense = (rng.random((6, 6)) > 0.7).astype(np.float32)
        np.fill_diagonal(dense, 0)
        A = sp.csr_matrix(dense)

        training_result = training_module.preprocess_adjacency(A, max_hops=4).toarray()
        app_result = app_preprocess_adjacency(A, max_hops=4).toarray()

        assert np.allclose(training_result, app_result, atol=1e-9)


@requires_training_modules
class TestSWMMGraphConstructorSingleSourceOfTruth:
    """T9 (SWMMGraphConstructor half)."""

    def test_source_reexports_rather_than_redefines(self):
        source = open(GRAPH_CONSTRUCTOR_PATH, encoding="utf-8").read()
        assert "from app.sa.core.gnn.graph import SWMMGraphConstructor" in source
        assert "class SWMMGraphConstructor" not in source

    @requires_tf
    def test_not_object_identical_by_construction(self, repo_root_on_path):
        from sa.core.gnn import SWMMGraphConstructor as AppSWMMGraphConstructor

        training_module = _load_module_by_path("t9_training_graph_constructor", GRAPH_CONSTRUCTOR_PATH)

        assert training_module.SWMMGraphConstructor is not AppSWMMGraphConstructor

    @requires_tf
    def test_numerically_equivalent_to_app_implementation(self, repo_root_on_path):
        from sa.core.gnn import SWMMGraphConstructor as AppSWMMGraphConstructor

        training_module = _load_module_by_path("t9_training_graph_constructor_2", GRAPH_CONSTRUCTOR_PATH)

        dfc = pd.DataFrame(
            {
                "Name": ["C1", "C2", "C3"],
                "InletNode": ["A", "B", "C"],
                "OutletNode": ["B", "C", "D"],
            }
        )

        app_adj, _ = AppSWMMGraphConstructor(dfc).build_conduit_graph()
        training_adj, _ = training_module.SWMMGraphConstructor(dfc).build_conduit_graph()

        assert np.array_equal(training_adj.toarray(), app_adj.toarray())


class TestGnnCacheContract:
    """T10 (D9): cache schema-version / feature-column mismatches must be
    rejected (regenerate), never silently used; the adjacency written to the
    cache must be raw (zero diagonal). No TensorFlow needed - dict fixtures
    built in-memory, no dependency on the real on-disk pickle.
    """

    @pytest.fixture
    def raw_adjacency(self):
        return sp.csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32))

    @pytest.fixture
    def selfloop_adjacency(self):
        return sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32))

    @pytest.fixture
    def default_feature_columns(self):
        from sa.core.data_manager import get_default_feature_columns

        return get_default_feature_columns()

    def test_valid_cache_dict_is_accepted(self, raw_adjacency, default_feature_columns):
        from sa.core.data_manager import GNN_CACHE_SCHEMA_VERSION, _validate_gnn_cache_data

        cache = {
            "schema_version": GNN_CACHE_SCHEMA_VERSION,
            "feature_columns": default_feature_columns,
            "adjacency_matrix": raw_adjacency,
        }
        assert _validate_gnn_cache_data(cache) is True

    def test_wrong_schema_version_is_rejected(self, raw_adjacency, default_feature_columns):
        from sa.core.data_manager import GNN_CACHE_SCHEMA_VERSION, _validate_gnn_cache_data

        cache = {
            "schema_version": GNN_CACHE_SCHEMA_VERSION + 1,
            "feature_columns": default_feature_columns,
            "adjacency_matrix": raw_adjacency,
        }
        assert _validate_gnn_cache_data(cache) is False

    def test_mismatched_feature_columns_are_rejected(self, raw_adjacency, default_feature_columns):
        from sa.core.data_manager import GNN_CACHE_SCHEMA_VERSION, _validate_gnn_cache_data

        cache = {
            "schema_version": GNN_CACHE_SCHEMA_VERSION,
            "feature_columns": default_feature_columns[:-1] + ["some_other_column"],
            "adjacency_matrix": raw_adjacency,
        }
        assert _validate_gnn_cache_data(cache) is False

    def test_legacy_cache_without_schema_version_accepted_when_raw(self, raw_adjacency):
        from sa.core.data_manager import _validate_gnn_cache_data

        cache = {"adjacency_matrix": raw_adjacency}
        assert _validate_gnn_cache_data(cache) is True

    def test_legacy_cache_without_schema_version_rejected_when_not_raw(self, selfloop_adjacency):
        from sa.core.data_manager import _validate_gnn_cache_data

        cache = {"adjacency_matrix": selfloop_adjacency}
        assert _validate_gnn_cache_data(cache) is False

    def test_non_raw_adjacency_rejected_even_with_correct_schema(self, selfloop_adjacency, default_feature_columns):
        from sa.core.data_manager import GNN_CACHE_SCHEMA_VERSION, _validate_gnn_cache_data

        cache = {
            "schema_version": GNN_CACHE_SCHEMA_VERSION,
            "feature_columns": default_feature_columns,
            "adjacency_matrix": selfloop_adjacency,
        }
        assert _validate_gnn_cache_data(cache) is False

    def test_raw_adjacency_for_cache_zeroes_the_diagonal(self, selfloop_adjacency):
        from sa.core.data_manager import _raw_adjacency_for_cache

        result = _raw_adjacency_for_cache(selfloop_adjacency)

        assert np.allclose(result.diagonal(), 0)
        # Off-diagonal structure (the real edges) must be preserved.
        assert np.array_equal(result.toarray(), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32))

    def test_raw_adjacency_for_cache_accepts_dense_input(self):
        from sa.core.data_manager import _raw_adjacency_for_cache

        dense = np.array([[1, 1], [0, 1]], dtype=np.float32)
        result = _raw_adjacency_for_cache(dense)

        assert sp.issparse(result)
        assert np.allclose(result.diagonal(), 0)
