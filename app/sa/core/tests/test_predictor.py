"""Tests for predictor module loading behavior.

Note: The predictor module loads models at import time, making it difficult
to test failure paths directly. These tests verify the module exports.
"""

import pytest


class TestPredictorModuleExports:
    """Tests for predictor module exports."""

    def test_classifier_is_exported(self):
        from sa.core.predictor import classifier
        assert classifier is not None

    def test_recommendation_is_exported(self):
        from sa.core.predictor import recommendation
        assert recommendation is not None

    def test_gnn_recommendation_is_exported(self):
        from sa.core.predictor import gnn_recommendation
        # gnn_recommendation can be None if GNN loading failed
        from sa.core import predictor
        assert hasattr(predictor, "gnn_recommendation")

    def test_all_exports_defined(self):
        from sa.core.predictor import __all__
        assert "classifier" in __all__
        assert "recommendation" in __all__
        assert "gnn_recommendation" in __all__
