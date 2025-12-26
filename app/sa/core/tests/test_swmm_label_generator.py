import pytest
import pandas as pd
import numpy as np

from sa.core.data_manager import SWMMLabelGenerator, prepare_swmm_labels
from sa.core.enums import RecommendationCategory


class TestSWMMLabelGeneratorGenerateLabel:
    """Tests for SWMMLabelGenerator.generate_label()."""

    def test_depth_increase_when_val_coverage_zero(self):
        row = pd.Series({"ValCoverage": 0, "ValMaxFill": 1, "ValMinV": 1})
        assert SWMMLabelGenerator.generate_label(row) == "depth_increase"

    def test_diameter_increase_when_val_max_fill_zero_and_increase_dia(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 0, "IncreaseDia": 1, "ValMinV": 1})
        assert SWMMLabelGenerator.generate_label(row) == "diameter_increase"

    def test_tank_when_val_max_fill_zero_and_no_increase_dia(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 0, "IncreaseDia": 0, "ValMinV": 1})
        assert SWMMLabelGenerator.generate_label(row) == "tank"

    def test_slope_increase_when_val_min_v_zero_and_increase_slope(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 0, "IncreaseSlope": 1})
        assert SWMMLabelGenerator.generate_label(row) == "slope_increase"

    def test_seepage_boxes_when_val_min_v_zero_and_no_increase_slope(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 0, "IncreaseSlope": 0})
        assert SWMMLabelGenerator.generate_label(row) == "seepage_boxes"

    def test_diameter_reduction_when_reduce_dia_one(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 1, "ReduceDia": 1})
        assert SWMMLabelGenerator.generate_label(row) == "diameter_reduction"

    def test_valid_when_all_validations_pass(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1, "ValMinV": 1, "ReduceDia": 0})
        assert SWMMLabelGenerator.generate_label(row) == "valid"

    def test_valid_with_missing_optional_columns(self):
        row = pd.Series({"ValCoverage": 1})
        assert SWMMLabelGenerator.generate_label(row) == "valid"

    def test_priority_depth_increase_over_diameter_increase(self):
        row = pd.Series({"ValCoverage": 0, "ValMaxFill": 0, "IncreaseDia": 1})
        assert SWMMLabelGenerator.generate_label(row) == "depth_increase"

    def test_priority_diameter_increase_over_slope_increase(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 0, "IncreaseDia": 1, "ValMinV": 0, "IncreaseSlope": 1})
        assert SWMMLabelGenerator.generate_label(row) == "diameter_increase"


class TestSWMMLabelGeneratorValidateRowData:
    """Tests for SWMMLabelGenerator.validate_row_data()."""

    def test_valid_row_passes(self):
        row = pd.Series({"ValCoverage": 1, "ValMaxFill": 1}, name="conduit_1")
        SWMMLabelGenerator.validate_row_data(row)  # Should not raise

    def test_raises_when_row_all_null(self):
        row = pd.Series({"ValCoverage": np.nan, "ValMaxFill": np.nan}, name="conduit_1")
        with pytest.raises(ValueError, match="missing from simulation"):
            SWMMLabelGenerator.validate_row_data(row)

    def test_raises_when_val_coverage_missing(self):
        row = pd.Series({"ValMaxFill": 1, "ValCoverage": np.nan}, name="conduit_1")
        with pytest.raises(ValueError, match="missing critical validation data"):
            SWMMLabelGenerator.validate_row_data(row)


class TestSWMMLabelGeneratorGenerateLabelsFromDataframe:
    """Tests for SWMMLabelGenerator.generate_labels_from_dataframe()."""

    def test_generates_labels_for_all_rows(self):
        df = pd.DataFrame(
            {
                "ValCoverage": [1, 0, 1],
                "ValMaxFill": [1, 1, 0],
                "ValMinV": [1, 1, 1],
                "IncreaseDia": [0, 0, 1],
            }
        )
        labels = SWMMLabelGenerator.generate_labels_from_dataframe(df)
        assert labels == ["valid", "depth_increase", "diameter_increase"]

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=["ValCoverage", "ValMaxFill"])
        labels = SWMMLabelGenerator.generate_labels_from_dataframe(df)
        assert labels == []


class TestPrepareSwmmLabels:
    """Tests for prepare_swmm_labels()."""

    def test_one_hot_encoding_from_series(self):
        labels = pd.Series(["valid", "depth_increase", "tank"])
        result = prepare_swmm_labels(labels, one_hot=True)

        assert result.shape[0] == 3
        assert result.dtype == np.float32
        # Check one-hot encoding structure
        assert result.sum(axis=1).tolist() == [1.0, 1.0, 1.0]

    def test_index_encoding_from_series(self):
        labels = pd.Series(["valid", "depth_increase"])
        result = prepare_swmm_labels(labels, one_hot=False)

        assert len(result) == 2
        assert isinstance(result, np.ndarray)

    def test_unknown_category_maps_to_index_8(self):
        labels = pd.Series(["unknown_category"])
        result = prepare_swmm_labels(labels, one_hot=False)

        assert result[0] == 8

    def test_with_recommendation_category_enum(self):
        labels = pd.Series([RecommendationCategory.VALID, RecommendationCategory.TANK])
        result = prepare_swmm_labels(labels, one_hot=True)

        assert result.shape[0] == 2
        assert result.dtype == np.float32

    def test_with_dataframe_input(self):
        df = pd.DataFrame(
            {
                "valid": [1, 0],
                "tank": [0, 1],
            }
        )
        result = prepare_swmm_labels(df, one_hot=True)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, df.values.astype(np.float32))

    def test_one_hot_includes_all_categories(self):
        labels = pd.Series(["valid"])
        result = prepare_swmm_labels(labels, one_hot=True)

        all_classes = [cat.value for cat in RecommendationCategory]
        assert result.shape[1] == len(all_classes)
