import unittest
import pandas as pd
from sa.core.data import ConduitFeatureEngineeringService


class TestHydraulicFlowNormalization(unittest.TestCase):
    def setUp(self):
        self.dfc = pd.DataFrame(
            {
                "Geom1": [0.3, 0.5, 0.8, 1.0, 1.2, 0.0, None],
                "SlopePerMile": [5.0, 10.0, 15.0, 20.0, 25.0, 5.0, 10.0],
                "MaxQ": [0.1, 0.5, 1.5, 3.0, 5.0, 0.1, 0.5],
            }
        )

        self.dfn = None

        self.service = ConduitFeatureEngineeringService(self.dfc, self.dfn, 1.0)

    def test_normalize_max_q_normal_values(self):
        """Test normalization with normal values within expected ranges."""
        self.service.normalize_max_q()

        self.assertIn("NMaxQ", self.service.dfc.columns)

        self.assertTrue(all(0 <= val <= 1 for val in self.service.dfc["NMaxQ"].dropna()))

        k = 0.312

        for i in range(3):
            diameter = self.dfc.loc[i, "Geom1"]
            slope = self.dfc.loc[i, "SlopePerMile"]
            max_q = self.dfc.loc[i, "MaxQ"]
            theoretical_max = k * (diameter**2.5) * (slope**0.5)
            expected = min(max_q / theoretical_max, 1.0)

            self.assertAlmostEqual(self.service.dfc.loc[i, "NMaxQ"], expected, places=6, msg=f"Row {i} normalization incorrect")

    def test_normalize_max_q_zero_diameter(self):
        """Test with zero diameter which should result in NaN or inf values that get clipped."""
        self.service.normalize_max_q()

        row_idx = 5

        result = self.service.dfc.loc[row_idx, "NMaxQ"]
        self.assertTrue(pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for zero diameter, got {result}")

    def test_normalize_max_q_none_diameter(self):
        """Test with None/NaN diameter which should result in NaN values."""
        self.service.normalize_max_q()

        row_idx = 6
        self.assertTrue(
            pd.isna(self.service.dfc.loc[row_idx, "NMaxQ"]),
            f"Expected NaN for None diameter, got {self.service.dfc.loc[row_idx, 'NMaxQ']}",
        )

    def test_normalize_max_q_zero_slope(self):
        """Test with zero slope which should result in division by zero."""

        row_idx = 7
        self.service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 0.0, "MaxQ": 0.5}

        self.service.normalize_max_q()

        result = self.service.dfc.loc[row_idx, "NMaxQ"]
        self.assertTrue(pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for zero slope, got {result}")

    def test_normalize_max_q_negative_slope(self):
        """Test with negative slope which is physically impossible."""

        row_idx = 8
        self.service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": -5.0, "MaxQ": 0.5}

        self.service.normalize_max_q()

        result = self.service.dfc.loc[row_idx, "NMaxQ"]
        self.assertTrue(pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for negative slope, got {result}")

    def test_normalize_max_q_zero_flow(self):
        """Test with zero flow which should result in zero normalized flow."""

        row_idx = 9
        self.service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": 0.0}

        self.service.normalize_max_q()

        self.assertEqual(
            self.service.dfc.loc[row_idx, "NMaxQ"],
            0.0,
            f"Expected 0.0 for zero flow, got {self.service.dfc.loc[row_idx, 'NMaxQ']}",
        )

    def test_normalize_max_q_negative_flow(self):
        """Test with negative flow which is physically impossible."""

        row_idx = 10
        self.service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": -0.5}

        self.service.normalize_max_q()

        self.assertEqual(
            self.service.dfc.loc[row_idx, "NMaxQ"],
            0.0,
            f"Expected 0.0 for negative flow, got {self.service.dfc.loc[row_idx, 'NMaxQ']}",
        )

    def test_normalize_max_q_very_large_flow(self):
        """Test with very large flow that exceeds theoretical maximum."""

        row_idx = 11
        self.service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": 1000.0}

        self.service.normalize_max_q()

        self.assertEqual(
            self.service.dfc.loc[row_idx, "NMaxQ"],
            1.0,
            f"Expected 1.0 for very large flow, got {self.service.dfc.loc[row_idx, 'NMaxQ']}",
        )

    def test_normalize_max_q_very_large_diameter(self):
        """Test with very large diameter."""

        row_idx = 12
        self.service.dfc.loc[row_idx] = {"Geom1": 10.0, "SlopePerMile": 10.0, "MaxQ": 5.0}

        self.service.normalize_max_q()

        result = self.service.dfc.loc[row_idx, "NMaxQ"]
        self.assertTrue(0 <= result < 0.1, f"Expected small value (<0.1) for large diameter, got {result}")

    def test_normalize_max_q_very_small_diameter(self):
        """Test with very small but non-zero diameter."""

        row_idx = 13
        self.service.dfc.loc[row_idx] = {"Geom1": 0.01, "SlopePerMile": 10.0, "MaxQ": 0.001}

        self.service.normalize_max_q()

        result = self.service.dfc.loc[row_idx, "NMaxQ"]
        self.assertTrue(result > 0.5, f"Expected large value (>0.5) for small diameter, got {result}")

    def test_normalize_max_q_none_dataframe(self):
        """Test with None dataframe which should return without error."""
        service = ConduitFeatureEngineeringService(None, None, 1.0)

        try:
            service.normalize_max_q()

            success = True
        except Exception as e:
            success = False
            self.fail(f"normalize_max_q raised exception with None dataframe: {e}")

        self.assertTrue(success)
        self.assertIsNone(service.dfc)

    def test_normalize_max_q_missing_columns(self):
        """Test with dataframe missing required columns."""

        dfc_missing_slope = pd.DataFrame({"Geom1": [0.3, 0.5], "MaxQ": [0.1, 0.5]})
        service = ConduitFeatureEngineeringService(dfc_missing_slope, None, 1.0)

        with self.assertRaises(KeyError) as context:
            service.normalize_max_q()
        self.assertIn("SlopePerMile", str(context.exception), "Expected KeyError for missing SlopePerMile column")

        dfc_missing_geom = pd.DataFrame({"SlopePerMile": [5.0, 10.0], "MaxQ": [0.1, 0.5]})
        service = ConduitFeatureEngineeringService(dfc_missing_geom, None, 1.0)

        with self.assertRaises(KeyError) as context:
            service.normalize_max_q()
        self.assertIn("Geom1", str(context.exception), "Expected KeyError for missing Geom1 column")

        dfc_missing_maxq = pd.DataFrame({"Geom1": [0.3, 0.5], "SlopePerMile": [5.0, 10.0]})
        service = ConduitFeatureEngineeringService(dfc_missing_maxq, None, 1.0)

        with self.assertRaises(KeyError) as context:
            service.normalize_max_q()
        self.assertIn("MaxQ", str(context.exception), "Expected KeyError for missing MaxQ column")

    def test_normalize_max_q_formula_correctness(self):
        """Test that the normalization formula is correctly implemented."""

        test_df = pd.DataFrame({"Geom1": [0.4], "SlopePerMile": [8.0], "MaxQ": [0.2]})
        service = ConduitFeatureEngineeringService(test_df, None, 1.0)

        service.normalize_max_q()

        k = 0.312
        diameter = 0.4
        slope = 8.0
        max_q = 0.2
        theoretical_max = k * (diameter**2.5) * (slope**0.5)
        expected = min(max_q / theoretical_max, 1.0)

        self.assertAlmostEqual(
            service.dfc.loc[0, "NMaxQ"], expected, places=6, msg="Normalization formula not correctly implemented"
        )


if __name__ == "__main__":
    unittest.main()
