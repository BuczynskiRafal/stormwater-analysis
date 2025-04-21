import pandas as pd
from unittest.mock import patch

from sa.core.data import ConduitFeatureEngineeringService, HydraulicCalculationsService


def test_min_conduit_diameter_valid_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.3], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]):
        with patch("sa.core.data.validate_filling", return_value=True):
            with patch.object(HydraulicCalculationsService, "calc_filling", return_value=0.3):
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.8


def test_min_conduit_diameter_invalid_filling():
    df = pd.DataFrame({"Geom1": [0.3], "Filling": [0.25], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [0]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]):
        with patch("sa.core.data.validate_filling") as mock_validate:
            mock_validate.side_effect = lambda f, d: d >= 0.4
            service.min_conduit_diameter()
            assert service.dfc.loc[0, "MinDiameter"] == 0.6


def test_min_conduit_diameter_very_small_flow():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.1], "MaxQ": [0.005], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.2, 0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_non_standard():
    df = pd.DataFrame({"Geom1": [0.35], "Filling": [0.2], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.35


def test_min_conduit_diameter_low_filling_ratio():
    df = pd.DataFrame({"Geom1": [0.6], "Filling": [0.15], "MaxQ": [0.03], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch.object(HydraulicCalculationsService, "calc_filling", return_value=0.15):
            with patch("sa.core.data.validate_filling", return_value=True):
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_max_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.33], "MaxQ": [0.1], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("sa.core.data.max_filling", return_value=0.827):
            with patch.object(HydraulicCalculationsService, "calc_filling", return_value=0.33):
                with patch("sa.core.data.validate_filling", return_value=True):
                    service.min_conduit_diameter()
                    assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_exceeds_max_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.34], "MaxQ": [0.1], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("sa.core.data.max_filling", return_value=0.827):
            with patch("sa.core.data.validate_filling") as mock_validate:
                mock_validate.side_effect = lambda f, d: d > 0.4
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.5


def test_min_conduit_diameter_zero_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.0], "MaxQ": [0.0], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_slope_too_small():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.2], "MaxQ": [0.1], "SlopeFtPerFt": [0.001], "ValMaxFill": [1]})
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with patch("sa.core.data.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("sa.core.data.validate_filling", return_value=True):
            with patch.object(HydraulicCalculationsService, "calc_filling") as mock_calc:
                mock_calc.side_effect = ValueError("Slope is too small")
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.5


def test_min_conduit_diameter_none_dataframe():
    service = ConduitFeatureEngineeringService(None, None, 1.0)
    service.min_conduit_diameter()
    assert service.dfc is None


def test_min_conduit_diameter_empty_dataframe():
    df = pd.DataFrame(columns=["Geom1", "Filling", "MaxQ", "SlopeFtPerFt", "ValMaxFill"])
    service = ConduitFeatureEngineeringService(df, None, 1.0)
    service.min_conduit_diameter()
    assert len(service.dfc) == 0
