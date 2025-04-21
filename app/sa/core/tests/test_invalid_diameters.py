import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath("app"))

from sa.core.data import ConduitFeatureEngineeringService


def create_test_df(diameter, slope):
    return pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [diameter * 0.5 if diameter > 0 else 0]})


def test_invalid_diameters():
    for diameter in [0.0, -0.5, 2.5, float("nan")]:
        with pytest.raises(ValueError):
            df = create_test_df(diameter, 50.0)
            service = ConduitFeatureEngineeringService(df, None, 1.0)
            service.slope_reduction()
            assert "ReduceSlope" in service.dfc.columns
            print(f"Test passed for diameter: {diameter}")


if __name__ == "__main__":
    test_invalid_diameters()
    print("All tests passed successfully!")
