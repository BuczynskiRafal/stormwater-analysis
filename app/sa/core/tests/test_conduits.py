import numpy as np
import pandas as pd
import pytest
import swmmio as sw

from sa.core.data import DataManager
from sa.core.tests import TEST_FILE
from sa.core.valid_round import validate_filling

desired_width = 500
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 30)


@pytest.fixture(scope="module")
def model():
    yield sw.Model(TEST_FILE, include_rpt=True)


@pytest.fixture(scope="class")
def data_manager(model):
    """Fixture to initialize the DataManager."""
    yield DataManager(model.inp.path)


class TestConduitsData:
    """
    A test class for the ConduitsData class, containing various test cases to ensure the correct
    functionality of the methods in ConduitsData.
    """

    def test_init(self, data_manager: DataManager):
        """
        Test the initialization of the ConduitsData object and ensure that the 'conduits' attribute
        is a pandas DataFrame.
        """
        assert isinstance(data_manager.dfc, pd.DataFrame)

    def test_calculate_filling(self, data_manager: DataManager):
        """
        Test the 'calculate_filling' method of the ConduitsData class to ensure that it
        correctly calculates and adds the 'Filling' column to the 'conduits' DataFrame.
        """
        data_manager.calculate_filling()
        assert "Filling" in data_manager.dfc.columns
        assert all(data_manager.dfc["Filling"] >= 0)

    def test_filling_is_valid(self, data_manager: DataManager):
        """
        Test the 'filling_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit filling and adds the 'ValMaxFill' column to the 'conduits' DataFrame.
        """
        data_manager.calculate_filling()
        data_manager.filling_is_valid()
        assert "ValMaxFill" in data_manager.dfc.columns
        assert all(data_manager.dfc["ValMaxFill"].isin([0, 1]))

    def test_validate_filling(self):
        """
        Test the 'validate_filling' function to ensure it correctly validates the filling value based
        on the provided diameter.
        """
        assert validate_filling(0, 1)
        assert validate_filling(0.5, 1)
        assert not validate_filling(1, 1)
        assert not validate_filling(1.1, 1)

    def test_velocity_is_valid_column(self, data_manager: DataManager):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        adds the 'ValMaxV' column to the 'conduits' DataFrame.
        """
        assert "ValMaxV" not in data_manager.dfc.columns
        data_manager.velocity_is_valid()
        assert "ValMaxV" in data_manager.dfc.columns

    def test_velocity_is_valid(self, data_manager: DataManager):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit velocities and updates the 'ValMaxV' and 'ValMinV' columns in the
        'conduits' DataFrame.
        """
        data_manager.velocity_is_valid()
        expected_values = [1, 1]
        assert list(data_manager.dfc["ValMaxV"])[:2] == expected_values
        assert list(data_manager.dfc["ValMinV"])[:2] == expected_values

    def test_slope_per_mile_column_added(self, data_manager: DataManager):
        """
        Test if the 'SlopePerMile' column is added to the conduits dataframe
        after calling the slope_per_mile() method.
        """
        data_manager.slope_per_mile()
        assert "SlopePerMile" in data_manager.dfc.columns

    def test_slope_per_mile_calculation(self, data_manager: DataManager):
        """
        Test if the calculated values in the 'SlopePerMile' column are correct after
        calling the slope_per_mile() method.
        """
        data_manager.slope_per_mile()
        expected_values = [1.80, 6.40]  # SlopeFtPerFt * 1000
        assert list(data_manager.dfc["SlopePerMile"])[:2] == pytest.approx(expected_values, abs=1e-9)

    def test_slopes_is_valid_columns_added(self, data_manager: DataManager):
        """
        Test if the 'ValMaxSlope' and 'ValMinSlope' columns are added to the conduits
        dataframe after calling slopes_is_valid() method.
        """
        data_manager.calculate_filling()
        data_manager.slopes_is_valid()
        assert "ValMaxSlope" in data_manager.dfc.columns
        assert "ValMinSlope" in data_manager.dfc.columns

    def test_slopes_is_valid_max_slope(self, data_manager: DataManager):
        """
        Test if the maximum slope validation is correct after
        calling the slopes_is_valid() method.
        """
        data_manager.calculate_filling()
        data_manager.slopes_is_valid()
        expected_values = [
            1,
            1,
        ]  # Assuming both conduits have valid maximum slopes
        assert list(data_manager.dfc["ValMaxSlope"])[:2] == expected_values

    def test_slopes_is_valid_min_slope(self, data_manager: DataManager):
        """
        Test if the minimum slope validation is correct after calling the slopes_is_valid() method.
        """
        data_manager.calculate_filling()
        data_manager.slopes_is_valid()
        expected_values = [
            1,
            1,
        ]  # Assuming both conduits have valid minimum slopes
        assert list(data_manager.dfc["ValMinSlope"])[:2] == expected_values

    def test_max_depth_columns_added(self, data_manager: DataManager):
        """
        Test if the 'InletMaxDepth' and 'OutletMaxDepth' columns are added to the conduits DataFrame after
        calling the `max_depth()` method.
        """
        data_manager.max_depth()
        assert "InletMaxDepth" in data_manager.dfc.columns
        assert "OutletMaxDepth" in data_manager.dfc.columns

    def test_max_depth_inlet_values_match(self, data_manager, model):
        """
        Test if the 'InletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'InletNode' as a reference.
        """
        data_manager.max_depth()
        nodes_data = model.nodes.dataframe
        for _, conduit in data_manager.dfc.iterrows():
            inlet_node = conduit["InletNode"]
            node_max_depth = nodes_data.loc[inlet_node, "MaxDepth"]
            conduit_inlet_max_depth = conduit["InletMaxDepth"]
            assert conduit_inlet_max_depth == node_max_depth

    def test_max_depth_outlet_values_match(self, data_manager, model):
        """
        Test if the 'OutletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'OutletNode' as a reference.
        """
        data_manager.max_depth()
        data_manager.calculate_max_depth()
        nodes_data = model.nodes.dataframe
        for _, conduit in data_manager.dfc.iterrows():
            outlet_node = conduit["OutletNode"]
            node_max_depth = nodes_data.loc[outlet_node, "MaxDepth"]
            conduit_outlet_max_depth = conduit["OutletMaxDepth"]
            if not pd.isna(conduit_outlet_max_depth) and not pd.isna(node_max_depth):
                assert conduit_outlet_max_depth == node_max_depth

    def test_frost_zone_valid_values(self):
        """
        Test setting frost_zone attribute with valid values.
        """
        valid_values = [1.0, 1.2, 1.4, 1.6]

        for value in valid_values:
            with DataManager(TEST_FILE) as data_manager:
                data_manager.frost_zone = value
                assert data_manager.frost_zone == value

    def test_frost_zone_invalid_values(self):
        """
        Test setting frost_zone attribute with invalid values.
        """
        invalid_values = [0.9, 1.7, -1.0, 2.0]

        for value in invalid_values:
            with pytest.raises(ValueError):
                with DataManager(TEST_FILE) as data_manager:
                    data_manager.frost_zone = value

    def test_frost_zone_setter_valid_values(self):
        """
        Test setting frost_zone attribute using the setter with valid values.
        """
        valid_values = [1.0, 1.2, 1.4, 1.6]
        with DataManager(TEST_FILE) as data_manager:
            for value in valid_values:
                data_manager.frost_zone = value
                assert data_manager.frost_zone == value

    def test_frost_zone_setter_invalid_values(self):
        """
        Test setting frost_zone attribute using the setter with invalid values.
        """
        invalid_values = [0.9, 1.7, -1.0, 2.0]
        with DataManager(TEST_FILE) as data_manager:
            for value in invalid_values:
                with pytest.raises(ValueError):
                    data_manager.frost_zone = value

    def test_max_ground_cover_valid_depths(self, data_manager: DataManager):
        """Test max_ground_cover_is_valid with valid depths for both inlet and outlet."""
        data_manager.dfc.at[0, "InletNodeInvert"] = 10
        data_manager.dfc.at[0, "OutletNodeInvert"] = 8
        data_manager.dfc.at[0, "InletGroundElevation"] = 15
        data_manager.dfc.at[0, "OutletGroundElevation"] = 13
        data_manager.dfc.at[0, "ValDepth"] = np.nan

        data_manager.max_ground_cover_is_valid()
        assert data_manager.dfc["ValDepth"].loc[0] == 1

    def test_max_ground_cover_invalid_depths(self, data_manager: DataManager):
        """Test max_ground_cover_is_valid with invalid depths for both inlet and outlet."""
        data_manager.dfc.at[0, "InletNodeInvert"] = 15
        data_manager.dfc.at[0, "OutletNodeInvert"] = 13
        data_manager.dfc.at[0, "InletGroundElevation"] = 35
        data_manager.dfc.at[0, "OutletGroundElevation"] = 32
        data_manager.dfc.at[0, "ValDepth"] = np.nan

        data_manager.max_ground_cover_is_valid()
        assert data_manager.dfc["ValDepth"].loc[0] == 0

    def test_calculate_max_depth(self, data_manager: DataManager):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        data_manager.max_depth()
        data_manager.dfc.at[0, "InletMaxDepth"] = 10
        data_manager.dfc.at[0, "Length"] = 100
        data_manager.dfc.at[0, "OutletMaxDepth"] = 5

        data_manager.dfc.at[1, "InletMaxDepth"] = 20
        data_manager.dfc.at[1, "Length"] = 200
        data_manager.dfc.at[1, "OutletMaxDepth"] = 10
        data_manager.calculate_max_depth()
        assert data_manager.dfc["OutletMaxDepth"].loc[0] == 5
        assert data_manager.dfc["OutletMaxDepth"].loc[1] == 10
        data_manager.dfc = data_manager.dfc.drop(index=[0, 1])

    def test_calculate_maximum_depth(self, data_manager: DataManager):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        data_manager.max_depth()
        # Set up some test data
        test_data = [
            {
                "InletMaxDepth": 10,
                "Length": 100,
                "SlopeFtPerFt": 0.01,
                "OutletMaxDepth": np.nan,
            },
            {
                "InletMaxDepth": 20,
                "Length": 200,
                "SlopeFtPerFt": 0.02,
                "OutletMaxDepth": np.nan,
            },
            {
                "InletMaxDepth": 30,
                "Length": 300,
                "SlopeFtPerFt": 0.03,
                "OutletMaxDepth": np.nan,
            },
        ]
        data_manager.dfc = pd.DataFrame(test_data)
        data_manager.calculate_max_depth()

        assert all(~pd.isna(data_manager.dfc["OutletMaxDepth"]))
        assert data_manager.dfc.loc[0, "OutletMaxDepth"] == pytest.approx(9, abs=1e-9)
        assert data_manager.dfc.loc[1, "OutletMaxDepth"] == pytest.approx(16, abs=1e-9)
        assert data_manager.dfc.loc[2, "OutletMaxDepth"] == pytest.approx(21, abs=1e-9)

    def test_min_ground_cover_is_valid(self, data_manager: DataManager):
        """
        Test the 'min_ground_cover_is_valid' method to ensure it correctly checks
        if the ground cover is above frost zone.
        """
        # Set up test data
        test_data = [
            {
                "InletNodeInvert": 10,
                "OutletNodeInvert": 20,
                "InletGroundElevation": 15,
                "OutletGroundElevation": 25,
                "InletGroundCover": 5,
                "OutletGroundCover": 5,
            },
            {
                "InletNodeInvert": 5,
                "OutletNodeInvert": 2,
                "InletGroundElevation": 4,
                "OutletGroundElevation": 1,
                "InletGroundCover": -1,
                "OutletGroundCover": -1,
            },
            {
                "InletNodeInvert": 30,
                "OutletNodeInvert": 40,
                "InletGroundElevation": 35,
                "OutletGroundElevation": 45,
                "InletGroundCover": 5,
                "OutletGroundCover": 5,
            },
        ]
        data_manager.dfc = pd.DataFrame(test_data)
        data_manager.frost_zone = 1.0

        # Call the method
        data_manager.min_ground_cover_is_valid()

        # Assertions
        assert data_manager.dfc.loc[0, "ValCoverage"] == 1
        assert data_manager.dfc.loc[1, "ValCoverage"] == 0
        assert data_manager.dfc.loc[2, "ValCoverage"] == 1

    def test_min_ground_cover_valid(self, data_manager: DataManager):
        """Test when both inlet and outlet ground covers are above frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.5
        data_manager.dfc.at[0, "OutletGroundCover"] = 1.4
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 1

    def test_min_ground_cover_invalid(self, data_manager: DataManager):
        """Test when both inlet and outlet ground covers are in frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 0.8
        data_manager.dfc.at[0, "OutletGroundCover"] = 0.9
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_mixed(self, data_manager: DataManager):
        """Test when inlet is valid but outlet is in frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.5
        data_manager.dfc.at[0, "OutletGroundCover"] = 0.9
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_edge_case(self, data_manager: DataManager):
        """Test when ground covers are exactly at frost zone depth."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.2
        data_manager.dfc.at[0, "OutletGroundCover"] = 1.2
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 1

    def test_ground_elevation(self, data_manager: DataManager):
        """
        Tests the 'ground_elevation' method of the ConduitsData class
        to ensure it correctly calculates ground cover over each conduit's inlet and outlet.
        """
        # Prepare test data
        test_data = [
            {"InletNodeInvert": 10, "InletMaxDepth": 2, "OutletNodeInvert": 40, "OutletMaxDepth": 8},
            {"InletNodeInvert": 20, "InletMaxDepth": 4, "OutletNodeInvert": 50, "OutletMaxDepth": 10},
            {"InletNodeInvert": 30, "InletMaxDepth": 6, "OutletNodeInvert": 60, "OutletMaxDepth": 12},
        ]
        data_manager.dfc = pd.DataFrame(test_data)

        # Call the method
        data_manager.calculate_ground_elevation()

        # Assertions for InletGroundElevation
        assert data_manager.dfc.loc[0, "InletGroundElevation"] == pytest.approx(
            12, abs=1e-9
        ), "Incorrect result for row 0 (Inlet)"
        assert data_manager.dfc.loc[1, "InletGroundElevation"] == pytest.approx(
            24, abs=1e-9
        ), "Incorrect result for row 1 (Inlet)"
        assert data_manager.dfc.loc[2, "InletGroundElevation"] == pytest.approx(
            36, abs=1e-9
        ), "Incorrect result for row 2 (Inlet)"

        # Assertions for OutletGroundElevation
        assert data_manager.dfc.loc[0, "OutletGroundElevation"] == pytest.approx(
            48, abs=1e-9
        ), "Incorrect result for row 0 (Outlet)"
        assert data_manager.dfc.loc[1, "OutletGroundElevation"] == pytest.approx(
            60, abs=1e-9
        ), "Incorrect result for row 1 (Outlet)"
        assert data_manager.dfc.loc[2, "OutletGroundElevation"] == pytest.approx(
            72, abs=1e-9
        ), "Incorrect result for row 2 (Outlet)"
