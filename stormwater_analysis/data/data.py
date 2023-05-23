from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import swmmio as sw

from stormwater_analysis.data.predictor import classifier
from stormwater_analysis.pipes.round import max_depth_value
from stormwater_analysis.pipes.valid_round import (
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)

desired_width = 500
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 30)


class Data(ABC):
    """
    Abstract base class for data classes.
    """

    def __init__(self, model: sw.Model) -> None:
        self.model = model

    @abstractmethod
    def set_frost_zone(self, frost_zone: str) -> None:
        """
        Abstract method for setting the frost zone value for a data instance.

        Args:
            frost_zone (str): A string representing the frost zone category, e.g., "I", "II", "III", "IV".
        """
        pass

    def drop_unused(self) -> None:
        """
        Drops unused columns from the data dataframe.
        """
        pass


class ConduitsData(Data):
    """
    Class for handling conduit data.
    """

    def __init__(self, model: sw.Model) -> None:
        super().__init__(model)
        self.conduits = model.conduits().copy()
        self.frost_zone = None

    def set_frost_zone(self, frost_zone: str) -> None:
        """
        Set the frost zone value for the ConduitsData instance.

        According to several standards (BN-83/8836-02, PN-81/B-03020, and others),
        the depth of the pipeline and tanks should be such that its cover from
        the external edge (upper edge) of the pipe (tank) to the elevation of
        the terrain is greater than the freezing depth by 20 cm (table).

        Args:
            frost_zone (str): A string representing the frost zone category, e.g., "I", "II", "III", "IV".

        """
        categories = {
            "I": 1,
            "II": 1.2,
            "III": 1.4,
            "IV": 1.6,
        }
        self.frost_zone = categories.get(frost_zone.upper(), 1.2)  # type: ignore

    def get_tag(self):  # type: ignore
        pass

    def drop_unused(self) -> None:
        """
        Drops unused columns from the conduits dataframe.
        """
        self.conduits = self.conduits.drop(
            columns=[
                "OutOffset",
                "InitFlow",
                "Barrels",
                "Shape",
                "InOffset",
                "coords",
                "Geom2",
                "Geom3",
                "Geom4",
            ]
        )

    def calculate_conduit_filling(self) -> None:
        """
        Calculates the conduit filling for a given SWMM model input file.
        Adding values unit is meter.
        """
        self.conduits["Filling"] = self.conduits["MaxDPerc"] * self.conduits.Geom1

    def filling_is_valid(self) -> None:
        """
        Check if the conduit filling is valid.
        Checks the filling of each conduit in the dataframe against its corresponding diameter.
        Adds a new column "ValMaxFill" to the dataframe indicating if the filling is valid (1) or invalid (0).
        """
        self.conduits["ValMaxFill"] = self.conduits.apply(lambda df: validate_filling(df.Filling, df.Geom1), axis=1).astype(int)

    def velocity_is_valid(self) -> None:
        """
        Validate maximum and minimum velocities in conduits.
        The results are stored as integer values (0 or 1) in two new columns:
        ValMaxV (1 if MaxV <= max_velocity_value, 0 otherwise) and
        ValMinV (1 if MaxV >= min_velocity_value, 0 otherwise).
        """
        self.conduits["ValMaxV"] = self.conduits.apply(lambda df: validate_max_velocity(df.MaxV), axis=1).astype(int)
        self.conduits["ValMinV"] = self.conduits.apply(lambda df: validate_min_velocity(df.MaxV), axis=1).astype(int)

    def slope_per_mile(self) -> None:
        """
        Calculate the slope per mile for each conduit in the network.
        """
        self.conduits["SlopePerMile"] = self.conduits.SlopeFtPerFt * 1000

    def slopes_is_valid(self) -> None:
        """
        Validates the maximum and minimum slopes of the conduits in the system by applying
        the `validate_max_slope` and `validate_min_slope` functions to the `conduits` DataFrame.

        The `ValMaxSlope` and `ValMinSlope` columns of the `conduits` DataFrame are updated
        with the validation results, with `1` indicating a valid slope and `0` indicating an invalid slope.
        """
        self.conduits["ValMaxSlope"] = self.conduits.apply(
            lambda df: validate_max_slope(slope=df.SlopeFtPerFt * 1000, diameter=df.Geom1),
            axis=1,
        ).astype(int)
        self.conduits["ValMinSlope"] = self.conduits.apply(
            lambda df: validate_min_slope(
                slope=df.SlopeFtPerFt * 1000,
                filling=df.Filling,
                diameter=df.Geom1,
            ),
            axis=1,
        ).astype(int)

    def max_depth(self) -> None:
        """
        Copies the 'MaxDepth' values from the model's nodes DataFrame to the 'conduits' DataFrame,
        using the 'InletNode' values to match the corresponding rows. A new 'MaxDepth' column is
        added to the 'conduits' DataFrame containing the copied values.
        """
        self.conduits["InletMaxDepth"] = self.conduits["InletNode"].map(self.model.nodes.dataframe["MaxDepth"])
        self.conduits["OutletMaxDepth"] = self.conduits["OutletNode"].map(self.model.nodes.dataframe["MaxDepth"])

    def calculate_max_depth(self) -> None:
        """
        Calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.

        First identifies any rows in the 'OutletMaxDepth' column of the 'conduits' dataframe that contain NaN values.
        For those rows, it calculates the outlet depth by subtracting the product of the conduit's length and slope
        from the inlet depth. The resulting values are then written to the 'OutletMaxDepth' column for those rows.
        """
        nan_rows = pd.isna(self.conduits.OutletMaxDepth)
        self.conduits.loc[nan_rows, "OutletMaxDepth"] = self.conduits.loc[nan_rows, "InletMaxDepth"] - (
            self.conduits.loc[nan_rows, "Length"] * self.conduits.loc[nan_rows, "SlopeFtPerFt"]
        )

    def inlet_ground_cover(self) -> None:
        """
        Calculates the amount of ground cover over each conduit's inlet and outlet.

        This method subtracts the maximum depth of each conduit's inlet and outlet from the corresponding node's invert
        elevation to determine the amount of ground cover over the inlet and outlet, respectively. The results
        are stored in the 'InletGroundCover' and 'OutletGroundCover' columns of the 'conduits' dataframe.
        """
        self.conduits["InletGroundCover"] = self.conduits.InletNodeInvert - self.conduits.InletMaxDepth
        self.conduits["OutletGroundCover"] = self.conduits.OutletNodeInvert - self.conduits.OutletMaxDepth

    def depth_is_valid(self) -> None:
        """
        Checks if the depth of each conduit is valid based on the inlet and
        outlet elevations and ground cover.

        This method creates a new column in the 'conduits' dataframe
        called 'ValDepth', which is set to 1 if the depth of the conduit
        is valid and 0 if it is not. The depth is considered valid if it
        falls within the range between the inlet invert elevation
        minus the maximum depth and the inlet ground cover elevation,
        and also within the range between the outlet
        invert elevation minus the maximum depth and the outlet ground
        cover elevation. The 'max_depth_value'
        parameter used in the calculations is specified
        in the class constructor.
        """
        self.conduits["ValDepth"] = (
            ((self.conduits.InletNodeInvert - max_depth_value) <= self.conduits.InletGroundCover)
            & ((self.conduits.OutletNodeInvert - max_depth_value) <= self.conduits.OutletGroundCover)
        ).astype(int)

    def coverage_is_valid(self) -> None:
        """
        Checks if the ground cover over each conduit's inlet and outlet is valid.

        This method creates a new column in the 'conduits' dataframe called 'ValCoverage',
        which is set to 1 if the ground cover over
        the inlet and outlet of the conduit is valid and 0 if it is not.
        The ground cover is considered valid if it is less than or
        equal to the difference between the node's invert elevation and the frost
        zone depth. The 'frost_zone' parameter used in the
        calculations is specified in the class constructor.
        """
        self.conduits["ValCoverage"] = (
            (self.conduits.InletGroundCover <= (self.conduits.InletNodeInvert - self.frost_zone))
            & (self.conduits.OutletGroundCover <= (self.conduits.OutletNodeInvert - self.frost_zone))
        ).astype(int)


class NodeData(Data):
    def __init__(self, model: sw.Model) -> None:
        super().__init__(model)
        self.nodes = model.nodes.dataframe.copy()
        self.frost_zone = None

    def set_frost_zone(self, frost_zone: str) -> None:
        """
        Set the frost zone value for the NodeData instance.

        According to several standards (BN-83/8836-02, PN-81/B-03020, and others),
        the depth of the pipeline and tanks should be such that its cover from
        the external edge (upper edge) of the pipe (tank) to the elevation of
        the terrain is greater than the freezing depth by 20 cm (table).

        Args:
            frost_zone (str): A string representing the frost zone category, e.g., "I", "II", "III", "IV".

        """
        categories = {
            "I": 1,
            "II": 1.2,
            "III": 1.4,
            "IV": 1.6,
        }
        self.frost_zone = categories.get(frost_zone.upper(), 1.2)  # type: ignore

    def drop_unused(self) -> None:
        """
        Drops unused columns from the nodes dataframe.
        """
        self.nodes = self.nodes.drop(
            columns=[
                #
            ]
        )

    # TODO calculate MAxDepth for outlets
    # def max_depth(self):
    #     self.nodes["MaxDepth"] = self.conduits["InletNode"].map(
    #         self.model.nodes.dataframe["MaxDepth"]
    #     )

    # def calculate_max_depth(self):
    #     """
    #     Calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
    #
    #     First identifies any rows in the 'OutletMaxDepth' column of the 'conduits' dataframe that contain NaN values.
    #     For those rows, it calculates the outlet depth by subtracting the product of the conduit's length and slope
    #     from the inlet depth. The resulting values are then written to the 'OutletMaxDepth' column for those rows.
    #     """
    #     nan_rows = pd.isna(self.nodes.OutletMaxDepth)
    #     self.nodes.loc[nan_rows, "OutletMaxDepth"] = self.conduits.loc[
    #         nan_rows, "InletMaxDepth"
    #     ] - (
    #         self.nodes.loc[nan_rows, "Length"]
    #         * self.nodes.loc[nan_rows, "SlopeFtPerFt"]
    #     )


class SubcatchmentData(Data):
    """
    Data class for subcatchments.
    """

    def __init__(self, model: sw.Model) -> None:
        super().__init__(model)
        self.subcatchments = model.subcatchments.dataframe.copy()
        self.frost_zone = None

    def set_frost_zone(self, frost_zone: str) -> None:
        """
        Sets the frost zone value for the SubcatchmentData instance.

        Args:
            frost_zone (str): A string representing the frost zone category, e.g., "I", "II", "III", "IV".
        """
        categories = {
            "I": 1,
            "II": 1.2,
            "III": 1.4,
            "IV": 1.6,
        }
        self.frost_zone = categories.get(frost_zone.upper(), 1.2)  # type: ignore

    def drop_unused(self) -> None:
        """
        Drops unused columns from the subcatchments dataframe.
        """
        self.subcatchments = self.subcatchments.drop(
            columns=[
                #
            ]
        )

    def classify(self, categories: bool = True) -> None:
        df = self.subcatchments[
            ["Area", "PercImperv", "Width", "PercSlope", "PctZero", "TotalPrecip", "TotalRunoffMG", "PeakRunoff", "RunoffCoeff"]
        ].copy()
        df["TotalPrecip"] = pd.to_numeric(df["TotalPrecip"])
        predictions = classifier.predict(df)
        predictions_cls = predictions.argmax(axis=-1)
        if categories:
            categories = [
                "compact_urban_development",
                "urban",
                "loose_urban_development",
                "wooded_area",
                "grassy",
                "loose_soil",
                "steep_area",
            ]
            self.subcatchments["category"] = [categories[i] for i in predictions_cls]
        else:
            self.subcatchments["category"] = predictions_cls
