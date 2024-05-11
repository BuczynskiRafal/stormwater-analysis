import numpy as np
import pandas as pd
import swmmio as sw
from pyswmm import Simulation
from sa.core.data.predictor import classifier, recomendation
from sa.core.pipes.round import max_depth_value
from sa.core.pipes.valid_round import (
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


class DataManager(sw.Model):
    def __init__(self, in_file_path):
        super().__init__(in_file_path)
        self.frost_zone = "I"
        self.df_subcatchments = self.get_dataframe_safe(self.subcatchments.dataframe)
        self.df_nodes = self.get_dataframe_safe(self.nodes.dataframe)
        self.df_conduits = self.get_dataframe_safe(self.conduits())

    def get_dataframe_safe(self, df_supplier):
        df = df_supplier.dataframe if hasattr(df_supplier, "dataframe") else df_supplier
        return df.copy() if df is not None else None

    def __enter__(self):
        self.set_frost_zone(self.frost_zone)
        self.calculate()
        self.feature_engineering()
        self.conduits_recomendations()
        float_columns_subcatchments = self.df_subcatchments.select_dtypes(include=["float"]).columns
        float_columns_nodes = self.df_nodes.select_dtypes(include=["float"]).columns
        float_columns_conduits = self.df_conduits.select_dtypes(include=["float"]).columns
        self.df_subcatchments[float_columns_subcatchments] = self.df_subcatchments[float_columns_subcatchments].round(2)
        self.df_nodes[float_columns_nodes] = self.df_nodes[float_columns_nodes].round(2)
        self.df_conduits[float_columns_conduits] = self.df_conduits[float_columns_conduits].round(2)
        self.drop_unused()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        return False

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
        self.frost_zone = categories.get(frost_zone.upper(), 1.2)

    def subcatchments_classify(self, categories: bool = True) -> None:
        df = self.df_subcatchments[
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
            self.df_subcatchments["category"] = [categories[i] for i in predictions_cls]
        else:
            self.df_subcatchments["category"] = predictions_cls

    def nodes_subcatchment_name(self):
        """
        Map the 'Outlet' column in subcatchments_df to 'Name' in nodes_df to get the subcatchment name.
        """
        df = self.df_subcatchments
        outlet_to_subcatchment_map = df.reset_index().set_index("Outlet")["Name"].to_dict()
        self.df_nodes["Subcatchment"] = self.df_nodes.index.map(lambda node: outlet_to_subcatchment_map.get(node, "-"))

    def conduits_calculate_conduit_filling(self) -> None:
        """
        Calculates the conduit filling for a given SWMM model input file.
        Adding values unit is meter.
        """
        self.df_conduits["Filling"] = self.df_conduits["MaxDPerc"] * self.df_conduits["Geom1"]

    def conduits_filling_is_valid(self) -> None:
        """
        Check if the conduit filling is valid.
        Checks the filling of each conduit in the dataframe against its corresponding diameter.
        Adds a new column "ValMaxFill" to the dataframe indicating if the filling is valid (1) or invalid (0).
        """
        self.df_conduits["ValMaxFill"] = self.df_conduits.apply(
            lambda df: validate_filling(df["Filling"], df["Geom1"]), axis=1
        ).astype(int)

    def conduits_velocity_is_valid(self) -> None:
        """
        Validate maximum and minimum velocities in conduits.
        The results are stored as integer values (0 or 1) in two new columns:
        ValMaxV (1 if MaxV <= max_velocity_value, 0 otherwise) and
        ValMinV (1 if MaxV >= min_velocity_value, 0 otherwise).
        """
        self.df_conduits["ValMaxV"] = self.df_conduits.apply(lambda df: validate_max_velocity(df.MaxV), axis=1).astype(int)
        self.df_conduits["ValMinV"] = self.df_conduits.apply(lambda df: validate_min_velocity(df.MaxV), axis=1).astype(int)

    def conduits_slope_per_mile(self) -> None:
        """
        Calculate the slope per mile for each conduit in the network.
        """
        self.df_conduits["SlopePerMile"] = self.df_conduits["SlopeFtPerFt"] * 1000

    def conduits_slopes_is_valid(self) -> None:
        """
        Validates the maximum and minimum slopes of the conduits in the system by applying
        the `validate_max_slope` and `validate_min_slope` functions to the `conduits` DataFrame.

        The `ValMaxSlope` and `ValMinSlope` columns of the `conduits` DataFrame are updated
        with the validation results, with `1` indicating a valid slope and `0` indicating an invalid slope.
        """
        self.df_conduits["ValMaxSlope"] = self.df_conduits.apply(
            lambda df: validate_max_slope(slope=df["SlopePerMile"], diameter=df["Geom1"]),
            axis=1,
        ).astype(int)
        self.df_conduits["ValMinSlope"] = self.df_conduits.apply(
            lambda df: validate_min_slope(
                slope=df["SlopePerMile"],
                filling=df["Filling"],
                diameter=df["Geom1"],
            ),
            axis=1,
        ).astype(int)

    def conduits_max_depth(self) -> None:
        """
        Copies the 'MaxDepth' values from the model's nodes DataFrame to the 'conduits' DataFrame,
        using the 'InletNode' values to match the corresponding rows. A new 'MaxDepth' column is
        added to the 'conduits' DataFrame containing the copied values.
        """
        self.df_conduits["InletMaxDepth"] = self.df_conduits["InletNode"].map(self.df_nodes["MaxDepth"])
        self.df_conduits["OutletMaxDepth"] = self.df_conduits["OutletNode"].map(self.df_nodes["MaxDepth"])

    def conduits_calculate_max_depth(self) -> None:
        """
        Calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.

        First identifies any rows in the 'OutletMaxDepth' column of the 'conduits' dataframe that contain NaN values.
        For those rows, it calculates the outlet depth by subtracting the product of the conduit's length and slope
        from the inlet depth. The resulting values are then written to the 'OutletMaxDepth' column for those rows.
        """
        nan_rows = pd.isna(self.df_conduits["OutletMaxDepth"])
        self.df_conduits.loc[nan_rows, "OutletMaxDepth"] = self.df_conduits.loc[nan_rows, "InletMaxDepth"] - (
            self.df_conduits.loc[nan_rows, "Length"] * self.df_conduits.loc[nan_rows, "SlopeFtPerFt"]
        )

    def conduits_ground_elevation(self) -> None:
        """
        Calculates the amount of ground cover over each conduit's inlet and outlet.

        This method subtracts the maximum depth of each conduit's inlet and outlet from the corresponding node's invert
        elevation to determine the amount of ground cover over the inlet and outlet, respectively. The results
        are stored in the 'InletGroundElevation' and 'OutletGroundElevation' columns of the 'conduits' dataframe.
        """
        self.df_conduits["InletGroundElevation"] = self.df_conduits.InletNodeInvert + self.df_conduits.InletMaxDepth
        self.df_conduits["OutletGroundElevation"] = self.df_conduits.OutletNodeInvert + self.df_conduits.OutletMaxDepth

    def conduits_ground_cover(self) -> None:
        """
        Calculates the amount of ground cover over each conduit's inlet and outlet.

        This method subtracts the maximum depth of each conduit's inlet and outlet from the corresponding node's invert
        elevation to determine the amount of ground cover over the inlet and outlet, respectively. The results
        are stored in the 'InletGroundElevation' and 'OutletGroundElevation' columns of the 'conduits' dataframe.
        """
        self.df_conduits["InletGroundCover"] = (
            self.df_conduits.InletGroundElevation - self.df_conduits.InletNodeInvert - self.df_conduits.Geom1
        )
        self.df_conduits["OutletGroundCover"] = (
            self.df_conduits.OutletGroundElevation + self.df_conduits.OutletNodeInvert - self.df_conduits.Geom1
        )

    def conduits_depth_is_valid(self) -> None:
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
        cover elevation.
        """
        self.df_conduits["ValDepth"] = (
            ((self.df_conduits.InletNodeInvert - max_depth_value) <= self.df_conduits.InletGroundElevation)
            & ((self.df_conduits.OutletNodeInvert - max_depth_value) <= self.df_conduits.OutletGroundElevation)
        ).astype(int)

    def conduits_coverage_is_valid(self) -> None:
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
        self.df_conduits["ValCoverage"] = (
            (self.df_conduits.InletGroundCover >= self.frost_zone) & (self.df_conduits.OutletGroundCover >= self.frost_zone)
        ).astype(int)

    def conduits_recomendations(self, categories: bool = True) -> None:
        predictions = recomendation.predict(
            self.df_conduits[
                [
                    "Geom1",
                    "MaxQ",
                    "MaxV",
                    "MaxQPerc",
                    "MaxDPerc",
                    "InletNodeInvert",
                    "OutletNodeInvert",
                    "UpstreamInvert",
                    "DownstreamInvert",
                    "Filling",
                    "ValMaxFill",
                    "ValMaxV",
                    "ValMinV",
                    "SlopePerMile",
                    "ValMaxSlope",
                    "ValMinSlope",
                    "InletMaxDepth",
                    "OutletMaxDepth",
                    "InletGroundElevation",
                    "OutletGroundElevation",
                    "InletGroundCover",
                    "OutletGroundCover",
                    "ValDepth",
                    "ValCoverage",
                ]
            ]
        )
        predictions_cls = predictions.argmax(axis=-1)
        if categories:
            categories = [
                "valid",
                "pump",
                "tank",
                "seepage_boxes",
                "diameter_increase",
                "diameter_reduction",
                "slope_increase",
                "slope_reduction",
                "depth_increase",
                "depth_reduction",
            ]
            self.df_conduits["recomendation"] = [categories[i] for i in predictions_cls]
        else:
            self.df_conduits["recomendation"] = predictions_cls

    def conduits_subcatchment_name(self):
        """
        Map the subcatchment name form outlet node form Nodes to get the subcatchment name.
        """
        # 1. Początek kanału to "InletNode"
        # 2. Mapowanie "InletNode" do conduit.
        # print(self.nodes.columns)
        # self.conduits["Subcatchment"] = self.conduits["OutletNode"].map(self.model.nodes.dataframe["Subcatchment"])

    def drop_unused(self):
        # Clean up each DataFrame
        conduits_cols = [
            "OutOffset",
            "InitFlow",
            "Barrels",
            "Shape",
            "InOffset",
            "coords",
            "Geom2",
            "Geom3",
            "Geom4",
            "SlopeFtPerFt",
            "Type",
        ]
        nodes_cols = ["coords", "StageOrTimeseries"]
        subcatchments_cols = ["coords"]

        self.df_conduits.drop(columns=conduits_cols, inplace=True)
        self.df_nodes.drop(columns=nodes_cols, inplace=True)
        self.df_subcatchments.drop(columns=subcatchments_cols, inplace=True)

    def feature_engineering(self):
        # self.subcatchments_name_to_node()
        self.subcatchments_classify()
        self.nodes_subcatchment_name()
        self.conduits_calculate_conduit_filling()
        self.conduits_filling_is_valid()
        self.conduits_velocity_is_valid()
        self.conduits_slope_per_mile()
        self.conduits_slopes_is_valid()
        self.conduits_max_depth()
        self.conduits_calculate_max_depth()
        self.conduits_ground_elevation()
        self.conduits_ground_cover()
        self.conduits_depth_is_valid()
        self.conduits_coverage_is_valid()
        self.conduits_subcatchment_name()

    def calculate(self):
        with Simulation(self.inp.path) as sim:
            for _ in sim:
                pass

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
