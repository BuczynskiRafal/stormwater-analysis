import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import swmmio as sw
from pyswmm import Simulation
from sa.core.data.predictor import classifier
from sa.core.pipes.round import common_diameters, max_depth_value, min_slope
from sa.core.pipes.valid_round import (
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)
from swmmio.utils.functions import trace_from_node

desired_width = 500
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 30)


class DataManager(sw.Model):
    def __init__(self, in_file_path: str, crs: Optional[str] = None, include_rpt: bool = True):
        super().__init__(in_file_path, crs=crs, include_rpt=include_rpt)
        self.crs = crs
        self.include_rpt = include_rpt
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
        self.conduits_recommendations()
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

    def enter(self):
        self.__enter__()

    def close(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)

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

    def conduits_recommendations(self, categories: bool = True) -> None:
        # predictions = recommendation.predict(
        #     self.df_conduits[
        #         [
        #             "Geom1",
        #             "MaxQ",
        #             "MaxV",
        #             "MaxQPerc",
        #             "MaxDPerc",
        #             "InletNodeInvert",
        #             "OutletNodeInvert",
        #             "UpstreamInvert",
        #             "DownstreamInvert",
        #             "Filling",
        #             "ValMaxFill",
        #             "ValMaxV",
        #             "ValMinV",
        #             "SlopePerMile",
        #             "ValMaxSlope",
        #             "ValMinSlope",
        #             "InletMaxDepth",
        #             "OutletMaxDepth",
        #             "InletGroundElevation",
        #             "OutletGroundElevation",
        #             "InletGroundCover",
        #             "OutletGroundCover",
        #             "ValDepth",
        #             "ValCoverage",
        #         ]
        #     ]
        # )
        # predictions_cls = predictions.argmax(axis=-1)
        # if categories:
        #     categories = [
        #         "valid",
        #         "pump",
        #         "tank",
        #         "seepage_boxes",
        #         "diameter_increase",
        #         "diameter_reduction",
        #         "slope_increase",
        #         "slope_reduction",
        #         "depth_increase",
        #         "depth_reduction",
        #     ]
        #     self.df_conduits["recommendation"] = [categories[i] for i in predictions_cls]
        # else:
        #     self.df_conduits["recommendation"] = predictions_cls
        import random

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
            predictions_cls = [random.randint(0, len(categories) - 1) for _ in range(len(self.df_conduits))]
            self.df_conduits["recommendation"] = [categories[i] for i in predictions_cls]
        else:
            self.df_conduits["recommendation"] = 0

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
        self.min_conduit_diameter()
        self.is_min_diameter()

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

    def all_traces(self) -> Dict[str, List[str]]:
        """
        Finds all traces in the SWMM model.

        A trace is a list of conduit IDs that connect a specific outfall to the rest of the network.

        Returns:
            Dict[str, List[str]]: A dictionary where the keys are outfall IDs and the values are lists
            of conduit IDs representing the traces connecting the outfalls to the rest of the network.
        """
        outfalls = self.model.inp.outfalls.index
        return {outfall: trace_from_node(self.model.conduits, outfall) for outfall in outfalls}

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns rain sewers in which the maximum filling height has been exceeded.

        Args:
            self: The instance of the class.

        Returns:
            pd.DataFrame: A DataFrame containing conduits
                        in which the maximum filling height has been exceeded.
        """
        return self.conduits_data.conduits[self.conduits_data.conduits["ValMaxFill"] == 0]

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Identifies the segments of the network system (from `all_traces`) where overflowing occurs.

        For each trace in `all_traces`, the function identifies the sections (conduits) where
        the maximum filling height has been exceeded (identified by `overflowing_pipes`).
        The output is a dictionary that contains these segments of overflowing. For each identified
        segment, the trace from the first conduit with overflowing to the last one is included.

        The dictionary keys are outfall IDs and the values are dictionaries where:
        - The dictionary keys are conduit IDs where overflowing occurs.
        - The dictionary values are the corresponding conduit's position in the trace.

        Finally, for each outfall, a trace is generated from the first to the last overflowing conduit.

        Returns
        -------
        dict
            A dictionary where the keys are outfall IDs and the values are traces from the first to
            the last conduit where overflowing occurs. Each trace is a list of tuples, where each
            tuple represents a conduit and the nodes it connects in the form (inlet_node, outlet_node, conduit_id).

        Notes
        -----
        This method requires that `self.all_traces()` and `self.overflowing_pipes()` be defined
        and return appropriate values. Specifically, `self.all_traces()` should return a dictionary
        of all traces and `self.overflowing_pipes()` should return a DataFrame of overflowing pipes.

        See Also
        --------
        all_traces : method to retrieve all traces in the system.
        overflowing_pipes : method to identify pipes where the maximum filling height is exceeded.

        Example
        -------
        >>> from core.inp_manage.inp import SwmmModel
        >>> swmm_model = SwmmModel(model, conduits_data, nodes_data, subcatchments_data)
        >>> swmm_model.overflowing_traces()
        >>> {'O4': {'nodes': ['J0', 'J1', 'J2', 'J3'], 'conduits': ['C1', 'C2', 'C3']}}
        """
        # Fetch the data
        all_traces = self.all_traces()
        overflowing_conduits = self.overflowing_pipes()

        overflowing_traces = {}
        for outfall_id, trace_data in all_traces.items():
            if overlapping_conduits := [c for c in trace_data["conduits"] if c in overflowing_conduits.index.tolist()]:
                # Create a sub-dict for each overlapping conduit and its position in the trace
                overflowing_trace = {c: trace_data["conduits"].index(c) for c in overlapping_conduits}
                overflowing_traces[outfall_id] = overflowing_trace
        # return {
        #     key:
        #     find_network_trace(
        #         self.model,
        #         overflowing_conduits.loc[list(value)[-1]]['InletNode'],
        #         overflowing_conduits.loc[list(value)[0]]['OutletNode'],
        #         )
        #         for key, value in overflowing_traces.items()
        #     }
        #     very interesting result
        #     >>> {'O4': [('J0', 'J1', 'C1'), ('J1', 'J2', 'C2'), ('J2', 'J3', 'C3')]}
        return {
            key: trace_from_node(
                conduits=self.conduits_data.conduits,
                startnode=overflowing_conduits.loc[list(value)[-1]]["InletNode"],
                mode="down",
                stopnode=overflowing_conduits.loc[list(value)[0]]["OutletNode"],
            )
            for key, value in overflowing_traces.items()
        }

    def place_to_change(self) -> List[str]:
        """
        Places pipes to apply a change in the SWMM model.

        1. Based on the list of overflowing conduits, determine
        where the recommended technical change should be applied.
        2. Returns a list of conduits or manholes where the change should be applied.

        Returns:
            List of nodes where the change should be applied.
        """

        # Get overflowing traces
        overflowing_traces = self.overflowing_traces()

        return [overflowing_traces[outfall]["nodes"][0] for outfall in overflowing_traces]

    def generate_technical_recommendation(self) -> None:
        """
        Generates a technical recommendation in the SWMM model.

        1. use a trained ANN to generate a technical recommendation.
        2. Prepare the data format in which you will store the recommendation.
        3. save the data to a file.
        4. Return the recommendation to the user.
        """

    def apply_class(self):
        """
        Recommendations made only for nodes.
            The plan is to classify all nodes in the first approach.
            In general I want the classifier to see the entire dataset.
            These are to be manually added learning labels.

            The plan:
                1. select Nodes
                2. I manually add recommendations / classifiers
                3. I add to the data frame.

            Below are some classes of recommendations.
        """
        pass

    def optimize_conduit_slope(self) -> None:
        # Currently, this function is not needed.
        # TODO: min_slope() returns a minimal slope as number/1000,  SlopeFtPerFt is a number.
        #       So we need to convert it to number/1000.
        #       SlopePerMile take number/1000, so there is no need to convert it to number/1000.
        self.model.conduits.SlopeFtPerFt = min_slope(
            filling=self.model.conduits.Filling,
            diameter=self.model.conduits.Geom1,
        )

    def optimize_conduit_depth(self):  # type: ignore
        # Currently, this function is not needed.
        pass

    def calc_filling_percentage(self, filling: float, diameter: float) -> float:
        """
        Calculate the percentage value of pipe filling height.

        Args:
            filling (int, float): pipe filling height [m]
            diameter (int, float): pipe diameter [m]

        Return:
            filled height (int, float): percentage of pipe that is filled with water.
        """
        return (filling / diameter) * 100

    def calc_u(self, filling: float, diameter: float) -> float:
        """
        Calculate the circumference of a wetted part of pipe

        Args:
            filling (int, float): pipe filling height [m]
            diameter (int, float): pipe diameter [m]

        Return:
            circumference (int, float): circumference of a wetted part of pipe
        """
        if validate_filling(filling, diameter):
            radius = diameter / 2
            chord = math.sqrt((radius**2 - (filling - radius) ** 2)) * 2
            alpha = math.degrees(math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2)))
            if filling > radius:
                return 2 * math.pi * radius - (alpha / 360 * 2 * math.pi * radius)
            return alpha / 360 * 2 * math.pi * radius

    def calc_rh(self, filling: float, diameter: float) -> float:
        """
        Calculate the hydraulic radius Rh, i.e. the ratio of the cross-section f
        to the contact length of the sewage with the sewer wall, called the wetted circuit U.

        Args:
            filling (int, float): pipe filling height [m]
            diameter (int, float): pipe diameter [m]

        Return:
            Rh (int, float): hydraulic radius [m]
        """
        try:
            return self.calc_area(filling, diameter) / self.calc_u(filling, diameter)
        except ZeroDivisionError:
            return 0

    def calc_velocity(self, filling: float, diameter: float, slope: float) -> float:
        """
        Calculate the speed of the sewage flow in the sewer.

        Args:
            filling (int, float): pipe filling height [m]
            diameter (int, float): pipe diameter [m]
            slope (int, float): fall in the bottom of the sewer [‰]

        Return:
            v (int, float): sewage flow velocity in the sewer [m/s]
        """
        slope /= 1000
        if validate_filling(filling, diameter):
            return 1 / 0.013 * self.calc_rh(filling, diameter) ** (2 / 3) * slope**0.5

    def calc_area(self, h: float, d: float) -> float:
        """
        Calculate the cross-sectional area of a pipe.
        Given its pipe filling height and diameter of pipe.

        Args:
            h (int, float): pipe filling height [m]
            d (int, float): pipe diameter [m]

        Return:
            area (int, float): cross-sectional area of the wetted part of the pipe [m2]
        """
        if validate_filling(h, d):
            radius = d / 2
            chord = math.sqrt((radius**2 - ((h - radius) ** 2))) * 2
            alpha = math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2))
            if h > radius:
                return math.pi * radius**2 - (1 / 2 * (alpha - math.sin(alpha)) * radius**2)
            elif h == radius:
                return math.pi * radius**2 / 2
            elif h == d:
                return math.pi * radius**2
            else:
                return 1 / 2 * (alpha - math.sin(alpha)) * radius**2

    def calc_flow(self, h: float, d: float, i: float) -> float:
        """
        Calculate sewage flow in the channel

        Args:
            h (int, float): pipe filling height [m]
            d (int, float): pipe diameter [m]
            i (int, float): fall in the bottom of the sewer [‰]

        Return:
            q (int, float): sewage flow in the channel [dm3/s]
        """
        if validate_filling(h, d):
            wet_area = self.calc_area(h, d)
            velocity = self.calc_velocity(h, d, i)
            return wet_area * velocity

    def calc_filling(self, q, diameter, slope):
        """
        Calculate the filling height needed to achieve a specified flow rate in a pipe.

        Args:
            q (int, float): desired flow rate [dm3/s]
            diameter (int, float): pipe diameter [m]
            slope (int, float): slope in the bottom of the sewer [‰]

        Return:
            filling (int, float): calculated filling height [m]
        """
        filling = 0
        flow = 0
        while flow < q:
            if validate_filling(filling, diameter):
                flow = self.calc_flow(filling, diameter, slope)
                filling += 0.001
            else:
                break

        print(f"flow: {flow:.2f}, q: {q}, filling: {filling}, ")
        return filling

    def min_conduit_diameter(self):
        min_diameters = []
        for index, row in self.df_conduits.iterrows():
            if row["ValMaxFill"] == 1:
                current_flow = row["MaxQ"]
                current_dim = row["Geom1"]
                current_slope = row["SlopePerMile"]
                current_dim_idx = common_diameters.index(current_dim)

                min_diameter = current_dim
                for i in range(current_dim_idx - 1, -1, -1):
                    smaller_diameter = common_diameters[i]
                    try:
                        filling = self.calc_filling(current_flow, smaller_diameter, current_slope)
                    except ValueError:
                        break
                    if validate_filling(filling, smaller_diameter):
                        min_diameter = smaller_diameter
                    else:
                        break

                min_diameters.append(min_diameter)
            else:
                min_diameters.append(current_dim)

        self.df_conduits["MinDiameter"] = min_diameters

    def is_min_diameter(self):
        """
        Determine if the current diameter is the minimum possible diameter
        that can handle the flow.

        Adding to conduits dataframe new column indicating
        whether the current diameter is the minimum diameter.
        """
        self.df_conduits["isMinDiameter"] = np.where(self.df_conduits["Geom1"] == self.df_conduits["MinDiameter"], 1, 0)
