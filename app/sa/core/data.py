import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import swmmio as sw
from pyswmm import Simulation
from swmmio.utils.functions import trace_from_node

from sa.core.predictor import classifier, recommendation
from sa.core.round import common_diameters, max_depth_value, min_slope
from sa.core.valid_round import (
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)

pd.set_option("display.width", 500)
np.set_printoptions(linewidth=500)
pd.set_option("display.max_columns", 30)


class DataManager(sw.Model):
    """
    Manages SWMM model data and performs preliminary data processing.
    Inherits from swmmio.Model for direct access to SWMM structures.
    """

    def __init__(self, in_file_path: str, crs: Optional[str] = None, include_rpt: bool = True):
        super().__init__(in_file_path, crs=crs, include_rpt=include_rpt)
        self.crs = crs
        self.include_rpt = include_rpt
        self.frost_zone = "I"  # Default zone; can be overridden in __enter__.

        self.df_subcatchments = self._get_df_safe(self.subcatchments.dataframe)
        self.df_nodes = self._get_df_safe(self.nodes.dataframe)
        self.df_conduits = self._get_df_safe(self.conduits())

    def _get_df_safe(self, df_source):
        """
        Safely obtains a DataFrame from a swmmio object or an existing DataFrame.
        Returns a copy if available, otherwise None.
        """
        df = getattr(df_source, "dataframe", df_source)
        return df.copy() if df is not None else None

    def __enter__(self):
        self.set_frost_zone(self.frost_zone)
        self.calculate()
        self.feature_engineering()
        self.conduits_recommendations()
        self._round_float_columns()
        self._drop_unused()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        return False

    def enter(self):
        """Alternative entry method equivalent to using 'with DataManager(...) as dm:'."""
        self.__enter__()

    def close(self, exc_type, exc_val, exc_tb):
        """Alternative exit method to clean up resources if needed."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    def set_frost_zone(self, frost_zone: str) -> None:
        """
        Sets the frost depth according to the climate zone.
        """
        zones = {
            "I": 1.0,
            "II": 1.2,
            "III": 1.4,
            "IV": 1.6,
        }
        self.frost_zone = zones.get(frost_zone.upper(), 1.2)

    def _round_float_columns(self) -> None:
        """
        Rounds all float columns in subcatchments, nodes, and conduits DataFrames to 2 decimal places.
        """
        for df in [self.df_subcatchments, self.df_nodes, self.df_conduits]:
            if df is not None:
                float_cols = df.select_dtypes(include=["float"]).columns
                df[float_cols] = df[float_cols].round(2)

    def _drop_unused(self) -> None:
        """
        Drops unused columns from each DataFrame to keep them clean.
        """
        unused_conduits = [
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
        unused_nodes = ["coords", "StageOrTimeseries"]
        unused_subcatchments = ["coords"]

        if self.df_conduits is not None:
            self.df_conduits.drop(columns=unused_conduits, inplace=True, errors="ignore")
        if self.df_nodes is not None:
            self.df_nodes.drop(columns=unused_nodes, inplace=True, errors="ignore")
        if self.df_subcatchments is not None:
            self.df_subcatchments.drop(columns=unused_subcatchments, inplace=True, errors="ignore")

    def calculate(self) -> None:
        """
        Performs a PySWMM simulation to update model values.
        """
        with Simulation(self.inp.path) as sim:
            for _ in sim:
                pass

    def feature_engineering(self) -> None:
        """
        Main entry point for feature engineering routines on subcatchments, nodes, and conduits.
        """
        self.subcatchments_classify()
        self.nodes_subcatchment_name()
        self.conduits_calculate_filling()
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

    # ------------------------
    # Subcatchment Methods
    # ------------------------

    def subcatchments_classify(self, categories: bool = True) -> None:
        """
        Classifies subcatchments using a trained classifier (e.g., ANN).
        Adds a 'category' column to df_subcatchments.
        """
        if self.df_subcatchments is None:
            return

        cols = [
            "Area",
            "PercImperv",
            "Width",
            "PercSlope",
            "PctZero",
            "TotalPrecip",
            "TotalRunoffMG",
            "PeakRunoff",
            "RunoffCoeff",
        ]
        df = self.df_subcatchments[cols].copy()
        df["TotalPrecip"] = pd.to_numeric(df["TotalPrecip"], errors="coerce").fillna(0)

        preds = classifier.predict(df)
        preds_cls = preds.argmax(axis=-1)

        if categories:
            labels = [
                "compact_urban_development",
                "urban",
                "loose_urban_development",
                "wooded_area",
                "grassy",
                "loose_soil",
                "steep_area",
            ]
            self.df_subcatchments["category"] = [labels[i] for i in preds_cls]
        else:
            self.df_subcatchments["category"] = preds_cls

    def nodes_subcatchment_name(self) -> None:
        """
        Maps subcatchment names to nodes based on 'Outlet' in df_subcatchments and node names in df_nodes.
        """
        if self.df_subcatchments is None or self.df_nodes is None:
            return

        mapping = (
            self.df_subcatchments.reset_index().set_index("Outlet")["Name"].to_dict()
            if "Outlet" in self.df_subcatchments.columns
            else {}
        )
        self.df_nodes["Subcatchment"] = self.df_nodes.index.map(lambda n: mapping.get(n, "-"))

    # ------------------------
    # Conduit Methods
    # ------------------------

    def conduits_calculate_filling(self) -> None:
        """
        Calculates conduit filling based on MaxDPerc * Geom1 (diameter).
        """
        if self.df_conduits is not None:
            self.df_conduits["Filling"] = self.df_conduits["MaxDPerc"] * self.df_conduits["Geom1"]

    def conduits_filling_is_valid(self) -> None:
        """
        Validates conduit filling. Adds 'ValMaxFill' column (1 if valid, 0 otherwise).
        """
        if self.df_conduits is not None:
            self.df_conduits["ValMaxFill"] = self.df_conduits.apply(
                lambda row: validate_filling(row["Filling"], row["Geom1"]), axis=1
            ).astype(int)

    def conduits_velocity_is_valid(self) -> None:
        """
        Validates conduit velocities. Adds 'ValMaxV' and 'ValMinV' columns.
        """
        if self.df_conduits is not None:
            self.df_conduits["ValMaxV"] = self.df_conduits.apply(lambda r: validate_max_velocity(r["MaxV"]), axis=1).astype(int)
            self.df_conduits["ValMinV"] = self.df_conduits.apply(lambda r: validate_min_velocity(r["MaxV"]), axis=1).astype(int)

    def conduits_slope_per_mile(self) -> None:
        """
        Converts slope from ft/ft to slope per mile (e.g., multiply by 1000).
        """
        if self.df_conduits is not None:
            self.df_conduits["SlopePerMile"] = self.df_conduits["SlopeFtPerFt"] * 1000

    def conduits_slopes_is_valid(self) -> None:
        """
        Validates conduit slopes. Adds 'ValMaxSlope' and 'ValMinSlope'.
        """
        if self.df_conduits is not None:
            self.df_conduits["ValMaxSlope"] = self.df_conduits.apply(
                lambda r: validate_max_slope(r["SlopePerMile"], r["Geom1"]), axis=1
            ).astype(int)
            self.df_conduits["ValMinSlope"] = self.df_conduits.apply(
                lambda r: validate_min_slope(r["SlopePerMile"], r["Filling"], r["Geom1"]), axis=1
            ).astype(int)

    def conduits_max_depth(self) -> None:
        """
        Copies MaxDepth from node DataFrame for InletNode and OutletNode.
        """
        if self.df_conduits is not None and self.df_nodes is not None:
            self.df_conduits["InletMaxDepth"] = self.df_conduits["InletNode"].map(self.df_nodes["MaxDepth"])
            self.df_conduits["OutletMaxDepth"] = self.df_conduits["OutletNode"].map(self.df_nodes["MaxDepth"])

    def conduits_calculate_max_depth(self) -> None:
        """
        For conduits with missing OutletMaxDepth, calculates it based on InletMaxDepth - (Length * SlopeFtPerFt).
        """
        if self.df_conduits is not None:
            nan_rows = pd.isna(self.df_conduits["OutletMaxDepth"])
            self.df_conduits.loc[nan_rows, "OutletMaxDepth"] = self.df_conduits.loc[nan_rows, "InletMaxDepth"] - (
                self.df_conduits.loc[nan_rows, "Length"] * self.df_conduits.loc[nan_rows, "SlopeFtPerFt"]
            )

    def ground_elevationground_elevation(self) -> None:
        """
        Computes ground elevation above the conduit (inlet/outlet).
        """
        if self.df_conduits is not None:
            self.df_conduits["InletGroundElevation"] = self.df_conduits["InletNodeInvert"] + self.df_conduits["InletMaxDepth"]
            self.df_conduits["OutletGroundElevation"] = self.df_conduits["OutletNodeInvert"] + self.df_conduits["OutletMaxDepth"]

    def conduits_ground_elevation(self) -> None:
        """Calculates the amount of ground cover
        over each conduit's inlet and outlet.
        """
        self.df_conduits["InletGroundElevation"] = self.df_conduits.InletNodeInvert + self.df_conduits.InletMaxDepth
        self.df_conduits["OutletGroundElevation"] = self.df_conduits.OutletNodeInvert + self.df_conduits.OutletMaxDepth

    def conduits_ground_cover(self) -> None:
        """
        Calculates the soil cover above conduit at inlet/outlet.
        """
        if self.df_conduits is not None:
            self.df_conduits["InletGroundCover"] = (
                self.df_conduits["InletGroundElevation"] - self.df_conduits["InletNodeInvert"] - self.df_conduits["Geom1"]
            )
            self.df_conduits["OutletGroundCover"] = (
                self.df_conduits["OutletGroundElevation"] + self.df_conduits["OutletNodeInvert"] - self.df_conduits["Geom1"]
            )

    def conduits_depth_is_valid(self) -> None:
        """
        Validates conduit depth based on node invert and max_depth_value.
        Adds 'ValDepth' column (1 if valid, 0 otherwise).
        """
        if self.df_conduits is not None:
            self.df_conduits["ValDepth"] = (
                ((self.df_conduits["InletNodeInvert"] - max_depth_value) <= self.df_conduits["InletGroundElevation"])
                & ((self.df_conduits["OutletNodeInvert"] - max_depth_value) <= self.df_conduits["OutletGroundElevation"])
            ).astype(int)

    def conduits_coverage_is_valid(self) -> None:
        """
        Checks if ground cover is >= frost_zone depth at inlet/outlet.
        Adds 'ValCoverage' column.
        """
        if self.df_conduits is not None:
            self.df_conduits["ValCoverage"] = (
                (self.df_conduits["InletGroundCover"] >= self.frost_zone)
                & (self.df_conduits["OutletGroundCover"] >= self.frost_zone)
            ).astype(int)

    def conduits_subcatchment_name(self) -> None:
        """
        Example stub method: Maps subcatchment names onto conduits if needed.
        """
        # Placeholder implementation if needed:
        # if self.df_conduits is not None and self.df_nodes is not None:
        #     self.df_conduits["Subcatchment"] = self.df_conduits["OutletNode"].map(self.df_nodes["Subcatchment"])
        pass

    def conduits_recommendations(self, categories: bool = True) -> None:
        """
        Generates recommendations via 'recommendation' model and adds 'recommendation' column.
        """
        if self.df_conduits is None:
            return

        cols = ["ValMaxFill", "ValMaxV", "ValMinV", "ValMaxSlope", "ValMinSlope", "ValDepth", "ValCoverage", "isMinDiameter"]
        for c in cols:
            if c not in self.df_conduits.columns:
                self.df_conduits[c] = 0

        preds = recommendation.predict(self.df_conduits[cols])
        preds_cls = preds.argmax(axis=-1)

        if categories:
            # Example category labels
            labels = ["diameter_reduction", "valid", "depth_increase"]
            self.df_conduits["recommendation"] = [labels[i] for i in preds_cls]
        else:
            self.df_conduits["recommendation"] = preds_cls

    # ------------------------
    # Overflowing / Trace Methods
    # ------------------------

    def all_traces(self) -> Dict[str, List[str]]:
        """
        Returns all SWMM model traces where each key is an outfall and each value is trace data (conduits).
        """
        outfalls = self.model.inp.outfalls.index
        return {outfall: trace_from_node(self.model.conduits, outfall) for outfall in outfalls}

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns conduits that exceeded maximum filling (ValMaxFill == 0).
        """
        return self.conduits_data.conduits[self.conduits_data.conduits["ValMaxFill"] == 0]

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Identifies segments in the system with overflowing conduits using all_traces().
        """
        all_tr = self.all_traces()
        overflow_df = self.overflowing_pipes()

        results = {}
        for outfall_id, trace_data in all_tr.items():
            overlap = [c for c in trace_data["conduits"] if c in overflow_df.index.tolist()]
            if overlap:
                indexes = {c: trace_data["conduits"].index(c) for c in overlap}
                results[outfall_id] = indexes

        # Return a trace for each overflowing segment
        return {
            key: trace_from_node(
                conduits=self.conduits_data.conduits,
                startnode=overflow_df.loc[list(value)[-1]]["InletNode"],
                mode="down",
                stopnode=overflow_df.loc[list(value)[0]]["OutletNode"],
            )
            for key, value in results.items()
        }

    def place_to_change(self) -> List[str]:
        """
        Determines where a technical change should be applied based on overflowing traces.
        Returns a list of nodes that may require intervention.
        """
        over_traces = self.overflowing_traces()
        locations = []
        for outfall, data in over_traces.items():
            if "nodes" in data:
                locations.append(data["nodes"][0])
        return locations

    # ------------------------
    # Recommendation / Helper Methods
    # ------------------------

    def generate_technical_recommendation(self) -> None:
        """Placeholder for generating and storing technical recommendations."""
        pass

    def apply_class(self) -> None:
        """
        Placeholder for node-based classification and manual labeling or recommendations.
        """
        pass

    def min_conduit_diameter(self) -> None:
        """
        Attempts to find the smallest diameter (from a predefined list) that can handle the flow (MaxQ) for each conduit.
        """
        if self.df_conduits is None:
            return

        diameters = []
        for _, row in self.df_conduits.iterrows():
            current_flow = row.get("MaxQ", 0.0)
            current_diam = row["Geom1"]
            current_slope = row["SlopePerMile"]
            val_fill = row.get("ValMaxFill", 0)

            if val_fill == 1:
                try:
                    idx = common_diameters.index(current_diam)
                except ValueError:
                    diameters.append(current_diam)
                    continue

                best_diam = current_diam
                for i in range(idx - 1, -1, -1):
                    candidate = common_diameters[i]
                    fill_height = self.calc_filling(current_flow, candidate, current_slope)
                    if validate_filling(fill_height, candidate):
                        best_diam = candidate
                    else:
                        break
                diameters.append(best_diam)
            else:
                diameters.append(current_diam)

        self.df_conduits["MinDiameter"] = diameters

    def is_min_diameter(self) -> None:
        """
        Marks conduits where the current diameter equals the minimal feasible diameter (1 if yes, 0 if no).
        """
        if self.df_conduits is not None:
            self.df_conduits["isMinDiameter"] = np.where(self.df_conduits["Geom1"] == self.df_conduits["MinDiameter"], 1, 0)

    # ------------------------
    # Hydraulic Calculations
    # ------------------------

    def calc_filling_percentage(self, filling: float, diameter: float) -> float:
        """
        Returns the percentage filling of the pipe.
        """
        if diameter <= 0:
            return 0.0
        return (filling / diameter) * 100

    def calc_area(self, h: float, d: float) -> float:
        """
        Computes the wetted cross-sectional area (m^2) for a circular pipe filled to height h.
        """
        if not validate_filling(h, d) or d <= 0:
            return 0.0

        radius = d / 2.0
        chord = 2.0 * math.sqrt(radius**2 - (h - radius) ** 2)
        alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

        if h > radius:
            return math.pi * radius**2 - 0.5 * (alpha - math.sin(alpha)) * radius**2
        elif h == radius:
            return 0.5 * math.pi * radius**2
        elif h == d:
            return math.pi * radius**2
        else:
            return 0.5 * (alpha - math.sin(alpha)) * radius**2

    def calc_u(self, filling: float, diameter: float) -> float:
        """
        Computes the wetted perimeter U for a circular pipe filled to 'filling'.
        """
        if not validate_filling(filling, diameter) or diameter <= 0:
            return 0.0

        radius = diameter / 2.0
        chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
        alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

        if filling > radius:
            return 2 * math.pi * radius - (alpha / (2 * math.pi)) * (2 * math.pi * radius)
        return (alpha / (2 * math.pi)) * (2 * math.pi * radius)

    def calc_rh(self, filling: float, diameter: float) -> float:
        """
        Computes the hydraulic radius Rh = A / U.
        """
        area = self.calc_area(filling, diameter)
        perimeter = self.calc_u(filling, diameter)
        return area / perimeter if perimeter else 0.0

    def calc_velocity(self, filling: float, diameter: float, slope: float) -> float:
        """
        Computes flow velocity [m/s] for a circular pipe using Manning's formula (n=0.013).
        Slope is given in ‰. Velocity = (1/n)*Rh^(2/3)*sqrt(i).
        """
        if not validate_filling(filling, diameter):
            return 0.0
        slope_m_m = slope / 1000.0  # converting ‰ to m/m
        rh = self.calc_rh(filling, diameter)
        return (1.0 / 0.013) * (rh ** (2 / 3)) * (slope_m_m**0.5)

    def calc_flow(self, h: float, d: float, i: float) -> float:
        """
        Computes flow rate Q [dm^3/s].
        - calc_area -> [m^2]
        - calc_velocity -> [m/s]
        - 1 m^3 = 1000 dm^3
        """
        if not validate_filling(h, d):
            return 0.0
        area = self.calc_area(h, d)
        velocity = self.calc_velocity(h, d, i)
        return area * velocity * 1000.0  # convert from m^3/s to dm^3/s

    def calc_filling(self, q: float, diameter: float, slope: float) -> float:
        """
        Iteratively approximates the filling height (m) needed to achieve flow q [dm^3/s].
        Uses a stepwise approach with a loop limit to prevent infinite loops.
        """
        if diameter <= 0 or q <= 0:
            return 0.0

        filling = 0.0
        step = 0.001
        max_iter = 10000

        for _ in range(max_iter):
            if filling > diameter:
                break
            flow = self.calc_flow(filling, diameter, slope)
            if flow >= q:
                break
            filling += step

        return filling

    def optimize_conduit_slope(self) -> None:
        """[Prototype] Adjust conduit slope based on min_slope logic."""
        self.model.conduits.SlopeFtPerFt = min_slope(
            filling=self.model.conduits.Filling,
            diameter=self.model.conduits.Geom1,
        )

    def optimize_conduit_depth(self) -> None:
        """[Prototype] Adjust conduit depth if needed."""
        pass
