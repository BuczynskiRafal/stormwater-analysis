import math
from enum import Enum
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


class RecommendationCategory(Enum):
    DIAMETER_REDUCTION = "diameter_reduction"
    VALID = "valid"
    DEPTH_INCREASE = "depth_increase"


###############################################################################
#                                   SERVICES
###############################################################################


def validate_inputs(func):
    """Decorator to validate filling and diameter inputs."""

    def wrapper(filling, diameter, *args, **kwargs):
        if not validate_filling(filling, diameter) or diameter <= 0:
            return 0.0
        if filling == 0:
            return 0.0
        return func(filling, diameter, *args, **kwargs)

    return wrapper


class HydraulicCalculationsService:
    @staticmethod
    def calc_filling_percentage(filling: float, diameter: float) -> float:
        """
        Returns the percentage filling of the pipe.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Percentage filling [%]. Returns 0.0 if filling is invalid or 0.
        """
        if not validate_filling(filling, diameter):
            return 0.0
        if filling == 0:
            return 0.0
        return (filling / diameter) * 100.0

    @staticmethod
    @validate_inputs
    def calc_area(filling: float, diameter: float) -> float:
        """
        Computes the wetted cross-sectional area (m^2) for a circular pipe filled to height h.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Wetted cross-sectional area [m^2].
        """
        radius = diameter / 2.0

        chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
        alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

        if filling > radius:
            area = math.pi * radius**2 - 0.5 * (alpha - math.sin(alpha)) * radius**2
        elif filling == radius:
            area = 0.5 * math.pi * radius**2
        elif filling == diameter:
            area = math.pi * radius**2
        else:
            area = 0.5 * (alpha - math.sin(alpha)) * radius**2

        return area

    @staticmethod
    @validate_inputs
    def calc_u(filling: float, diameter: float) -> float:
        """
        Computes the wetted perimeter U for a circular pipe filled to 'filling'.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Wetted perimeter [m].
        """
        radius = diameter / 2.0

        # Obliczenie długości cięciwy
        chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
        # Obliczenie kąta centralnego w radianach
        alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

        if filling > radius:
            if filling == diameter:
                perimeter = 2 * math.pi * radius  # Pełne wypełnienie
            else:
                perimeter = 2 * math.pi * radius - alpha * radius
        else:
            perimeter = alpha * radius

        return perimeter

    @staticmethod
    @validate_inputs
    def calc_rh(filling: float, diameter: float) -> float:
        """
        Computes the hydraulic radius Rh = A / U.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Hydraulic radius [m].
        """

        area = HydraulicCalculationsService.calc_area(filling, diameter)
        perimeter = HydraulicCalculationsService.calc_u(filling, diameter)
        rh = area / perimeter if perimeter else 0.0

        return rh

    @staticmethod
    @validate_inputs
    def calc_velocity(filling: float, diameter: float, slope: float) -> float:
        """
        Calculates the flow velocity using Manning's equation.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Flow velocity [m/s].

        Raises:
            ValueError: If slope is too small or exceeds maximum allowed value.
        """
        if not validate_max_slope(slope, diameter):
            raise ValueError("Slope exceeds maximum allowed value")
        if not validate_min_slope(slope, filling, diameter):
            raise ValueError("Slope is too small")

        # Manning's coefficient
        n = 0.013
        rh = HydraulicCalculationsService.calc_rh(filling, diameter)

        if rh == 0:
            return 0.0
        velocity = (1.0 / n) * (rh ** (2.0 / 3.0)) * math.sqrt(slope)
        return velocity

    @staticmethod
    @validate_inputs
    def calc_flow(filling: float, diameter: float, slope: float) -> float:
        """
        Computes flow rate Q [dm³/s].

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Flow rate [dm³/s].

        Raises:
            ValueError: If slope is too small or exceeds maximum allowed.
        """
        if not validate_max_slope(slope, diameter):
            raise ValueError("Slope exceeds maximum allowed value")
        if not validate_min_slope(slope, filling, diameter):
            raise ValueError("Slope is too small")

        area = HydraulicCalculationsService.calc_area(filling, diameter)
        velocity = HydraulicCalculationsService.calc_velocity(filling, diameter, slope)
        flow = area * velocity * 1000.0

        return flow

    @staticmethod
    def calc_filling(q: float, diameter: float, slope: float) -> float:
        """
        Iteratively approximates the filling height (m) needed to achieve flow q [dm³/s].
        Uses a stepwise approach with a loop limit to prevent infinite loops.

        Args:
            q (float): Flow rate [dm³/s].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Filling height [m].

        Raises:
            ValueError: If filling exceeds diameter without achieving desired flow.
        """
        if not (common_diameters[0] <= diameter <= common_diameters[-1]):
            raise ValueError("Diameter out of range for common diameters")
        if 0 > q:
            raise ValueError("Invalid flow rate, must be positive")
        if q == 0:
            return 0.0

        filling = 0.0
        step = 0.001
        max_iter = 10000

        for i in range(max_iter):
            if filling > diameter:
                break
            flow = HydraulicCalculationsService.calc_flow(filling, diameter, slope)
            if flow >= q:
                break
            filling += step

        if filling > diameter:
            raise ValueError("Filling exceeds diameter without achieving desired flow")

        return filling


class ConduitFeatureEngineeringService:
    def __init__(self, dfc: pd.DataFrame, dfn: pd.DataFrame, frost_zone: float):
        self.dfc = dfc
        self.dfn = dfn
        self.frost_zone = frost_zone

    def calculate_filling(self) -> None:
        if self.dfc is not None:
            self.dfc["Filling"] = self.dfc["MaxDPerc"] * self.dfc["Geom1"]

    def filling_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValMaxFill"] = self.dfc.apply(lambda row: validate_filling(row["Filling"], row["Geom1"]), axis=1).astype(
                int
            )

    def velocity_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValMaxV"] = self.dfc.apply(lambda r: validate_max_velocity(r["MaxV"]), axis=1).astype(int)
            self.dfc["ValMinV"] = self.dfc.apply(lambda r: validate_min_velocity(r["MaxV"]), axis=1).astype(int)

    def slope_per_mile(self) -> None:
        if self.dfc is not None:
            self.dfc["SlopePerMile"] = self.dfc["SlopeFtPerFt"] * 1000

    def slopes_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValMaxSlope"] = self.dfc.apply(lambda r: validate_max_slope(r["SlopePerMile"], r["Geom1"]), axis=1).astype(
                int
            )
            self.dfc["ValMinSlope"] = self.dfc.apply(
                lambda r: validate_min_slope(r["SlopePerMile"], r["Filling"], r["Geom1"]), axis=1
            ).astype(int)

    def max_depth(self) -> None:
        if self.dfc is not None and self.dfn is not None:
            self.dfc["InletMaxDepth"] = self.dfc["InletNode"].map(self.dfn["MaxDepth"])
            self.dfc["OutletMaxDepth"] = self.dfc["OutletNode"].map(self.dfn["MaxDepth"])

    def calculate_max_depth(self) -> None:
        if self.dfc is not None:
            nan_rows = pd.isna(self.dfc["OutletMaxDepth"])
            self.dfc.loc[nan_rows, "OutletMaxDepth"] = self.dfc.loc[nan_rows, "InletMaxDepth"] - (
                self.dfc.loc[nan_rows, "Length"] * self.dfc.loc[nan_rows, "SlopeFtPerFt"]
            )

    def calculate_ground_elevation(self) -> None:
        if self.dfc is not None:
            self.dfc["InletGroundElevation"] = self.dfc["InletNodeInvert"] + self.dfc["InletMaxDepth"]
            self.dfc["OutletGroundElevation"] = self.dfc["OutletNodeInvert"] + self.dfc["OutletMaxDepth"]

    def ground_cover(self) -> None:
        if self.dfc is None:
            return
        required_cols = ["InletGroundElevation", "InletNodeInvert", "Geom1"]
        if not all(col in self.dfc.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        self.dfc["InletGroundCover"] = self.dfc["InletGroundElevation"] - self.dfc["InletNodeInvert"] - self.dfc["Geom1"]
        self.dfc["OutletGroundCover"] = self.dfc["OutletGroundElevation"] - self.dfc["OutletNodeInvert"] - self.dfc["Geom1"]

    def max_ground_cover_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValDepth"] = (
                (self.dfc["InletNodeInvert"] >= (self.dfc["InletGroundElevation"] - max_depth_value))
                & (self.dfc["OutletNodeInvert"] >= (self.dfc["OutletGroundElevation"] - max_depth_value))
            ).astype(int)

    def min_ground_cover_is_valid(self) -> None:
        if self.dfc is None:
            return
        if "InletGroundCover" not in self.dfc.columns:
            self.ground_cover()

        self.dfc["ValCoverage"] = (
            (self.dfc["InletGroundCover"] >= self.frost_zone) & (self.dfc["OutletGroundCover"] >= self.frost_zone)
        ).astype(int)

    def min_conduit_diameter(self) -> None:
        if self.dfc is None:
            return

        common_diameters_sorted = sorted(common_diameters)
        diam_index_map = {diam: idx for idx, diam in enumerate(common_diameters_sorted)}

        def find_min_diam(row):
            val_fill = row.get("ValMaxFill", 0)
            current_diam = row["Geom1"]

            if val_fill != 1:
                return current_diam

            try:
                idx = diam_index_map[current_diam]
            except KeyError:
                return current_diam

            best_diam = current_diam
            for i in range(idx - 1, -1, -1):
                candidate = common_diameters_sorted[i]
                fill_height = HydraulicCalculationsService.calc_filling(row["MaxQ"], candidate, row["SlopePerMile"])
                if validate_filling(fill_height, candidate):
                    best_diam = candidate
                else:
                    break
            else:
                best_diam = common_diameters_sorted[0]

            return best_diam

        self.dfc["MinDiameter"] = self.dfc.apply(find_min_diam, axis=1)

    def is_min_diameter(self) -> None:
        if self.dfc is not None:
            self.dfc["isMinDiameter"] = np.where(self.dfc["Geom1"] == self.dfc["MinDiameter"], 1, 0)


class SubcatchmentFeatureEngineeringService:
    def __init__(self, dfs: pd.DataFrame):
        self.dfs = dfs

    def subcatchments_classify(self, categories: bool = True) -> None:
        """Classifies subcatchments using the classifier model (ANN).
        Adds 'category' column.
        """
        if self.dfs is None:
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
        df = self.dfs[cols].copy()
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
            self.dfs["category"] = [labels[i] for i in preds_cls]
        else:
            self.dfs["category"] = preds_cls


class NodeFeatureEngineeringService:
    """
    Maps subcatchment names to nodes based on the 'Outlet' column from dfs.
    """

    def __init__(self, dfn: pd.DataFrame, dfs: pd.DataFrame):
        self.dfn = dfn
        self.dfs = dfs

    def nodes_subcatchment_name(self) -> None:
        """
        Maps subcatchment names to nodes based on the 'Outlet' column from dfs.
        """
        if self.dfs is None or self.dfn is None:
            return

        mapping = {}
        if "Outlet" in self.dfs.columns:
            mapping = self.dfs.reset_index().set_index("Outlet")["Name"].to_dict()
        self.dfn["Subcatchment"] = self.dfn.index.map(lambda n: mapping.get(n, "-"))


class RecommendationService:
    """
    Class responsible for generating recommendations (e.g., depth change, diameter change, etc.).
    """

    def __init__(self, dfc: pd.DataFrame):
        self.dfc = dfc

    def recommendations(self, categories: bool = True) -> None:
        """
        Generates recommendations via 'recommendation' model and adds 'recommendation' column.
        """
        if self.dfc is None:
            return

        cols = [
            "ValMaxFill",
            "ValMaxV",
            "ValMinV",
            "ValMaxSlope",
            "ValMinSlope",
            "ValDepth",
            "ValCoverage",
            "isMinDiameter",
        ]
        for c in cols:
            if c not in self.dfc.columns:
                self.dfc[c] = 0

        preds = recommendation.predict(self.dfc[cols])
        preds_cls = preds.argmax(axis=-1)

        if categories:
            labels = [
                RecommendationCategory.DIAMETER_REDUCTION.value,
                RecommendationCategory.VALID.value,
                RecommendationCategory.DEPTH_INCREASE.value,
            ]
            self.dfc["recommendation"] = [labels[i] for i in preds_cls]
        else:
            self.dfc["recommendation"] = preds_cls


class TraceAnalysisService:
    """
    Class contains the logic for analyzing flows and overflows in the SWMM network.
    """

    def __init__(self, model: sw.Model):
        self.model = model

    def all_traces(self) -> Dict[str, List[str]]:
        """
        Returns the route (trace) for each outfall in the model.
        """
        outfalls = self.model.inp.outfalls.index
        return {outfall: trace_from_node(self.model.conduits, outfall) for outfall in outfalls}

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns all conduits that exceeded the allowed filling (ValMaxFill == 0).
        """
        return self.model.conduits_data.conduits[self.model.conduits_data.conduits["ValMaxFill"] == 0]

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Identifies segments with overflow in the model and returns their traces.
        """
        all_tr = self.all_traces()
        overflow_df = self.overflowing_pipes()

        results = {}
        for outfall_id, trace_data in all_tr.items():
            overlap = [c for c in trace_data["conduits"] if c in overflow_df.index.tolist()]
            if overlap:
                indexes = {c: trace_data["conduits"].index(c) for c in overlap}
                results[outfall_id] = indexes

        return {
            key: trace_from_node(
                conduits=self.model.conduits_data.conduits,
                startnode=overflow_df.loc[list(value)[-1]]["InletNode"],
                mode="down",
                stopnode=overflow_df.loc[list(value)[0]]["OutletNode"],
            )
            for key, value in results.items()
        }

    def place_to_change(self) -> List[str]:
        """
        Determines nodes where intervention is needed based on overflow traces.
        """
        over_traces = self.overflowing_traces()
        locations = []
        for outfall, data in over_traces.items():
            if "nodes" in data:
                locations.append(data["nodes"][0])
        return locations


class SimulationRunnerService:
    """Class responsible for running the simulation using PySWMM."""

    def __init__(self, inp_path: str):
        self.inp_path = inp_path

    def run_simulation(self) -> None:
        """Runs the PySWMM simulation in a loop to update model values."""
        with Simulation(self.inp_path) as sim:
            for _ in sim:
                pass


###############################################################################
#                                   DATA MANAGER
###############################################################################


class DataManager(sw.Model):
    """
    Main class that combines various services/classes responsible for different areas
    and manages data flow. Following SOLID principles, its purpose is mainly to orchestrate
    logic rather than implementing all logic in a single class.
    """

    def __init__(self, inp_file_path: str, crs: Optional[str] = None, include_rpt: bool = True, zone: float = 1.2):
        super().__init__(inp_file_path, crs=crs, include_rpt=include_rpt)
        self._frost_zone: float = None
        self.frost_zone = zone

        # ---------------------------
        # DataFrames from swmmio model
        # ---------------------------
        self.dfs = self._get_df_safe(self.subcatchments.dataframe)
        self.dfn = self._get_df_safe(self.nodes.dataframe)
        self.dfc = self._get_df_safe(self.conduits())

        # ---------------------------
        # Initialization of services
        # ---------------------------
        self.subcatchment_service = SubcatchmentFeatureEngineeringService(self.dfs)
        self.node_service = NodeFeatureEngineeringService(self.dfn, self.dfs)
        self.conduit_service = ConduitFeatureEngineeringService(dfc=self.dfc, dfn=self.dfn, frost_zone=self.frost_zone)
        self.recommendation_service = RecommendationService(self.dfc)
        self.simulation_service = SimulationRunnerService(self.inp.path)
        self.trace_analysis_service = TraceAnalysisService(self)

    @property
    def frost_zone(self) -> float:
        return self._frost_zone

    @frost_zone.setter
    def frost_zone(self, value: float) -> None:
        if not (1.0 <= value <= 1.6):
            raise ValueError("Frost zone must be between 1.0 and 1.6 meters")
        self._frost_zone = value

    def _get_df_safe(self, df_source):
        """
        Safely retrieve DataFrame from swmmio object or existing DataFrame.
        """
        df = getattr(df_source, "dataframe", df_source)
        return df.copy() if df is not None else None

    def __enter__(self):
        self.calculate()
        self.feature_engineering()
        self.recommendations()
        self._round_float_columns()
        self._drop_unused()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred: {exc_val}")
        return False

    ############################################################################
    #                   HELPER METHODS (round, drop, etc.)
    ############################################################################
    def _round_float_columns(self) -> None:
        """
        Rounds float columns in dfs, dfn, and dfc to 2 decimal places.
        """
        for df in [self.dfs, self.dfn, self.dfc]:
            if df is not None:
                float_cols = df.select_dtypes(include=["float"]).columns
                df[float_cols] = df[float_cols].round(2)

    def _drop_unused(self) -> None:
        """
        Removes unused columns from DataFrames to maintain cleanliness.
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

        if self.dfc is not None:
            self.dfc.drop(columns=unused_conduits, inplace=True, errors="ignore")
        if self.dfn is not None:
            self.dfn.drop(columns=unused_nodes, inplace=True, errors="ignore")
        if self.dfs is not None:
            self.dfs.drop(columns=unused_subcatchments, inplace=True, errors="ignore")

    ############################################################################
    #                              MAIN OPERATIONS
    ############################################################################
    def calculate(self) -> None:
        """
        Runs the simulation to update model values.
        """
        self.simulation_service.run_simulation()

    def feature_engineering(self) -> None:
        """
        Calls individual feature engineering stages for subcatchments,
        nodes, and conduits.
        """
        # Subcatchments
        self.subcatchment_service.subcatchments_classify(categories=True)

        # Nodes
        self.node_service.nodes_subcatchment_name()

        # Conduits
        self.conduit_service.calculate_filling()
        self.conduit_service.filling_is_valid()
        self.conduit_service.velocity_is_valid()
        self.conduit_service.slope_per_mile()
        self.conduit_service.slopes_is_valid()
        self.conduit_service.max_depth()
        self.conduit_service.calculate_max_depth()
        self.conduit_service.calculate_ground_elevation()
        self.conduit_service.ground_cover()
        self.conduit_service.max_ground_cover_is_valid()
        self.conduit_service.min_ground_cover_is_valid()
        self.conduit_service.min_conduit_diameter()
        self.conduit_service.is_min_diameter()

    def recommendations(self) -> None:
        """
        Generates recommendations using the 'recommendation' model.
        """
        self.recommendation_service.recommendations(categories=True)

    ############################################################################
    #      ROUTING / OVERFLOW / RECOMMENDATION METHODS
    ############################################################################
    def all_traces(self) -> Dict[str, List[str]]:
        """
        Returns all traces from the SWMM model.
        """
        return self.trace_analysis_service.all_traces()

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns overflowing pipes (ValMaxFill == 0).
        """
        return self.trace_analysis_service.overflowing_pipes()

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns all traces related to overflowing pipes.
        """
        return self.trace_analysis_service.overflowing_traces()

    def place_to_change(self) -> List[str]:
        """
        Determines which nodes require technical intervention.
        """
        return self.trace_analysis_service.place_to_change()

    def generate_technical_recommendation(self) -> None:
        """Further technical recommendations generation possible."""
        pass

    def apply_class(self) -> None:
        """
        Example method that can classify nodes or other elements
        based on data in dfn/dfc/dfs.
        """
        pass

    def optimize_conduit_slope(self) -> None:
        """
        [Prototype] Modifies conduit slopes, e.g., based on min_slope from sa.core.round.
        """
        # self.model.conduits in swmmio is an object,
        # here you can insert logical updates for slopes, etc.
        self.model.conduits.SlopeFtPerFt = min_slope(
            filling=self.model.conduits.Filling,
            diameter=self.model.conduits.Geom1,
        )

    def optimize_conduit_depth(self) -> None:
        """[Prototype] Modifies conduit depths if needed."""
        pass
