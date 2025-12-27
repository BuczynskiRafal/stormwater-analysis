"""
DataManager - Main orchestrator class for SWMM model data processing.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import swmmio as sw

from .conduits import ConduitFeatureEngineeringService, RecommendationService
from .enums import RecommendationCategory
from .nodes import NodeFeatureEngineeringService
from .predictor import gnn_recommendation, recommendation
from .round import max_slope, min_slope  # Re-export for backwards compatibility
from .services import HydraulicCalculationsService, SimulationRunnerService, TraceAnalysisService
from .subcatchments import SubcatchmentFeatureEngineeringService

# Re-exports for backwards compatibility
__all__ = [
    "DataManager",
    "HydraulicCalculationsService",
    "ConduitFeatureEngineeringService",
    "SubcatchmentFeatureEngineeringService",
    "NodeFeatureEngineeringService",
    "RecommendationService",
    "TraceAnalysisService",
    "SimulationRunnerService",
    "min_slope",
    "max_slope",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s")


class DataManager(sw.Model):
    """
    Main class that combines various services/classes responsible for different areas
    and manages data flow. Following SOLID principles, its purpose is mainly to orchestrate
    logic rather than implementing all logic in a single class.
    """

    def __init__(self, inp_file_path: str, crs: Optional[str] = None, include_rpt: bool = True, zone: float = 1.2):
        super().__init__(inp_file_path, crs=crs, include_rpt=include_rpt)
        self._frost_zone: float = zone
        self.frost_zone = zone

        # ---------------------------
        # DataFrames from swmmio model
        # ---------------------------
        self.dfs = self._get_df_safe(self.subcatchments.dataframe)
        self.dfn = self._get_df_safe(self.nodes.dataframe)
        # Fix for pandas merge error - ensure consistent column types
        conduits_df = self.conduits()
        if hasattr(conduits_df, "InletNode") and hasattr(conduits_df, "OutletNode"):
            conduits_df["InletNode"] = conduits_df["InletNode"].astype(str)
            conduits_df["OutletNode"] = conduits_df["OutletNode"].astype(str)
        self.dfc = self._get_df_safe(conduits_df)

        # ---------------------------
        # Initialization of services
        # ---------------------------
        self.subcatchment_service = SubcatchmentFeatureEngineeringService(self.dfs, self)
        self.node_service = NodeFeatureEngineeringService(self.dfn, self.dfs)
        self.conduit_service = ConduitFeatureEngineeringService(dfc=self.dfc, dfn=self.dfn, frost_zone=self.frost_zone)

        # MLP Recommendation Service is always available for fallback or experiments
        self.mlp_recommendation_service = RecommendationService(self.dfc, model=recommendation, model_name="MLP")
        logging.info("MLP Recommendation Service initialized.")

        # Store the GNN model if it's available, but don't initialize a service here
        self.gnn_model = gnn_recommendation
        if self.gnn_model:
            logging.info("GNN model is available and loaded.")
        else:
            logging.info("GNN model is not available.")

        self.simulation_service = SimulationRunnerService(self.inp.path)
        self.trace_analysis_service = TraceAnalysisService(self)

    @property
    def frost_zone(self) -> float:
        return self._frost_zone

    @frost_zone.setter
    def frost_zone(self, value: float) -> None:
        if not (0.8 <= value <= 1.6):
            raise ValueError("Frost zone must be between 0.8 and 1.6 meters")
        self._frost_zone = value

    def _get_df_safe(self, df_source):
        """
        Safely retrieve DataFrame from swmmio object or existing DataFrame.
        """
        df = getattr(df_source, "dataframe", df_source)
        if df is None:
            raise ValueError("DataFrame source is None")
        return df.copy()

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
        Excludes confidence columns which are already properly rounded.
        """
        confidence_cols = [f"confidence_{cat.value}" for cat in RecommendationCategory]

        for df in [self.dfs, self.dfn, self.dfc]:
            if df is not None:
                float_cols = df.select_dtypes(include=["float"]).columns
                # Exclude confidence columns from rounding
                cols_to_round = [col for col in float_cols if col not in confidence_cols]
                if cols_to_round:
                    df[cols_to_round] = df[cols_to_round].round(2)

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

        # Reinitialize parent sw.Model to include newly generated .rpt file
        try:
            super().__init__(self.inp.path, crs=self.crs, include_rpt=True)

            # Update DataManager's DataFrames with new data from reinitialized model
            new_conduits_df = self.conduits()
            if new_conduits_df is not None and not new_conduits_df.empty:
                self.dfc = new_conduits_df.copy()

            new_nodes_df = self.nodes()
            if new_nodes_df is not None and not new_nodes_df.empty:
                self.dfn = new_nodes_df.copy()

            new_subcatch_df = self.subcatchments()
            if new_subcatch_df is not None and not new_subcatch_df.empty:
                self.dfs = new_subcatch_df.copy()

            # Reinitialize services with updated DataFrames
            self.subcatchment_service = SubcatchmentFeatureEngineeringService(self.dfs, self)
            self.node_service = NodeFeatureEngineeringService(self.dfn, self.dfs)
            self.conduit_service = ConduitFeatureEngineeringService(dfc=self.dfc, dfn=self.dfn, frost_zone=self.frost_zone)

            # Choose recommendation service based on available models
            if gnn_recommendation is not None:
                logging.info("Using GNN model for recommendations (GraphSAGE)")
                self.recommendation_service = RecommendationService(self.dfc, model=gnn_recommendation, model_name="GNN")
            else:
                logging.info("Using MLP model for recommendations (fallback - GNN not available)")
                self.recommendation_service = RecommendationService(self.dfc, model=recommendation, model_name="MLP")

        except Exception as e:
            logging.warning(f"Could not reinitialize with report file: {e}")

    def feature_engineering(self) -> None:
        """
        Calls individual feature engineering stages for subcatchments,
        nodes, and conduits.
        """
        # Subcatchments
        self.subcatchment_service.subcatchments_classify(categories=True)

        # Nodes - zmienione z nodes_subcatchment_name() na nodes_subcatchment_info()
        self.node_service.nodes_subcatchment_info()

        # Conduits
        self.conduit_service.calculate_filling()
        self.conduit_service.filling_is_valid()
        self.conduit_service.velocity_is_valid()
        self.conduit_service.slope_per_mile()
        self.conduit_service.max_depth()
        self.conduit_service.calculate_max_depth()
        self.conduit_service.calculate_ground_elevation()
        self.conduit_service.ground_cover()
        self.conduit_service.max_ground_cover_is_valid()
        self.conduit_service.min_ground_cover_is_valid()
        self.conduit_service.min_conduit_diameter()
        self.conduit_service.diameter_features()

        self.conduit_service.conduits_subcatchment_info()
        self.conduit_service.propagate_subcatchment_info()

        self.conduit_service.normalize_roughness()
        self.conduit_service.normalize_max_velocity()
        self.conduit_service.normalize_depth()
        self.conduit_service.normalize_filling()
        self.conduit_service.normalize_max_q()
        self.conduit_service.normalize_ground_cover()
        self.conduit_service.normalize_slope()
        self.conduit_service.slopes_is_valid()
        self.conduit_service.slope_increase()
        self.conduit_service.slope_reduction()
        self.conduit_service.encode_sbc_category()

    def recommendations(self) -> None:
        """
        Generates recommendations using the 'recommendation' model.
        """
        self.recommendation_service.recommendations()

    ############################################################################
    #      ROUTING / OVERFLOW / RECOMMENDATION METHODS
    ############################################################################
    def all_traces(self):
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
