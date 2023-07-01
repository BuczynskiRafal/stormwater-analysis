from typing import Dict, List

import pandas as pd
import swmmio
from swmmio.utils.functions import trace_from_node

from stormwater_analysis.data.data import ConduitsData, NodesData, SubcatchmentsData
from stormwater_analysis.pipes.round import min_slope


class SwmmModel:
    """
    A class representing a Storm Water Management Model (SWMM) with processed data.

    Attributes:
        model (swmmio.Model): The original SWMM model.
        conduits_data (ConduitsData): The processed conduits data after feature engineering.
        nodes_data (NodesData): The processed nodes data after feature engineering.
        subcatchments_data (SubcatchmentsData): The processed subcatchments data after feature engineering.
    """

    def __init__(
        self,
        model: swmmio.Model,
        conduits_data: ConduitsData,
        nodes_data: NodesData,
        subcatchments_data: SubcatchmentsData,
    ) -> None:
        """
        Initializes a SwmmModel object with the given SWMM model and processed data.

        Args:
            model (swmmio.Model): The original SWMM model.
            conduits_data (ConduitsData): The processed conduits data after feature engineering.
            nodes_data (NodesData): The processed nodes data after feature engineering.
            subcatchments_data (SubcatchmentsData): The processed subcatchments data after feature engineering.
        """
        self.model = model
        self.conduits_data = conduits_data
        self.nodes_data = nodes_data
        self.subcatchments_data = subcatchments_data

    def all_traces(self) -> Dict[str, List[str]]:
        """
        Finds all traces in the SWMM model.

        A trace is a list of conduit IDs that connect a specific outfall to the rest of the network.

        Returns:
            Dict[str, List[str]]: A dictionary where the keys are outfall IDs and the values are lists
            of conduit IDs representing the traces connecting the outfalls to the rest of the network.
        """
        outfalls = self.model.inp.outfalls.index
        return {outfall: trace_from_node(self.conduits_data.conduits, outfall) for outfall in outfalls}

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

    # def overflowing_traces(self) -> pd.DataFrame:
    #     traces = self.all_traces()
    #     print(traces)
    #     print("\n")
    #     overflowing = self.overflowing_pipes()
    #     print(overflowing)
    #     # overflowing_traces =
    #     return traces

    def overflowing_traces(self) -> pd.DataFrame:
        traces = self.all_traces()
        overflowing = self.overflowing_pipes()
        overflowing_conduits = overflowing.index.tolist()

        overflowing_traces = {}
        for outfall, trace in traces.items():
            overlapping_conduits = list(set(trace["conduits"]) & set(overflowing_conduits))
            if overlapping_conduits:
                start_index = trace["conduits"].index(overlapping_conduits[0])
                end_index = trace["conduits"].index(overlapping_conduits[-1])
                overflowing_trace = trace["conduits"][start_index : end_index + 1]
                overflowing_traces[outfall] = {"conduits": overflowing_trace}

        return overflowing_traces

    def place_pipes_to_apply_change(self) -> None:
        """
        Places pipes to apply a change in the SWMM model.

        1. Na podstawie listy przepełnionych kanałów określić
           gdzie należy zastosować rekomendowaną zmianę techniczną.
        2. Zwrócić liste kanałów lub studzienek na których zastosować zmianę.
        """

    def generate_technical_recommendation(self) -> None:
        """
        Generates a technical recommendation in the SWMM model.

        1. Użyj wytrenowanego ANN do generowania rekomendacji technicznej.
        2. Przygotuj format danych w jakich bęziesz przechowywał rekomendację.
        3. Zapisz dane do pliku.
        4. Zwrócić użytkownikowi rekomendację.
        """

    def optimize_conduit_slope(self) -> None:
        # Currently, this function is not needed.
        # TODO: min_slope() returns a minimal slope as number/1000,  SlopeFtPerFt is a number.
        #       So we need to convert it to number/1000.
        #       SlopePerMile take number/1000, so there is no need to convert it to number/1000.
        self.conduits_data.conduits.SlopeFtPerFt = min_slope(
            filling=self.conduits_data.conduits.Filling,
            diameter=self.conduits_data.conduits.Geom1,
        )

    def optimize_conduit_depth(self):  # type: ignore
        # Currently, this function is not needed.
        pass

    def optimize_conduit_diameter(self):  # type: ignore
        # Currently, this function is not needed.
        pass

    def optimize(self):  # type: ignore
        # Currently, this function is not needed.
        pass
