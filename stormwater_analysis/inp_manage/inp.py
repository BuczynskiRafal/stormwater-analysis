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
        >>> from stormwater_analysis.inp_manage.inp import SwmmModel
        >>> swmm_model = SwmmModel(model, conduits_data, nodes_data, subcatchments_data)
        >>> swmm_model.overflowing_traces()
        >>> {'O4': {'nodes': ['J0', 'J1', 'J2', 'J3'], 'conduits': ['C1', 'C2', 'C3']}}
        """
        # Fetch the data
        all_traces = self.all_traces()
        overflowing_conduits = self.overflowing_pipes()

        overflowing_traces = {}
        for outfall_id, trace_data in all_traces.items():
            # Check for overlap between this trace's conduits and the overflowing ones
            overlapping_conduits = [c for c in trace_data["conduits"] if c in overflowing_conduits.index.tolist()]

            if overlapping_conduits:
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

        # Prepare list to store nodes to apply changes
        nodes_to_apply_change = []

        # Loop through each outfall in the overflowing traces
        for outfall in overflowing_traces:
            # Add the first node of each trace to the list
            nodes_to_apply_change.append(overflowing_traces[outfall]["nodes"][0])

        return nodes_to_apply_change

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
