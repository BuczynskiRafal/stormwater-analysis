from typing import Tuple

import swmmio

from stormwater_analysis.data.data import ConduitsData, NodeData, SubcatchmentData


def perform_conduits_feature_engineering(model: swmmio.Model) -> ConduitsData:
    """
    Performs feature engineering on the conduits data.

    Reads the SWMM model variable and performs the following feature engineering steps on the conduits data:
    - Sets the frost zone to selected category.
    - Drops unused columns from the conduits dataframe.
    - Calculates the filling of each conduit based on the flow rate.
    - Checks if the filling of each conduit is valid.
    - Checks if the velocity of each conduit is valid.
    - Calculates the slope per mile of each conduit.
    - Checks if the slopes of each conduit are valid.
    - Calculates the maximum depth of each conduit.
    - Calculates the amount of ground cover over each conduit's inlet and outlet.
    - Checks if the depth of each conduit is valid based on the inlet and outlet elevations and ground cover.
    - Checks if the ground cover over each conduit's inlet and outlet is valid.

    Args:
        model (swmmio.Model): The SWMM model containing the data to be processed.

    Returns:
        A ConduitsData object representing the conduits data after feature engineering steps have been applied.
    """
    conduits_data = ConduitsData(model)
    conduits_data.set_frost_zone("II")
    conduits_data.drop_unused()
    conduits_data.calculate_conduit_filling()
    conduits_data.filling_is_valid()
    conduits_data.velocity_is_valid()
    conduits_data.slope_per_mile()
    conduits_data.slopes_is_valid()
    conduits_data.max_depth()
    conduits_data.calculate_max_depth()
    conduits_data.inlet_ground_cover()
    conduits_data.depth_is_valid()
    conduits_data.coverage_is_valid()
    # print(conduits_data.conduits)
    return conduits_data


def perform_nodes_feature_engineering(model: swmmio.Model) -> NodeData:
    """
    Performs feature engineering on the nodes data.

    Reads the SWMM model variable and performs the following feature engineering steps on the nodes data:
    - Sets the frost zone to selected category.
    - Drops unused columns from the nodes dataframe.

    Args:
        model (swmmio.Model): The SWMM model containing the data to be processed.

    Returns:
        A NodeData object representing the nodes data after feature engineering steps have been applied.
    """
    nodes_data = NodeData(model)
    nodes_data.set_frost_zone("II")
    nodes_data.drop_unused()
    return nodes_data


def perform_subcatchments_feature_engineering(
    model: swmmio.Model,
) -> SubcatchmentData:
    """
    Performs feature engineering on the subcatchments data.

    Reads the SWMM model variable and performs the following feature engineering steps on the subcatchments data:
    - Sets the frost zone to selected category.
    - Drops unused columns from the subcatchments dataframe.

    Args:
        model (swmmio.Model): The SWMM model containing the data to be processed.


    Returns:
        SubcatchmentData: An instance of the SubcatchmentData class after feature engineering.
    """
    subcatchments_data = SubcatchmentData(model)
    subcatchments_data.set_frost_zone("II")
    subcatchments_data.drop_unused()
    subcatchments_data.classify(categories=True)
    return subcatchments_data


def feature_engineering(
    model: swmmio.Model,
) -> Tuple[ConduitsData, NodeData, SubcatchmentData]:
    """
    Performs feature engineering on the SWMM model data.

    Reads the SWMM model variable and performs the following feature engineering steps on the data:
    - Performs feature engineering on the conduits data.
    - Performs feature engineering on the nodes data.
    - Performs feature engineering on the subcatchments data.

    Returns:
        tuple: A tuple containing three processed data objects: (conduits_data, nodes_data, subcatchments_data), where
            - conduits_data (ConduitsData): The processed conduits data after feature engineering.
            - nodes_data (NodeData): The processed nodes data after feature engineering.
            - subcatchments_data (SubcatchmentsData): The processed subcatchments data after feature engineering.
    """
    conduits_data = perform_conduits_feature_engineering(model)
    nodes_data = perform_nodes_feature_engineering(model)
    subcatchments_data = perform_subcatchments_feature_engineering(model)
    return conduits_data, nodes_data, subcatchments_data
