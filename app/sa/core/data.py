import math
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import swmmio as sw
from pyswmm import Simulation
from swmmio.utils.functions import trace_from_node

from sa.core.predictor import classifier, recommendation
from sa.core.round import common_diameters, max_depth_value, min_slope, max_slope, max_velocity_value, max_filling
from sa.core.valid_round import (
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)


class RecommendationCategory(Enum):
    PUMP = "pump"
    TANK = "tank"
    SEEPAGE_BOXES = "seepage_boxes"
    DIAMETER_INCREASE = "diameter_increase"
    DIAMETER_REDUCTION = "diameter_reduction"
    SLOPE_INCREASE = "slope_increase"
    SLOPE_REDUCTION = "slope_reduction"
    DEPTH_INCREASE = "depth_increase"
    DEPTH_REDUCTION = "depth_reduction"
    VALID = "valid"


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
        Computes flow rate Q [m³/s].

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Flow rate [m³/s].

        Raises:
            ValueError: If slope is too small or exceeds maximum allowed.
        """
        if not validate_max_slope(slope, diameter):
            raise ValueError("Slope exceeds maximum allowed value")
        if not validate_min_slope(slope, filling, diameter):
            raise ValueError("Slope is too small")

        area = HydraulicCalculationsService.calc_area(filling, diameter)
        velocity = HydraulicCalculationsService.calc_velocity(filling, diameter, slope)
        flow = area * velocity  # Removed the 1000.0 multiplier since we're now using m³/s instead of dm³/s

        return flow

    @staticmethod
    def calc_filling(q: float, diameter: float, slope: float) -> float:
        """
        Iteratively approximates the filling height (m) needed to achieve flow q [m³/s].

        Args:
            q (float): Flow rate [m³/s].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Filling height [m].

        Raises:
            ValueError: If filling exceeds diameter without achieving desired flow.
        """
        if not (common_diameters[0] <= diameter <= common_diameters[-1]):
            raise ValueError("Diameter out of range for common diameters")
        if q < 0:
            raise ValueError("Invalid flow rate, must be positive")
        if q == 0:
            return 0.0

        filling = 0.0
        step = 0.001
        max_iter = 10000

        for _ in range(max_iter):
            if filling > diameter:
                break
            flow = HydraulicCalculationsService.calc_flow(filling, diameter, slope)  # już [m³/s]
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
        """
        Calculates the filling height of conduits in meters [m].

        Filling is calculated as the product of maximum depth percentage (MaxDPerc)
        and the conduit diameter (Geom1). The result represents the actual water
        level inside the pipe in meters.

        This method adds a 'Filling' column to the conduits dataframe (self.dfc).
        """
        if self.dfc is not None:
            self.dfc["Filling"] = self.dfc["MaxDPerc"] * self.dfc["Geom1"]

    def filling_is_valid(self) -> None:
        """
        Validates if the filling height is within acceptable limits for each conduit.

        This method checks if the filling height (in meters [m]) is valid for the given
        pipe diameter using the validate_filling function. The result is stored as a
        binary value (0 or 1) in the 'ValMaxFill' column, where:
        - 1 indicates the filling is valid (within acceptable limits)
        - 0 indicates the filling exceeds the maximum allowed value

        The validation is based on the maximum filling ratio defined in the system
        (typically 80-90% of pipe diameter).
        """
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
        """
        Determine the smallest standard conduit diameter that can safely convey
        the design flow (MaxQ) while meeting the maximum filling constraint.

        A new column *MinDiameter* is added to self.dfc.

        Workflow
        --------
        1. For rows where filling already exceeds limits (ValMaxFill == 0),
            increase the diameter to the next larger standard size that can handle the flow.
        2. For rows where filling is within limits (ValMaxFill == 1):
            a. Find the index of the current diameter in *common_diameters*.
                If it is not present, leave the diameter unchanged.
            b. Check if current filling exceeds maximum allowed filling (0.827 * diameter).
                If so, increase the diameter to the next larger standard size.
            c. Iterate over smaller standard diameters (descending).
                • For each candidate compute the filling height required to pass MaxQ.
                • If validate_filling succeeds, update *best_diam* and continue.
                • If it fails due to slope constraints, try with increased slope.
                • If it fails due to other reasons, break the loop.
        """
        import logging

        # Configure logging to show detailed information
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if self.dfc is None:
            return

        common_diameters_sorted: List[float] = sorted(common_diameters)
        diam_index_map: Dict[float, int] = {d: i for i, d in enumerate(common_diameters_sorted)}

        def get_max_filling_height(diameter: float) -> float:
            """Calculate maximum allowed filling height for a given diameter."""
            filling_value = max_filling(diameter) if callable(max_filling) else max_filling
            return filling_value * diameter if filling_value <= 1 else (filling_value / 100) * diameter

        def find_larger_diameter(current_diam: float, required_filling: float) -> float:
            """Find the next larger diameter that can handle the required filling."""
            idx = diam_index_map.get(current_diam, -1)
            if idx != -1 and idx + 1 < len(common_diameters_sorted):
                for larger_diam in common_diameters_sorted[idx + 1 :]:
                    max_filling_height = get_max_filling_height(larger_diam)
                    if required_filling <= max_filling_height:
                        logger.info(f"Increasing diameter from {current_diam}m to {larger_diam}m")
                        return larger_diam
            # If no suitable larger diameter found, keep current
            logger.warning(f"Could not find a suitable larger diameter for filling: {required_filling}")
            return current_diam

        def handle_excessive_filling(row: pd.Series) -> float:
            """Handle case where filling already exceeds limits."""
            logger.info(f"ROW {row.name}: Filling already exceeds limits - need to increase diameter")
            current_diam = row["Geom1"]
            current_filling = row.get("Filling", 0.0)
            new_diam = find_larger_diameter(current_diam, current_filling)
            return new_diam

        def try_smaller_diameters(row: pd.Series, current_diam: float, idx: int, max_q: float, original_slope: float) -> float:
            """Try to find a smaller diameter that can still handle the flow."""
            best_diam = current_diam

            for candidate in reversed(common_diameters_sorted[:idx]):
                # Try different slope multipliers
                slope_multipliers = [1.0, 2.0, 3.0, 5.0]
                diameter_works = False

                for multiplier in slope_multipliers:
                    adjusted_slope = original_slope * multiplier

                    try:
                        fill_height = HydraulicCalculationsService.calc_filling(max_q, candidate, adjusted_slope)

                        if validate_filling(fill_height, candidate):
                            best_diam = candidate
                            diameter_works = True
                            break  # Exit multiplier loop
                        else:
                            break  # This diameter doesn't work even with this slope
                    except ValueError as e:
                        error_msg = str(e)
                        if "Slope is too small" in error_msg and multiplier < slope_multipliers[-1]:
                            # Try next multiplier
                            continue
                        else:
                            # Other error or reached max multiplier
                            break

                if diameter_works:
                    # Continue checking even smaller diameters
                    continue
                else:
                    # This diameter doesn't work with any slope, stop checking smaller ones
                    break

            return best_diam

        def handle_small_filling_ratio(row: pd.Series, current_diam: float, current_filling: float) -> float:
            """Handle the case where filling ratio is very small."""
            if current_filling / current_diam < 0.3:
                # Target 50% filling for better hydraulic efficiency
                target_diameter = current_filling / 0.5

                for candidate in common_diameters_sorted:
                    if candidate >= target_diameter and candidate < current_diam:
                        return candidate

            return current_diam

        def find_min_diam(row: pd.Series) -> float:
            # Step 1: Handle rows where filling already exceeds limits
            if row.get("ValMaxFill", 0) != 1:
                return handle_excessive_filling(row)

            # Get common parameters
            current_diam: float = row["Geom1"]
            current_filling: float = row.get("Filling", 0.0)
            max_q: float = row.get("MaxQ", 0.0)

            # Step 2: Check if current filling exceeds maximum allowed
            max_allowed_filling_height = get_max_filling_height(current_diam)

            if current_filling > max_allowed_filling_height:
                logger.info(f"ROW {row.name}: Current filling exceeds max allowed - need to increase diameter")
                return find_larger_diameter(current_diam, current_filling)

            # Step 3: Handle very small flows
            if max_q < 0.01:  # Very small flow (below 10 l/s)
                for candidate in common_diameters_sorted:
                    if candidate >= 0.3:  # Don't go below 300mm for practical reasons
                        return candidate
                return common_diameters_sorted[0]  # Return smallest available diameter

            # Step 4: Find current diameter index
            idx: int = diam_index_map.get(current_diam, -1)
            if idx == -1:  # Non-standard diameter - leave as is
                return current_diam

            # Step 5: Check if current diameter is sufficient based on calculated filling
            original_slope: float = row.get("SlopeFtPerFt", 0.0)
            best_diam: float = current_diam

            try:
                current_fill_height = HydraulicCalculationsService.calc_filling(max_q, current_diam, original_slope)

                if not validate_filling(current_fill_height, current_diam):
                    logger.info(f"ROW {row.name}: Current diameter is insufficient for calculated flow")

                    # Find larger diameter that works
                    for larger_diam in common_diameters_sorted[idx + 1 :] if idx + 1 < len(common_diameters_sorted) else []:
                        try:
                            larger_fill_height = HydraulicCalculationsService.calc_filling(max_q, larger_diam, original_slope)

                            if validate_filling(larger_fill_height, larger_diam):
                                logger.info(f"ROW {row.name}: Increasing diameter from {current_diam}m to {larger_diam}m")
                                return larger_diam
                        except ValueError:
                            continue
            except ValueError:
                pass  # Continue with finding smaller diameter

            # Step 6: Try to find smaller diameters that still meet requirements
            best_diam = try_smaller_diameters(row, current_diam, idx, max_q, original_slope)

            # Step 7: For cases with very low filling, try an alternative approach
            if best_diam == current_diam:
                best_diam = handle_small_filling_ratio(row, current_diam, current_filling)

            return best_diam

        # Apply to every row
        self.dfc["MinDiameter"] = self.dfc.apply(find_min_diam, axis=1)

    def diameter_features(self) -> None:
        """
        Calculates three features related to pipe diameter optimization:

        - isMinDiameter: 1 if current diameter equals calculated minimum diameter, 0 otherwise
        - IncreaseDia: 1 if current diameter is too small and should be increased, 0 otherwise
        - ReduceDia: 1 if current diameter is larger than necessary and could be reduced, 0 otherwise

        These features provide more detailed information for the neural network model
        about the relationship between current diameter (Geom1) and calculated
        minimum diameter (MinDiameter).
        """
        if self.dfc is not None:
            # Check if current diameter equals calculated minimum diameter
            self.dfc["isMinDiameter"] = np.where(self.dfc["Geom1"] == self.dfc["MinDiameter"], 1, 0)

            # Check if current diameter is too small and should be increased
            self.dfc["IncreaseDia"] = np.where(self.dfc["Geom1"] < self.dfc["MinDiameter"], 1, 0)

            # Check if current diameter is larger than necessary and could be reduced
            self.dfc["ReduceDia"] = np.where(self.dfc["Geom1"] > self.dfc["MinDiameter"], 1, 0)

    def normalize_roughness(self) -> None:
        """
        Normalizes the Roughness coefficient (Manning's n) in the dataframe.

        The Roughness coefficient is normalized to a range of [0, 1] based on typical
        values for different pipe materials used in stormwater drainage systems.

        Typical Manning's n values for different pipe materials:
        - PVC/Plastic pipes: 0.009 - 0.013 (Smooth surfaces)
        - Concrete pipes: 0.011 - 0.015 (Moderately smooth)
        - Cast iron pipes: 0.012 - 0.016 (Moderately rough)
        - Steel pipes: 0.010 - 0.014 (Varies based on coating)
        - Brick channels: 0.012 - 0.018 (Rough surfaces)
        - Stone channels: 0.013 - 0.020 (Very rough surfaces)

        The normalization uses the formula:
        Roughness_normalized = (Roughness - MIN_ROUGHNESS) / (MAX_ROUGHNESS - MIN_ROUGHNESS)

        Where:
        - MIN_ROUGHNESS = 0.009 (smoothest plastic pipes)
        - MAX_ROUGHNESS = 0.020 (roughest stone channels)

        Note:
        Values outside the range [0.009, 0.020] will be clipped to ensure
        the normalized values stay within [0, 1].
        """
        if self.dfc is None:
            return

        min_roughness = 0.009  # Smoothest plastic pipes
        max_roughness = 0.020  # Roughest stone channels

        roughness_clipped = self.dfc["Roughness"].clip(min_roughness, max_roughness)
        self.dfc["NRoughness"] = (roughness_clipped - min_roughness) / (max_roughness - min_roughness)

    def normalize_max_velocity(self) -> None:
        """
        Normalizes the MaxV (maximum velocity) values in the dataframe.

        The maximum velocity is normalized to a range of [0, 1] based on
        the maximum allowed velocity value from max_velocity_value.

        The normalization uses the formula:
        NMaxV = MaxV / max_velocity_value

        Where:
        - MaxV = Original maximum velocity value
        - max_velocity_value = Maximum allowed velocity (typically 5.0 m/s)

        Note:
        Values greater than max_velocity_value will be clipped to ensure
        the normalized values stay within [0, 1].
        """
        if self.dfc is None:
            return

        max_v_clipped = self.dfc["MaxV"].clip(0, max_velocity_value)
        self.dfc["NMaxV"] = max_v_clipped / max_velocity_value

    def normalize_depth(self) -> None:
        """
        Normalizes depth values for inlet and outlet nodes to range [0, 1].

        Normalization is based on the valid depth range for pipes:
        - Minimum valid depth: frost_zone + pipe diameter (frost protection + pipe size)
        - Maximum valid depth: max_depth_value (maximum allowed installation depth)

        Values outside the valid range are clipped before normalization.
        Rows where min equals max (zero range) are set to 0 to avoid division by zero.
        """
        if self.dfc is None:
            return

        min_depth = self.frost_zone + self.dfc["Geom1"]
        inlet_depth_clipped = self.dfc["InletMaxDepth"].clip(lower=min_depth, upper=max_depth_value)
        outlet_depth_clipped = self.dfc["OutletMaxDepth"].clip(lower=min_depth, upper=max_depth_value)
        depth_range = max_depth_value - min_depth
        valid_range = depth_range > 0
        self.dfc["NInletDepth"] = 0.0
        self.dfc["NOutletDepth"] = 0.0
        self.dfc.loc[valid_range, "NInletDepth"] = (
            inlet_depth_clipped.loc[valid_range] - min_depth.loc[valid_range]
        ) / depth_range.loc[valid_range]
        self.dfc.loc[valid_range, "NOutletDepth"] = (
            outlet_depth_clipped.loc[valid_range] - min_depth.loc[valid_range]
        ) / depth_range.loc[valid_range]

    def normalize_filling(self) -> None:
        """
        Normalizes the Filling values to range [0, 1].

        Normalization is based on the ratio of filling to maximum effective filling
        calculated as 0.827 * pipe diameter (Geom1), according to Colebrook-White formula.
        Values exceeding this threshold indicate hydraulic overload.

        According to hydraulic principles, a circular pipe reaches its maximum flow capacity
        at approximately 82.7% of its diameter, not at full diameter.
        """
        if self.dfc is None:
            return

        max_effective_filling = 0.827 * self.dfc["Geom1"]
        self.dfc["NFilling"] = self.dfc["Filling"] / max_effective_filling

    def normalize_max_q(self) -> None:
        """
        Normalizes the MaxQ (maximum flow) values in the dataframe.

        The normalization is based on the theoretical maximum flow capacity
        for each pipe, calculated using pipe diameter and slope.
        """
        if self.dfc is None:
            return

        # Calculate theoretical max flow capacity for each pipe
        # Using a simplified approach based on pipe diameter and slope
        # Q_max = k * D^2.5 * S^0.5 where D is diameter and S is slope
        k = 0.312  # Constant for circular pipes with Manning's n=0.013

        # Calculate theoretical max flow in m³/s
        theoretical_max_q = k * (self.dfc["Geom1"] ** 2.5) * (self.dfc["SlopePerMile"] ** 0.5)

        # Normalize actual MaxQ by theoretical max flow
        # Clip to [0, 1] range to handle cases where actual flow exceeds theoretical max
        self.dfc["NMaxQ"] = (self.dfc["MaxQ"] / theoretical_max_q).clip(0, 1)

    def normalize_ground_cover(self) -> None:
        """
        Normalizes ground cover values (InletGroundCover and OutletGroundCover) to range [0, 1].

        Normalization is based on the valid ground cover range for pipes:
        - Minimum valid cover: self.frost_zone (frost protection)
        - Maximum valid cover: max_depth_value - Geom1 (maximum installation depth minus pipe diameter)

        Values outside the valid range are clipped before normalization.
        Rows where min equals max (zero range) are set to 0 to avoid division by zero.

        A ground cover value of 0 means minimum acceptable cover (frost_zone).
        A value of 1 means maximum possible cover (max_depth_value - pipe diameter).
        """
        if self.dfc is None:
            return

        if "InletGroundCover" not in self.dfc.columns or "OutletGroundCover" not in self.dfc.columns:
            self.ground_cover()

        min_cover = self.frost_zone
        max_cover = max_depth_value - self.dfc["Geom1"]

        # Clip values to valid range
        inlet_cover_clipped = self.dfc["InletGroundCover"].clip(lower=min_cover, upper=max_cover)
        outlet_cover_clipped = self.dfc["OutletGroundCover"].clip(lower=min_cover, upper=max_cover)

        # Calculate range (avoiding division by zero)
        cover_range = max_cover - min_cover
        valid_range = cover_range > 0

        # Initialize with zeros
        self.dfc["NInletGroundCover"] = 0.0
        self.dfc["NOutletGroundCover"] = 0.0

        # Normalize only where range is valid
        self.dfc.loc[valid_range, "NInletGroundCover"] = (inlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]
        self.dfc.loc[valid_range, "NOutletGroundCover"] = (outlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]

    def normalize_slope(self) -> None:
        """
        Normalizes the slope values in the dataframe to a range [0, 1] based on both
        minimum required slope and maximum allowable slope.

        The normalization maps the slope values to a range where:
        - 0 represents a slope at or below the minimum required slope
        - 0.5 represents an optimal slope (midpoint between min and max)
        - 1 represents a slope at or above the maximum allowable slope

        This ensures that slopes are evaluated relative to their hydraulic requirements
        based on pipe diameter and filling.
        """
        if self.dfc is None or self.dfc.empty:
            return

        # Calculate minimum required slope for each pipe based on its filling and diameter
        self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: min_slope(r["Filling"], r["Geom1"]), axis=1)

        # Calculate maximum allowable slope for each pipe based on its diameter
        self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: max_slope(r["Geom1"]), axis=1)

        # Normalize slope to range [0, 1] where:
        # 0 = at or below minimum required slope
        # 1 = at or above maximum allowable slope
        def normalize(row):
            min_slope_val = row["MinRequiredSlope"]
            max_slope_val = row["MaxAllowableSlope"]
            actual_slope = row["SlopePerMile"]

            # Handle edge cases
            if min_slope_val >= max_slope_val:  # If min and max are too close
                return 0.5 if actual_slope >= min_slope_val else 0.0

            # Linear normalization between min and max
            if actual_slope <= min_slope_val:
                return 0.0
            elif actual_slope >= max_slope_val:
                return 1.0
            else:
                # Map to range [0, 1]
                return (actual_slope - min_slope_val) / (max_slope_val - min_slope_val)

        self.dfc["NSlope"] = self.dfc.apply(normalize, axis=1)

    def slope_increase(self) -> None:
        """
        Identifies pipes that need slope increase because their current slope is too small.

        This method adds a binary indicator column 'SlopeIncrease' to the dataframe where:
        - 1 indicates the pipe needs slope increase (current slope < minimum required slope)
        - 0 indicates the pipe does not need slope increase

        The minimum required slope is calculated based on the pipe's filling height and diameter
        using the min_slope function.
        """
        if self.dfc is None:
            return

        # Ensure MinRequiredSlope is calculated
        if "MinRequiredSlope" not in self.dfc.columns:
            self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: min_slope(r["Filling"], r["Geom1"]), axis=1)

        # Check if current slope is less than minimum required slope
        self.dfc["IncreaseSlope"] = np.where(self.dfc["SlopePerMile"] < self.dfc["MinRequiredSlope"], 1, 0)

    def slope_reduction(self) -> None:
        """
        Identifies pipes that need slope reduction because their current slope is too large.

        This method adds a binary indicator column 'SlopeReduction' to the dataframe where:
        - 1 indicates the pipe needs slope reduction (current slope > maximum allowable slope)
        - 0 indicates the pipe does not need slope reduction

        The maximum allowable slope is calculated based on the pipe's diameter
        using the max_slope function.
        """
        if self.dfc is None or self.dfc.empty:
            return

        # Ensure MaxAllowableSlope is calculated
        if "MaxAllowableSlope" not in self.dfc.columns:
            self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: max_slope(r["Geom1"]), axis=1)

        # Check if current slope is greater than maximum allowable slope
        self.dfc["ReduceSlope"] = np.where(self.dfc["SlopePerMile"] > self.dfc["MaxAllowableSlope"], 1, 0)

    def conduits_subcatchment_info(self) -> None:
        """
        Maps subcatchment information to conduits based on their inlet and outlet nodes.

        This method assigns subcatchment ID and category to conduits based on:
        1. Outlet node's subcatchment information if available
        2. Inlet node's subcatchment information if outlet node has no subcatchment

        If neither node has subcatchment information, conduit's subcatchment fields remain empty.
        """
        if self.dfc is None or self.dfn is None:
            return

        # Check if nodes dataframe has the required subcatchment columns
        if "Subcatchment" not in self.dfn.columns or "SbcCategory" not in self.dfn.columns:
            return

        # Initialize conduit subcatchment columns
        self.dfc["Subcatchment"] = "-"
        self.dfc["SbcCategory"] = "-"

        # Map node subcatchment info to conduits
        for idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            # First try outlet node for subcatchment info
            if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
            # If no info on outlet node, try inlet node
            elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]

    def propagate_subcatchment_info(self) -> None:
        """
        Propagates subcatchment information through the network for nodes and conduits
        that don't have directly connected subcatchments.

        If a node doesn't have subcatchment info, it inherits from its inlet nodes.
        If a conduit doesn't have subcatchment info, it inherits from its nodes.

        This method should be called after conduits_subcatchment_info().
        """
        if self.dfc is None or self.dfn is None:
            return

        # First, gather all connections (which node is connected to which)
        node_connections = {}  # Dictionary mapping outlet_node -> list of inlet_nodes

        for idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            if outlet_node not in node_connections:
                node_connections[outlet_node] = []

            if inlet_node:
                node_connections[outlet_node].append(inlet_node)

        # Propagate subcatchment info through the network
        # Multiple iterations to ensure info propagates through the entire network
        for _ in range(5):  # Arbitrary number of iterations
            changes_made = False

            # Update nodes based on their inlet nodes
            for outlet_node, inlet_nodes in node_connections.items():
                if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] == "-":
                    # Try to find any inlet node with subcatchment info
                    for inlet_node in inlet_nodes:
                        if inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                            self.dfn.at[outlet_node, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                            self.dfn.at[outlet_node, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                            changes_made = True
                            break

            # Update conduits based on their nodes
            for idx, row in self.dfc.iterrows():
                if self.dfc.at[idx, "Subcatchment"] == "-":
                    inlet_node = row.get("InletNode")
                    outlet_node = row.get("OutletNode")

                    # First try outlet node
                    if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
                        changes_made = True
                    # Then try inlet node
                    elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                        changes_made = True

            # If no changes were made in this iteration, we've reached a stable state
            if not changes_made:
                break

    def encode_sbc_category(self) -> None:
        """One-hot encodes the SbcCategory column for use in machine learning models.

        Creates columns for all possible subcatchment categories, not just those present in the data,
        ensuring consistent feature sets across different datasets.

        This method adds one-hot encoded columns directly to the conduits dataframe (self.dfc).
        Each column is named 'SbcCategory_{category}' and contains integer values (0 or 1).
        """
        if self.dfc is None or len(self.dfc) == 0:
            return

        if "SbcCategory" not in self.dfc.columns:
            return

        all_categories = [
            "compact_urban_development",
            "urban",
            "loose_urban_development",
            "wooded_area",
            "grassy",
            "loose_soil",
            "steep_area",
        ]

        encoded_categories = pd.get_dummies(self.dfc["SbcCategory"], drop_first=False)

        for category in all_categories:
            col_name = f"{category}"
            if col_name not in encoded_categories.columns:
                encoded_categories[col_name] = 0
            else:
                encoded_categories[col_name] = encoded_categories[col_name].astype(int)

        for col in encoded_categories.columns:
            self.dfc[col] = encoded_categories[col]


class SubcatchmentFeatureEngineeringService:
    def __init__(self, dfs: pd.DataFrame):
        """Initialize the service with subcatchment dataframe.

        Args:
            dfs (pd.DataFrame): DataFrame containing subcatchment data.
        """
        self.dfs = dfs

    def encode_category_column(self, category_column: str = "category") -> pd.DataFrame:
        """One-hot encodes the category column for use in machine learning models.

        Creates columns for all possible categories, not just those present in the data,
        ensuring consistent feature sets across different datasets.

        Args:
            category_column (str, optional): Name of the category column to encode.
                                            Defaults to "category".

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded category columns added
        """
        if self.dfs is None or len(self.dfs) == 0:
            return self.dfs

        if category_column not in self.dfs.columns:
            raise ValueError(f"Column '{category_column}' not found in dataframe")

        all_categories = [
            "compact_urban_development",
            "urban",
            "loose_urban_development",
            "wooded_area",
            "grassy",
            "loose_soil",
            "steep_area",
        ]

        encoded_categories = pd.get_dummies(self.dfs[category_column], prefix=None, drop_first=False)

        for category in all_categories:
            if category not in encoded_categories.columns:
                encoded_categories[category] = 0
            else:
                encoded_categories[category] = encoded_categories[category].astype(int)

        result_df = pd.concat([self.dfs, encoded_categories], axis=1)
        return result_df

    def subcatchments_classify(self, categories: bool = True) -> None:
        """Classifies subcatchments using the classifier model (ANN).

        Uses a pre-trained neural network to classify subcatchments into different
        categories based on their physical and hydrological characteristics.

        Args:
            categories (bool, optional): If True, adds category labels as strings.
                                        If False, adds numeric category codes.
                                        Defaults to True.

        Returns:
            None: Adds 'category' column to the dataframe in-place.
        """
        if self.dfs is None or len(self.dfs) == 0:
            return

        # Required columns for the model
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

        # Check if all required columns exist
        missing_cols = [col for col in cols if col not in self.dfs.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare data for prediction
        df = self.dfs[cols].copy()
        df["TotalPrecip"] = pd.to_numeric(df["TotalPrecip"], errors="coerce").fillna(0)

        # Make predictions
        preds = classifier.predict(df)
        preds_cls = preds.argmax(axis=-1)

        # Add category column
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
    Maps subcatchment information to nodes based on the 'Outlet' column from dfs.
    """

    def __init__(self, dfn: pd.DataFrame, dfs: pd.DataFrame):
        self.dfn = dfn
        self.dfs = dfs

    def nodes_subcatchment_name(self) -> None:
        """
        Legacy method that calls nodes_subcatchment_info() for backward compatibility.
        """
        self.nodes_subcatchment_info()

    def nodes_subcatchment_info(self) -> None:
        """
        Maps subcatchment information (ID and category) to nodes based on the 'Outlet' column from dfs.

        Adds two columns to nodes dataframe:
        - Subcatchment: The ID of the connected subcatchment
        - SbcCategory: The category of the connected subcatchment
        """
        if self.dfs is None or self.dfn is None:
            return

        # Initialize columns with default values
        self.dfn["Subcatchment"] = "-"
        self.dfn["SbcCategory"] = "-"

        # Create mappings for subcatchment name and category if columns exist
        if "Outlet" in self.dfs.columns:
            # Get subcatchment ID mapping
            name_mapping = self.dfs.reset_index().set_index("Outlet")["Name"].to_dict()

            # Get subcatchment category mapping if category column exists
            category_mapping = {}
            if "category" in self.dfs.columns:
                category_mapping = self.dfs.reset_index().set_index("Outlet")["category"].to_dict()

            # Assign values to nodes that are outlets for subcatchments
            for node_id in self.dfn.index:
                if node_id in name_mapping:
                    self.dfn.at[node_id, "Subcatchment"] = name_mapping[node_id]
                    if node_id in category_mapping:
                        self.dfn.at[node_id, "SbcCategory"] = category_mapping[node_id]


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

        # encoded_subcatchments = self.subcatchment_service.encode_category_column(category_column="category")
        # # Update the subcatchment dataframe with the encoded version if needed
        # if encoded_subcatchments is not None:
        #     self.dfs = encoded_subcatchments

        # Nodes - zmienione z nodes_subcatchment_name() na nodes_subcatchment_info()
        self.node_service.nodes_subcatchment_info()

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
        self.conduit_service.slope_increase()
        self.conduit_service.slope_reduction()
        self.conduit_service.encode_sbc_category()

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
