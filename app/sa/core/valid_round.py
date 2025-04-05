import math

from sa.core.round import common_diameters

logger = logging.getLogger(__name__)


def validate_max_filling(filling: float, diameter: float) -> bool:
    """
    Check that the maximum filling is not exceeded.

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].

    Returns:
        bool: True if filling is within range, False otherwise.
    """
    return filling <= max_filling(diameter)


def validate_filling(filling: float, diameter: float) -> bool:
    """
    Validates the filling height.

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].

    Returns:
        bool: True if filling is valid, else False.
    """
    if not isinstance(diameter, (float, int)) or not isinstance(filling, (float, int)):
        return False
    if 0 > filling or filling >= diameter:
        return False
    if not (common_diameters[0] <= diameter <= common_diameters[-1]):
        return False
    if not validate_max_filling(filling, diameter):
        return False
    return True


def validate_max_velocity(velocity: float) -> bool:
    """
    Check that the maximum velocity is not exceeded.

    Args:
        velocity (int, float): flow velocity in the sewer [m/s].

    Return:
        bool: given velocity value is lower than the maximum velocity.
    """
    return velocity <= max_velocity_value


def validate_min_velocity(velocity: float) -> bool:
    """
    Check that the minimum velocity is not exceeded.
    For hydraulic calculations of sewers and rainwater collectors,
    it is recommended to adopt a self-cleaning velocity
    of not less than 0.7 m - s-1 to prevent solids from settling at the bottom of the sewer.
    source: PN-EN 752:2008 Zewnętrzne systemy kanalizacyjne - in polish

    Args:
        velocity (int, float): flow velocity in the sewer [m/s].

    Return:
        bool: given velocity value is higher than the minimum velocity.
    """
    return velocity >= min_velocity_value  # type: ignore


def check_slope(slope: float) -> bool:
    """
    Check passed value for slope.

    Args:
        slope (float): the slope of a pipe.

    Returns:
        bool: True if the slope is positive, False otherwise.

    Raises:
        TypeError: if slope is not a float or int.
        ValueError: if slope is not positive.
    """
    if not isinstance(slope, (int, float)):
        raise TypeError(f"slope must be a float or int, not {type(slope)}")
    if slope <= 0:
        raise ValueError(f"slope must be positive, not {slope}")
    return True


def validate_min_slope(slope: float, filling: float, diameter: float) -> bool:
    """
    Check that the minimum slope is not exceeded.

    Args:
        slope (float): Slope of the channel [m/m].
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].

    Returns:
        bool: True if slope is above the minimum required, False otherwise.
    """
    # h_over_D = filling / diameter
    # if h_over_D > 0.3:
    #     return slope >= min_slope_hydromechanic(filling, diameter)
    # else:
    return slope >= min_slope_imhoff(diameter)


def min_slope_imhoff(diameter: float) -> float:
    """
    Calculate the minimal slope using Imhoff's formula.

    Imhoff defines the minimum slope as the reciprocal of the pipe diameter in millimeters.

    Args:
        diameter (float): Diameter of the channel [m].

    Returns:
        float: Minimal slope [m/m], calculated as 1 / D[mm].
    """
    return 1.0 / (diameter * 1000)


def min_slope_hydromechanic(filling: float, diameter: float, theta: float = 1.5, g: float = 9.81) -> float:
    """
    Calculate the minimal slope using the hydromechanic criterion.

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].
        theta (float, optional): Shear stress [Pa]. Defaults to 1.5.
        g (float, optional): Acceleration due to gravity [m/s²]. Defaults to 9.81.

    Returns:
        float: Minimal slope [m/m].
    """
    R_b = diameter / 4.0  # Hydraulic radius for full flow [m]
    area = math.pi * (filling**2) / 4.0  # Cross-sectional area for partial flow [m²]
    perimeter = math.pi * filling  # Wetted perimeter for partial flow [m]
    R_hn = area / perimeter if perimeter != 0 else 0  # Hydraulic radius [m]
    if R_hn == 0:
        return float("inf")  # Avoid division by zero
    i_min = 0.612e-3 * (R_hn / R_b) * (1 / diameter)  # [m/m]
    return i_min


def validate_max_slope(slope: float, diameter: float) -> bool:
    """
    Check that the maximum slope is not exceeded.
    """
    if check_slope(slope) and check_dimensions(diameter, diameter):
        return slope <= max_slopes.get(str(diameter))  # type: ignore
    return False
