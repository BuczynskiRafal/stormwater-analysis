import logging
from pipes.round import (
    check_dimensions,
    max_filling,
    min_slope,
    max_slopes,
    max_velocity_value,
    min_velocity_value,
    max_depth_value,
)

logger = logging.getLogger(__name__)


def validate_filling(filling: float, diameter: float) -> bool:
    """
    Check that the maximum filling is not exceeded.

    Args:
        filling (int, float): pipe filling height [m].
        diameter (int, float): pipe diameter [m].

    Return:
        bool: given filling value is lower than the maximum filling.
    """
    return filling <= max_filling(diameter)


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
    return velocity >= min_velocity_value


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


def validate_min_slope(
    slope: float, filling: float, diameter: float, theta: float = 1.5, g: float = 9.81
) -> bool:
    """
    Check that the minimum slope is not exceeded.
    For hydraulic calculations of sewers and rainwater collectors,
    it is recommended to adopt a self-cleaning slope
    of not less than 0.7 m - m/s to prevent solids from settling at the bottom of the sewer.
    source: PN-EN 752:2008 Zewnętrzne systemy kanalizacyjne - in polish

    Args:
        slope (int, float): flow slope in the sewer [m/s].
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        theta (float, optional): theta value. Defaults to 1.5, shear stress [Pa].
        g (float, optional): specific gravity of liquid (water/wastewater) [N/m3].

    Return:
        bool: given slope value is higher than the minimum velocity.
    """
    return slope >= min_slope(filling, diameter, theta, g)


def validate_max_slope(slope: float, diameter: float) -> bool:
    """
    Check that the maximum slope is not exceeded.
    """
    if check_slope(slope) and check_dimensions(diameter, diameter):
        return slope <= max_slopes.get(str(diameter))
    return False
