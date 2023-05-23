import logging
import math
from typing import Union

import matplotlib.pyplot
import matplotlib.pyplot as plt
from numpy import cos, linspace, pi, sin

from stormwater_analysis.utils.lazy_object import LazyObject

logger = logging.getLogger(__name__)


def check_dimensions(filling: float, diameter: float) -> bool:
    """
    Check if the given filling and diameter values are valid.

    Args:
        filling (float, int): the height of the filling in the pipe, in meters.
        diameter (float, int): the diameter of the pipe, in meters.

    Returns:
        bool: True if the values are valid, False otherwise.

    Raises:
        TypeError: If either the filling or the
        diameter is not an int or float.
        ValueError: If either the filling
        or the diameter is not between 0.2 and 2.0 meters.

    """
    if not isinstance(filling, (int, float)):
        raise TypeError("Filling must be an int or float")
    if not isinstance(diameter, (int, float)):
        raise TypeError("Diameter must be an int or float")
    if filling > diameter:
        raise ValueError("Filling must be less than or equal to the diameter")
    if not (0 <= filling <= 2.0 and 0.2 <= diameter <= 2.0):
        raise ValueError(
            """Value out of bounds. Filling must be between 0.2 and 2.0
            meters and diameter must be between 0.2 and 2.0 meters"""
        )
    return True


def max_filling(diameter: float) -> float:
    """
    > Calculates the maximum filling of a rainwater
    drain with a circular cross-section.

    According to a methodology based on the Colebrook-White formula, the total
    capacity of the channel (100%), i.e., with total cross-sectional filling
    (100%), is already achieved at a relative
    filling of 4/D = 0.827 - in circular pipes,
    Kotowski, A., Kaźmierczak, B., & Dancewicz, A. (2010).
    Modelowanie opadów do wymiarowania kanalizacji.
    Polska Akademia Nauk. Instytut Podstawowych Problemów Techniki.

    Args:
        diameter (int, float): the diameter of the rainwater pipe [m].

    Return:
        maximum filling (float): he maximum pipe filling [m].
    """
    if isinstance(diameter, (int, float)):
        if 0.2 <= diameter <= 2.0:
            return 0.827 * diameter
        else:
            raise ValueError("Invalid diameter value.")
    else:
        raise TypeError("Diameter must be an int or float")


def max_velocity() -> float:
    """
    Maximum stormwater flow velocity in the sewer [m/s].

    Return:
         velocity (int): maximum stormwater flow velocity in pipe [m].
    """
    return 5


def min_velocity():
    """
    Minimum stormwater flow velocity in the sewer [m/s].

    Return:
         velocity (int): minimum stormwater flow velocity in pipe [m].
    """
    return 0.7


def max_depth():
    """
    The maximum depth of the sewer pipe [m].

    Return:
         depth (int): maximum depth of the sewer pipe [m].
    """
    return 8


def calc_f(filling: float, diameter: float) -> float:
    """
    Calculate the cross-sectional area of a pipe.
    The cross-sectional area through which the wastewater flows,
    active cross-section f,
    characterized by the filling h and the diameter of the pipe D.
    source: Biedugnis S., “Metody informatyczne w wodociągach i kanalizacji”,
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        area (int, float): cross-sectional
        area of the wetted part of the pipe [m2]
    """
    if check_dimensions(filling, diameter):
        radius = diameter / 2
        chord = math.sqrt((radius**2 - ((filling - radius) ** 2))) * 2
        alpha = math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2))
        if filling > radius:
            return pi * radius**2 - (1 / 2 * (alpha - math.sin(alpha)) * radius**2)
        elif filling == radius:
            return pi * radius**2 / 2
        elif filling == diameter:
            return pi * radius**2
        else:
            return 1 / 2 * (alpha - math.sin(alpha)) * radius**2


def calc_u(filling: float, diameter: float) -> float:
    """
    Calculate the circumference of a wetted part of pipe.
    source: Biedugnis S., “Metody informatyczne w wodociągach i kanalizacji”,
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        circumference (int, float): circumference of a wetted part of pipe
    """
    if check_dimensions(filling, diameter):
        radius = diameter / 2
        chord = math.sqrt((radius**2 - (filling - radius) ** 2)) * 2
        alpha = math.degrees(math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2)))
        if filling > radius:
            return 2 * math.pi * radius - (alpha / 360 * 2 * math.pi * radius)
        return alpha / 360 * 2 * math.pi * radius


def calc_rh(filling: float, diameter: float) -> float:
    """
    Calculate the hydraulic radius Rh, i.e. the ratio of the cross-section f
    to the contact length of the sewage with the sewer wall,
    called the wetted circuit U.
    source: Biedugnis S., “Metody informatyczne w wodociągach i kanalizacji”,
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        Rh (int, float): hydraulic radius [m]
    """
    if check_dimensions(filling, diameter):
        try:
            return calc_f(filling, diameter) / calc_u(filling, diameter)
        except ZeroDivisionError:
            return 0


def calc_velocity(
    filling: Union[float, int],
    diameter: Union[float, int],
    slope: Union[float, int],
) -> Union[float, int]:
    """
    Calculate the speed of the sewage flow in the sewer.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        slope (int, float): slope of the sewer bottom [‰]

    Return:
        v (int, float): sewage flow velocity in the sewer [m/s]
    """
    if check_dimensions(filling, diameter):
        slope = slope / 1000
        return 1 / 0.013 * calc_rh(filling, diameter) ** (2 / 3) * (slope**0.5)


def min_slope(filling: float, diameter: float, theta: float = 1.5, g: float = 9.81) -> float:
    """
    Get the minimal slope for sewer pipe.
    If the pipe  filling is greater than 0.3,
    then the minimum slope is 1/d, otherwise it's 0.25/rh

    source: Suligowski : Samooczyszczanie przewodów kanalizacyjnych.
    Instal 2010, nr 2, s. 48-53. - in polish
    source: https://seidel-przywecki.eu/2021/06/04/obliczenia-
    hydraulicznych-kanalow-sciekowych-i-deszczowych/

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        theta (float, optional): theta value.
        Defaults to 1.5, shear stress [Pa].
        g (float, optional): specific gravity
        of liquid (water/wastewater) [N/m3].

    Return:
        slope (int, float): The minimum slope of the channel [‰]
    """
    if check_dimensions(filling, diameter):
        return 4 * (theta / g) * ((diameter / 4) / calc_rh(filling, diameter)) * (1 / diameter)


def max_slope(diameter: Union[float, int]) -> Union[float, int]:
    """
    Calculates the maximum slope for a given pipe diameter.

    The function starts with an initial slope based
    on the diameter, and then iteratively
    adjusts the slope until the calculated velocity
    matches the maximum velocity for the
    pipeline.

    Args:
        diameter (Union[float, int]): pipe diameter [m].

    Returns:
        Union[float, int]: The maximum slope that can be achieved for the pipe.
    """
    if check_dimensions(diameter, diameter):
        start_slope = min_slope(diameter, diameter)
        slope = start_slope
        v_max = max_velocity()
        v_clc = 0.0
        while round(v_clc, 2) != float(v_max):
            v_clc = calc_velocity(diameter, diameter, slope)
            if v_clc < v_max:
                slope += start_slope
            else:
                start_slope = start_slope / 2
                slope -= start_slope
        return slope


def draw_pipe_section(filling: float, diameter: float, max_filling: Union[float, None] = None) -> matplotlib.pyplot:
    """
    Plot a pipe section with a given diameter and filling height.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        max_filling (int, flow): The maximum filling of the pipe.
        If the pipe is filled above this level, the wetted part
        of the pipe will be drawn in red
    """
    if max_filling is None:
        max_filling = diameter

    radius = diameter / 2
    # draw center point  - 0, 0
    plt.plot(0, 0, color="black", marker="o")
    plt.gca().annotate(
        "O (0, 0)",
        xy=(0 + radius / 10, 0 + radius / 10),
        xycoords="data",
        fontsize=12,
    )
    plt.xlim(-radius - 0.05, radius + 0.05)
    plt.ylim(-radius, radius + 0.07)
    plt.gca().set_aspect("equal")

    # draw circle
    angels = linspace(0 * pi, 2 * pi, 100)
    xs = radius * cos(angels)
    ys = radius * sin(angels)
    # add circle
    plt.plot(xs, ys, color="brown", label=f"Pipe: DN {diameter} [m]")

    # draw diameter
    plt.plot(radius, 0, marker="o", color="blue")
    plt.plot(-radius, 0, marker="o", color="blue")
    plt.plot([radius, -radius], [0, 0])
    # annotation to diameter
    plt.gca().annotate(
        f"Diameter={diameter}",
        xy=(radius / 8, -radius / 5),
        xycoords="data",
        fontsize=12,
    )

    # draw level of water
    plt.plot(0, -radius, marker="o", color="purple")
    plt.plot(0, filling - radius, marker="o", color="purple")
    plt.plot(
        [0, 0],
        [-radius, filling - radius],
        color="purple",
        label=f"Pipe filling height: {filling} [m]",
    )
    plt.gca().annotate(
        f"Water lvl={filling}",
        xy=(radius / 2, filling - radius + 0.01),
        xycoords="data",
        fontsize=12,
    )

    # Draw arc as created by water level
    chord = math.sqrt((radius**2 - ((filling - radius) ** 2))) * 2
    alpha = math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2))

    if filling > max_filling:
        color = "red"
    else:
        color = "blue"
    # Create arc
    if filling <= radius:
        diff = math.radians(180) - alpha
        arc_angles = linspace(diff / 2, alpha + diff / 2, 20)
    else:
        diff = math.radians(180) - alpha
        arc_angles = linspace(-diff / 2, alpha + diff + diff / 2, 100)
    arc_xs = radius * cos(arc_angles)
    arc_ys = radius * sin(arc_angles)
    plt.plot(arc_xs, -arc_ys, color=color, lw=3)
    plt.plot(
        [-arc_xs[0], -arc_xs[-1]],
        [-arc_ys[0], -arc_ys[-1]],
        marker="o",
        color=color,
        lw=3,
        label=f"Wetted part of pipe: {calc_f(filling, diameter):.2f} [m2]",
    )
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()


max_slopes = LazyObject(  # type: ignore
    lambda: {
        "0.2": max_slope(0.2),
        "0.3": max_slope(0.3),
        "0.4": max_slope(0.4),
        "0.5": max_slope(0.5),
        "0.6": max_slope(0.6),
        "0.7": max_slope(0.7),
        "0.8": max_slope(0.8),
        "0.9": max_slope(0.9),
        "1.0": max_slope(1.0),
        "1.2": max_slope(1.2),
        "1.5": max_slope(1.5),
        "2.0": max_slope(2.0),
    }
)


max_velocity_value = max_velocity()

min_velocity_value = min_velocity()  # type: ignore

max_depth_value = max_depth()  # type: ignore
