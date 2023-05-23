import math

import numpy as np
import pytest

from stormwater_analysis.pipes.round import (
    calc_f,
    calc_rh,
    calc_u,
    check_dimensions,
    max_filling,
    max_slope,
    max_slopes,
    max_velocity,
    min_slope,
    min_velocity,
)
from stormwater_analysis.utils.lazy_object import LazyObject


class TestCheckDimensions:
    def test_check_dimensions_filling_and_diameter_equal(self):
        """
        Test check_dimensions function when filling and diameter are equal.
        """
        assert check_dimensions(0.5, 0.5) is True

    def test_check_dimensions_filling_lower_bound(self):
        """
        Test check_dimensions function when filling is at lower bound.
        """
        assert check_dimensions(0.2, 0.2) is True

    def test_check_dimensions_filling_upper_bound(self):
        """
        Test check_dimensions function when filling is at upper bound.
        """
        assert check_dimensions(2.0, 2.0) is True

    def test_check_dimensions_diameter_lower_than_bound(self):
        """
        Test check_dimensions function when diameter
        is lower than the lower bound.
        """
        with pytest.raises(ValueError):
            check_dimensions(0.1, 0.1)

    def test_check_dimensions_diameter_upper_bound(self):
        """
        Checks if the diameter is less than or equal to the upper bound
        """
        assert check_dimensions(0.5, 2.0) is True

    def test_check_dimensions_filling_and_diameter_out_of_range(self):
        """
        Check that the filling and diameter are within the range of 0.1 to 2.0
        """
        with pytest.raises(ValueError):
            check_dimensions(0.1, 2.1)

    def test_check_dimensions_filling_and_diameter_not_specified(self):
        """
        This function checks the dimensions of the pizza to make sure
        that the filling is less than the diameter
        """
        with pytest.raises(TypeError):
            check_dimensions()  # type: ignore

    def test_check_dimensions_filling_and_diameter_with_none(self):
        """
        Check that the filling and diameter are both positive numbers
        """
        with pytest.raises(TypeError):
            check_dimensions(None, None)  # type: ignore

    def test_check_dimensions_valid_values(self):
        """
        Check if the dimensions of a rectangle are valid
        """
        assert check_dimensions(0.5, 1.0) is True

    def test_check_dimensions_filling_greater_than_diameter(self):
        """
        Check that the filling is less than the diameter
        """
        with pytest.raises(ValueError):
            check_dimensions(1.0, 0.5)

    def test_check_dimensions_filling_not_float_or_int(self):
        """
        Check that the dimensions of the rectangle are floats or integers
        """
        with pytest.raises(TypeError):
            check_dimensions("0.5", 1.0)  # type: ignore

    def test_check_dimensions_diameter_not_float_or_int(self):
        """
        Check that the diameter and height are both floats or ints
        """
        with pytest.raises(TypeError):
            check_dimensions(0.5, "1.0")  # type: ignore

    def test_check_dimensions_filling_out_of_range(self):
        """
        Check that the dimensions of the input are valid
        """
        with pytest.raises(ValueError):
            check_dimensions(-1, 1.0)

    def test_check_dimensions_diameter_out_of_range(self):
        """
        Check that the diameter and height of a cylinder are both
        positive and the diameter is no
        greater than 2.1
        """
        with pytest.raises(ValueError):
            check_dimensions(0.5, 2.1)


class TestMaxFilling:
    """Class contain tests the max_filling function."""

    def test_valid_input(self):
        """
        Tests that the function `max_filling` returns the correct value
        for three different inputs
        """
        assert math.isclose(max_filling(0.2), 0.1654, rel_tol=1e-4)
        assert math.isclose(max_filling(1.0), 0.827, rel_tol=1e-4)
        assert math.isclose(max_filling(2.0), 1.654, rel_tol=1e-4)

    def test_max_filling_valid_values(self):
        """
        Check if max_filling takes a diameter and returns
        the maximum filling for that diameter
        """
        for val, dia in zip(
            [0.165, 0.248, 0.331, 0.414, 0.496, 0.579, 0.662, 0.744, 0.827],
            np.arange(0.2, 1, 0.1),
        ):
            assert pytest.approx(val) == round(max_filling(dia), 3)

    def test_invalid_input(self):
        """
        Tests the max_filling function to make sure that it raises
        a TypeError if the input is not a number,
        and a ValueError if the input is not an integer between 1 and 3
        """
        with pytest.raises(TypeError):
            max_filling("invalid")  # type: ignore
        with pytest.raises(ValueError):
            max_filling(0.1)
        with pytest.raises(ValueError):
            max_filling(3.0)


class TestMaxVelocity:
    """
    Class contain tests the max velocity of runoff water in sewer pipe.
    """

    def test_max_velocity(self):
        """
        Test if max velocity returns the maximum velocity of the robot
        """
        assert max_velocity() == 5

    def test_min_velocity(self):
        """
        Test if max velocity returns the minimum velocity of the car.
        """
        assert min_velocity() == 0.7


class TestCalcF:
    """
    Class contain tests the calc_f function.
    """

    def test_calc_f_zero_filling(self):
        """
        Test if calc_f returns 0 when the second argument is 0
        """
        assert calc_f(0, 1) == 0
        assert calc_f(0, 2) == 0
        assert calc_f(0, 0.5) == 0
        assert calc_f(0, 0.2) == 0

    def test_calc_f_filling_equal_diameter(self):
        """
        Test if calc_f calculates the filling factor of a circle with a
        given diameter and a given hole diameter
        """
        assert calc_f(1, 1) == 0.7853981633974483
        assert calc_f(2, 2) == 3.141592653589793
        assert calc_f(0.5, 0.5) == 0.19634954084936207
        assert calc_f(0.1, 0.2) == 0.015707963267948967

    def test_calc_f_filing_equal_radius(self):
        """
        Test if calc_f calculates the angle of the arc of a circle with a
        given radius and chord length
        """
        assert calc_f(0.5, 1) == 0.39269908169872414
        assert calc_f(1, 2) == 1.5707963267948966
        assert calc_f(0.25, 0.5) == 0.09817477042468103
        assert calc_f(0.1, 0.2) == 0.015707963267948967

    def test_calc_f_filing_greater_than_radius(self):
        """
        Test if calc_f function raises a ValueError when the filing_radius
        is greater than the radius
        """
        with pytest.raises(ValueError):
            calc_f(1.5, 1)
            calc_f(2.5, 2)
            calc_f(0.75, 0.5)
            calc_f(0.4, 0.3)

    def test_calc_f_raises_exception(self):
        """
        Test if calc_f() raises a TypeError exception
        if the arguments are not of type int
        """
        with pytest.raises(TypeError):
            calc_f("1", 2)  # type: ignore
        with pytest.raises(TypeError):
            calc_f(1, "2")  # type: ignore
        with pytest.raises(TypeError):
            calc_f(1.5, "2.5")  # type: ignore

    def test_calc_f_filing_greater_than_diameter(self):
        """
        Test if calc_f() raises a ValueError if the filing
        diameter is greater than the pipe diameter
        """
        with pytest.raises(ValueError):
            calc_f(3, 1)
        with pytest.raises(ValueError):
            calc_f(4, 2)
        with pytest.raises(ValueError):
            calc_f(1.1, 1)

    def test_calc_f_filling_less_than_zero(self):
        """
        Test if calc_f raises a `ValueError` when the filling is less than zero
        """
        with pytest.raises(ValueError):
            calc_f(-1, 1)
        with pytest.raises(ValueError):
            calc_f(-0.5, 0.5)


class TestCalcU:
    """
    Class contain tests the calc_u function.
    """

    def test_calc_u_with_valid_input(self):
        """
        Test if calc_u function with valid input
        """
        assert round(calc_u(1, 2), 3) == 3.142

    def test_calc_u_with_zero_filling(self):
        """
        Test if calc_u work with zero filling
        """
        assert round(calc_u(0, 2), 3) == 0

    def test_calc_u_with_filling_above_radius(self):
        """
        Test if calc_u raises a `ValueError` when the filling is greater than 1
        """
        with pytest.raises(ValueError):
            calc_u(2.5, 2)

    def test_calc_u_with_negative_filling(self):
        """
        Test if calc_u raises a `ValueError` when the filling is negative
        """
        with pytest.raises(ValueError):
            calc_u(-1, 2)

    def test_calc_u_with_negative_diameter(self):
        """
        Test if calc_u raises a `ValueError` when the diameter is negative
        """
        with pytest.raises(ValueError):
            calc_u(1, -2)

    def test_calc_u_with_invalid_dimensions(self):
        """
        Test if calc_u raises a ValueError if filling is greater than diameter.
        """
        with pytest.raises(ValueError):
            calc_u(2, 1)

    def test_calc_u_with_invalid_type_for_filling(self):
        """
        Test calc_u function with an invalid type for the `filling`
        parameter
        """
        with pytest.raises(TypeError):
            calc_u("1", 2)  # type: ignore

    def test_calc_u_with_invalid_type_for_diameter(self):
        """
        Test if calc_u raises a `TypeError` when the diameter is not a number.
        """
        with pytest.raises(TypeError):
            calc_u(1, [])  # type: ignore

    def test_calc_u_with_no_dimensions_specified(self):
        """
        Test if calc_u raises a `TypeError` when no dimensions are specified.
        """
        with pytest.raises(TypeError):
            calc_u()  # type: ignore

    def test_calc_u_with_none_for_filling_and_diameter(self):
        """
        Test calc_u with invalid inputs.
        """
        with pytest.raises(TypeError):
            calc_u(None, None)  # type: ignore


class TestCalcRh:
    """
    Class contain tests the calc_rh function.
    """

    @pytest.mark.parametrize(
        "filling, diameter, expected",
        [
            (1.0, 0.5, ValueError),
            (0.3, 0.3, 0.075),
            (0.5, 1.0, 0.25),
            (2.5, 1.0, ValueError),
            (0.1, 0.1, ValueError),
            (3, "two", TypeError),
            (1, 2, 0.5),
        ],
    )
    def test_calc_rh(self, filling, diameter, expected):
        """
        Test the calc_rh function with various input arguments.

        Args:
            filling (float or int): the filling height of the pipe, in meters.
            diameter (float or int): the diameter of the pipe, in meters.
            expected: the expected output of the function
            for the given input arguments.
                      If it is a type, it indicates
                      the expected exception type.

        Raises:
            AssertionError: if the actual output of the function does not match
                             the expected output, or if it does not raise the
                             expected exception.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                calc_rh(filling, diameter)
        else:
            assert calc_rh(filling, diameter) == expected

    def test_calc_rh_valid_input(self):
        """
        Test valid inputs for the `calc_rh` function.

        It tests whether the function returns the expected
        value for the given inputs.

        """
        assert calc_rh(0.5, 1.0) == 0.25
        assert calc_rh(1.0, 1.0) == 0.25
        assert calc_rh(0.2, 0.2) == 0.05
        assert calc_rh(2.0, 2.0) == 0.5

    def test_calc_rh_invalid_input(self):
        """
        Test invalid inputs for the `calc_rh` function.

        It tests whether the function raises the expected
        exception for the given inputs.

        """
        # Test if the function raises a TypeError for non-float input
        with pytest.raises(TypeError):
            calc_rh("0.5", 1.0)  # type: ignore
        with pytest.raises(TypeError):
            calc_rh(0.5, "1.0")  # type: ignore

        # Test if the function raises a ValueError for out-of-range input
        with pytest.raises(ValueError):
            calc_rh(2.1, 1.0)
        with pytest.raises(ValueError):
            calc_rh(0.5, 0.1)
        with pytest.raises(ValueError):
            calc_rh(0.5, 2.1)

    def test_calc_rh_division_by_zero(self):
        """
        Test the `calc_rh` function for division by zero.

        It tests whether the function returns 0 when the denominator
        of the calculation is zero.

        """
        assert calc_rh(0.0, 1.0) == 0
        with pytest.raises(ValueError):
            calc_rh(0.0, 0.0)


class TestMinSlope:
    """
    Class contain tests the min_slope function.
    """

    def test_min_slope_filling_greater_than_0_3(self):
        assert round(min_slope(0.4, 1.0), 2) == 0.71

    def test_min_slope_filling_less_than_or_equal_to_0_3(self):
        assert round(min_slope(0.2, 1.0), 2) == 1.27

    def test_min_slope_invalid_filling_value(self):
        with pytest.raises(ValueError):
            min_slope(2.0, 1.0)

    def test_min_slope_invalid_diameter_value(self):
        with pytest.raises(ValueError):
            min_slope(0.5, 0.1)

    def test_min_slope_invalid_filling_and_diameter_value(self):
        with pytest.raises(ValueError):
            min_slope(2.0, 0.1)

    def test_min_slope_valid_filling_and_diameter_values(self):
        assert round(min_slope(0.5, 1.0), 2) == 0.61

    def test_min_slope_with_theta_and_g_values(self):
        assert round(min_slope(0.5, 1.0, theta=1.2, g=9.81), 2) == 0.49

    @pytest.mark.parametrize(
        "filling, diameter, expected",
        [
            (0.1, 1.0, 2.4071926038740865),
            (0.2, 1.0, 1.2679613277936983),
            (0.3, 1.0, 0.8944912435239528),
            (0.4, 1.0, 0.7137552908970533),
            (0.5, 1.0, 0.6116207951070336),
            (0.6, 1.0, 0.550723538128701),
            (0.7, 1.0, 0.5161624734677323),
            (0.8, 1.0, 0.5026580734069008),
            (0.9, 1.0, 0.5130415616498445),
            (0.1, 2.0, 2.348685545379234),
            (0.2, 2.0, 1.2035963019370433),
            (0.3, 2.0, 0.8231546097405654),
            (0.4, 2.0, 0.6339806638968492),
            (0.5, 2.0, 0.5214128935469622),
            (0.6, 2.0, 0.4472456217619764),
            (0.7, 2.0, 0.39512094038839024),
            (0.8, 2.0, 0.3568776454485266),
            (0.9, 2.0, 0.3280021275695537),
        ],
    )
    def test_min_slope(self, filling, diameter, expected):
        assert min_slope(filling, diameter) == pytest.approx(expected, rel=1e-3)

    def test_min_slope_with_custom_theta(self):
        assert min_slope(0.4, 1.0, theta=2.0) == pytest.approx(0.9516737211960711, rel=1e-3)

    def test_min_slope_with_custom_g(self):
        assert min_slope(0.4, 1.0, g=10.0) == pytest.approx(0.7001939403700093, rel=1e-3)

    def test_min_slope_with_invalid_input(self):
        with pytest.raises(TypeError):
            min_slope("0.4", 1.0)  # type: ignore


class TestMaxSlope:
    """
    Class contain tests the max_slope function.
    """

    @pytest.mark.parametrize(
        "diameter, expected",
        [
            (0.2, 232.42),
            (0.3, 134.56),
            (0.4, 91.74),
            (0.5, 67.89),
            (0.6, 54.03),
            (0.7, 43.25),
            (0.8, 36.03),
            (0.9, 30.75),
            (1.0, 26.8),
        ],
    )
    def test_max_slope_with_valid_input(self, diameter, expected):
        """
        Test the `max_slope` function with valid input.
        """
        assert round(max_slope(diameter), 2) == expected

    def test_max_slope_with_negative_diameter(self):
        """
        Test the `max_slope` function with negative diameter.
        """
        with pytest.raises(ValueError):
            max_slope(-0.3)

    def test_max_slope_with_non_numeric_input(self):
        """
        Test the `max_slope` function with non-numeric input.
        """
        with pytest.raises(TypeError):
            max_slope("0.3")

    def test_max_slope_with_large_diameter(self):
        """
        Test the `max_slope` function with large diameter.
        """
        with pytest.raises(ValueError):
            max_slope(1000)

    def test_max_slope_with_small_diameter(self):
        """
        Test the `max_slopes` function with small diameter.
        """
        with pytest.raises(ValueError):
            max_slope(0.001)

    def test_max_slopes_with_valid_input(self):
        """
        Test the `max_slopes` object with valid input.
        """
        assert round(max_slopes["0.2"], 2) == 232.42
        assert round(max_slopes["0.3"], 2) == 134.56
        assert round(max_slopes["0.4"], 2) == 91.74
        assert round(max_slopes["0.5"], 2) == 67.89
        assert round(max_slopes["0.6"], 2) == 54.03
        assert round(max_slopes["0.7"], 2) == 43.25
        assert round(max_slopes["0.8"], 2) == 36.03
        assert round(max_slopes["0.9"], 2) == 30.75
        assert round(max_slopes["1.0"], 2) == 26.8
        assert round(max_slopes["1.2"], 2) == 21.15
        assert round(max_slopes["1.5"], 2) == 15.7
        assert round(max_slopes["2.0"], 2) == 10.7

    def test_max_sopes_isinstance_lazy_object(self):
        """
        Test the `max_slopes` object type.
        """
        assert isinstance(max_slopes, LazyObject) is True
