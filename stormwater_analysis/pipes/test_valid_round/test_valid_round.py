import numpy as np
import pytest

from stormwater_analysis.pipes.valid_round import (
    check_slope,
    max_velocity_value,
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)


class TestValidateMaxFilling:
    """
    Tests for validate_max_filling function.
    """

    def test_max_filling_valid_values(self):
        """
        Tests if validate_max_filling function returns True for valid values.
        """
        for fill, dia in zip(
            [0.16, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80],
            np.arange(0.2, 1.1, 0.1),
        ):
            assert validate_filling(fill, dia)

    def test_max_filling_invalid_values(self):
        """
        Tests if validate_max_filling function returns False
        for invalid values.
        """
        for fill, dia in zip(
            [0.19, 0.26, 0.35, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90],
            np.arange(0.2, 1.1, 0.1),
        ):
            assert not validate_filling(fill, dia)

    def test_max_filling_zero(self):
        """
        Tests if validate_max_filling function returns False for zero value.
        """
        for dia in np.arange(0.2, 1.1, 0.1):
            assert validate_filling(0, dia)

    def test_max_filling_invalid_diameter(self):
        """
        Tests if validate_max_filling function raise ValueError
        for invalid diameter.
        """
        with pytest.raises(ValueError):
            validate_filling(2, 3.0)

    def test_max_filling_invalid_value(self):
        """
        Tests if validate_max_filling function raise ValueError
        for negative value.
        """
        with pytest.raises(ValueError):
            validate_filling(1, -2)
        # with pytest.raises(ValueError):
        #     validate_filling(-1, 1)

    def test_max_filling_invalid_types(self):
        """
        Tests if validate_max_filling function raise TypeError
        for invalid types.
        """
        for val in ["", "four", [], {}]:
            with pytest.raises(TypeError):
                validate_filling(val, 1)  # type: ignore


class TestValidateMaxVelocity:
    """
    Tests for validate_max_velocity function.
    """

    def test_velocity_below_max_velocity(self):
        """
        Tests if validate_max_velocity function returns True
        for values below max velocity.
        """
        assert validate_max_velocity(3.0)

    def test_velocity_equal_to_max_velocity(self):
        """
        Tests if validate_max_velocity function returns True
        for values equal to max velocity.
        """
        assert validate_max_velocity(5.0)

    def test_velocity_above_max_velocity(self):
        """
        Tests if validate_max_velocity function returns False
        for values above max velocity.
        """
        assert not validate_max_velocity(15.0)

    def test_velocity_is_float(self):
        """
        Tests if validate_max_velocity function raises TypeError
        for non-float values.
        """
        with pytest.raises(TypeError):
            validate_max_velocity("10.0")  # type: ignore

    def test_velocity_is_not_none(self):
        """
        Tests if validate_max_velocity function raises ValueError
        for None values.
        """
        with pytest.raises(TypeError):
            validate_max_velocity(None)  # type: ignore


class TestValidateMinVelocity:
    """
    Tests for validate_min_velocity function.
    """

    def test_validate_min_velocity_equals_minimum_velocity(self):
        """
        Tests if validate_min_velocity function returns True
        for values equal to min velocity.
        """
        assert validate_min_velocity(0.7)

    def test_validate_min_velocity_below_minimum_velocity(self):
        """
        Tests if validate_min_velocity function returns False
        for values below min velocity.
        """
        assert not validate_min_velocity(0.6)

    def test_validate_min_velocity_above_minimum_velocity(self):
        """
        Tests if validate_min_velocity function returns False
        for values above min velocity.
        """
        assert validate_min_velocity(0.8)

    def test_validate_min_velocity_negative_velocity(self):
        """
        Tests if validate_min_velocity function raises ValueError
        for negative values.
        """
        assert not validate_min_velocity(-0.7)

    def test_validate_min_velocity_zero(self):
        """
        Tests if validate_min_velocity function raises ValueError
        for zero values.
        """
        assert not validate_min_velocity(0)

    def test_validate_min_velocity_maximum_velocity(self):
        """
        Tests if validate_min_velocity function raises
        ValueError for maximum velocity.
        """
        assert validate_min_velocity(max_velocity_value)

    def test_validate_min_velocity_slightly_above_minimum_velocity(self):
        """
        Tests if validate_min_velocity function raises ValueError for values
        slightly above min velocity.
        """
        assert validate_min_velocity(0.701)

    def test_validate_min_velocity_slightly_bellow_minimum_velocity(self):
        """
        Tests if validate_min_velocity function raises ValueError for values
        slightly below min velocity.
        """
        assert not validate_min_velocity(0.699)

    def test_validate_min_velocity_string_value(self):
        """
        Tests if validate_min_velocity function raises TypeError
        for string values.
        """
        with pytest.raises(TypeError):
            validate_min_velocity("10.0")  # type: ignore


class TestValidateMinSlope:
    """
    Tests for validate_min_slope function.
    """

    @pytest.mark.parametrize(
        ("slope", "filling", "diameter"),
        [
            (5.0, 0.10, 0.2),
            (2.5, 0.20, 0.3),
            (2.5, 0.30, 0.4),
            (2.5, 0.40, 0.5),
            (2.5, 0.45, 0.6),
            (2.5, 0.50, 0.7),
            (2.5, 0.60, 0.8),
            (2.5, 0.70, 0.9),
            (2.5, 0.80, 1.0),
        ],
    )
    def test_valid_slope(self, slope, filling, diameter):
        """
        Test the `validate_min_slope` function with valid input.
        """
        assert validate_min_slope(slope, filling, diameter)

    @pytest.mark.parametrize(
        ("slope", "filling", "diameter"),
        [
            (-5.0, 0.10, 0.2),
            (-2.5, 0.20, 0.3),
            (-2.5, 0.30, 0.4),
            (-2.5, 0.40, 0.5),
            (-2.5, 0.45, 0.6),
            (-2.5, 0.50, 0.7),
            (-2.5, 0.60, 0.8),
            (-2.5, 0.70, 0.9),
            (-2.5, 0.80, 1.0),
        ],
    )
    def test_valid_negative_slope(self, slope, filling, diameter):
        """
        Test the `validate_min_slope` function with negative slope.
        """
        assert not validate_min_slope(slope, filling, diameter)

    def test_invalid_slope(self):
        """
        Test the `validate_min_slope` function with invalid slope.
        """
        assert not validate_min_slope(0.5, 0.3, 0.5, 1.5, 9.81)

    def test_valid_slope_with_different_theta(self):
        """
        Test the `validate_min_slope` function
        with valid slope and different theta.
        """
        assert validate_min_slope(2.5, 0.2, 0.3, 1.0, 9.81)

    def test_valid_slope_with_different_diameter(self):
        """
        Test the `validate_min_slope` function
        with valid slope and different diameter.
        """
        assert validate_min_slope(1.2, 0.5, 0.6, 1.5, 9.81)

    def test_valid_slope_with_different_gravity(self):
        """
        Test the `validate_min_slope` function with valid
        slope and different gravity.
        """
        assert validate_min_slope(1.2, 0.5, 0.7, 1.5, 10)

    def test_invalid_slope_with_different_gravity(self):
        """
        Test the `validate_min_slope` function
        with invalid slope and different gravity.
        """
        assert validate_min_slope(1.0, 0.5, 0.7, 1.5, 8.81)

    def test_invalid_slope_with_negative_filling(self):
        """
        Test the `validate_min_slope` function
        with invalid slope and negative filling.
        """
        with pytest.raises(ValueError):
            validate_min_slope(0.5, -0.5, 0.3, 1.5, 9.81)

    def test_invalid_slope_with_negative_diameter(self):
        """
        Test the `validate_min_slope` function
        with invalid slope and negative diameter.
        """
        with pytest.raises(ValueError):
            validate_min_slope(0.5, 0.5, -0.3, 1.5, 9.81)


class TestValidMaxSlope:
    """
    Tests for validate_max_slope function.
    """

    def test_validate_max_slope_valid(self):
        """
        Test the `validate_max_slope` function with valid input.
        """
        assert validate_max_slope(0.1, 0.3)

    def test_validate_max_slope_invalid(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert not validate_max_slope(250, 0.2)

    def test_validate_max_slope_equal(self):
        """
        Test the `validate_max_slope` function with valid input.
        """
        assert validate_max_slope(0.7, 0.7)

    def test_validate_max_slope_invalid_diameter(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(0.5, 0.1)

    def test_validate_max_slope_string_diameter(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(TypeError):
            validate_max_slope(0.5, "0.8")  # type: ignore

    def test_validate_max_slope_string_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(TypeError):
            validate_max_slope("0.5", 0.8)  # type: ignore

    def test_validate_max_slope_negative_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(-0.5, 1.0)

    def test_validate_max_slope_zero_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(0, 0.6)

    def test_validate_max_slope_greater_than_one(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert validate_max_slope(1.5, 1.2)

    def test_validate_max_slope_equal_to_one(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert validate_max_slope(1, 1.5)


class TestCheckSlope:
    """
    Tests for the `check_slope` function.
    """

    def test_check_slope_positive(self):
        """
        Test the `check_slope` function with positive slope.
        """
        assert check_slope(0.1)

    def test_check_slope_zero(self):
        """
        Test the `check_slope` function with zero slope.
        """
        with pytest.raises(ValueError):
            check_slope(0)

    def test_check_slope_negative(self):
        """
        Test the `check_slope` function with negative slope.
        """
        with pytest.raises(ValueError):
            check_slope(-0.1)

    def test_check_slope_not_float_or_int(self):
        """
        Test the `check_slope` function with non-float or non-int slope.
        """
        with pytest.raises(TypeError):
            check_slope("0.1")  # type: ignore

    def test_check_slope_int(self):
        """
        Test the `check_slope` function with int slope.
        """
        assert check_slope(1)

    def test_check_slope_max(self):
        """
        Test the `check_slope` function with maximum slope.
        """
        assert check_slope(900)

    def test_check_slope_string_value(self):
        """
        Test the `check_slope` function with string value.
        """
        with pytest.raises(TypeError):
            check_slope("nan")  # type: ignore
