"""Contract tests for the serving GNN feature list."""

from sa.core.data_manager import get_default_feature_columns


def test_get_default_feature_columns_returns_a_fresh_ordered_list():
    first = get_default_feature_columns()
    second = get_default_feature_columns()

    assert first == second
    assert first is not second
    assert len(first) == 33
    assert first[:3] == ["ValMaxFill", "ValMaxV", "ValMinV"]
    assert first[-3:] == [
        "urban_highly_impervious",
        "mountains_rocky",
        "mountains_vegetated",
    ]
