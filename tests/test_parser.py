import pytest
from io import StringIO
from ffm import read_importance_weights, read_ffm_data


def test_read_importance_weights() -> None:
    data = StringIO("""0.9
0.3
0.5
""")
    expected = [0.9, 0.3, 0.5]
    actual = read_importance_weights(data)
    assert pytest.approx(actual) == expected


def test_read_weights_contains_negative() -> None:
    data = StringIO("""0.9
-0.1
0.3
""")
    with pytest.raises(AssertionError):
        read_importance_weights(data)


def test_read_ffm_data() -> None:
    data = StringIO("""
0 1:7985267:1 2:2281974:1 3:4974058:1 4:3977160:1
1 1:7985267:1 2:2281974:1 3:4974058:1 4:9538220:1
""")
    actual_data, actual_target = read_ffm_data(data)

    assert len(actual_data) == 2
    assert pytest.approx(actual_data[0]) == [(1, 7985267, 1), (2, 2281974, 1), (3, 4974058, 1), (4, 3977160, 1)]
    assert pytest.approx(actual_data[1]) == [(1, 7985267, 1), (2, 2281974, 1), (3, 4974058, 1), (4, 9538220, 1)]

    assert pytest.approx(actual_target) == [0, 1]


def test_read_ffm_data_contains_negative_feature_idx() -> None:
    data = StringIO("""
0 1:-1:1 2:2281974:1 3:4974058:1 4:3977160:1
""")
    with pytest.raises(AssertionError):
        read_ffm_data(data)
