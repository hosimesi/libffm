import pytest
from io import StringIO
from ffm import read_importance_weights, read_ffm_data, read_ffm_model


def test_read_importance_weights() -> None:
    data = StringIO(
        """0.9
0.3
0.5
"""
    )
    expected = [0.9, 0.3, 0.5]
    actual = read_importance_weights(data)
    assert pytest.approx(actual) == expected


def test_read_weights_contains_negative() -> None:
    data = StringIO(
        """0.9
-0.1
0.3
"""
    )
    with pytest.raises(AssertionError):
        read_importance_weights(data)


def test_read_ffm_data() -> None:
    data = StringIO(
        """
0 1:7985267:1 2:2281974:1 3:4974058:1 4:3977160:1
1 1:7985267:1 2:2281974:1 3:4974058:1 4:9538220:1
"""
    )
    actual_data, actual_target = read_ffm_data(data)

    assert len(actual_data) == 2
    assert pytest.approx(actual_data[0]) == [
        (1, 7985267, 1),
        (2, 2281974, 1),
        (3, 4974058, 1),
        (4, 3977160, 1),
    ]
    assert pytest.approx(actual_data[1]) == [
        (1, 7985267, 1),
        (2, 2281974, 1),
        (3, 4974058, 1),
        (4, 9538220, 1),
    ]

    assert pytest.approx(actual_target) == [0, 1]


def test_read_ffm_data_contains_negative_feature_idx() -> None:
    data = StringIO(
        """
0 1:-1:1 2:2281974:1 3:4974058:1 4:3977160:1
"""
    )
    with pytest.raises(AssertionError):
        read_ffm_data(data)


def test_read_ffm_model() -> None:
    data = StringIO(
        """n 9991
m 18
k 4
normalization 1
w0,0 1.12387e-05 0.0425162 0.300676 0.445806 
w0,1 0.483978 0.0948449 0.257488 0.199004 
w0,2 0.16383 0.413805 0.0657625 0.310082 
w0,3 0.308203 0.427194 0.309345 0.277694 
"""
    )
    model = read_ffm_model(data)
    assert model.weights.shape == (9991, 18, 4)
    assert model.normalization

    assert model.weights[0, 0] == pytest.approx(
        [1.12387e-05, 0.0425162, 0.300676, 0.445806]
    )
    assert model.weights[0, 1] == pytest.approx(
        [0.483978, 0.0948449, 0.257488, 0.199004]
    )
    assert model.weights[0, 2] == pytest.approx(
        [0.16383, 0.413805, 0.0657625, 0.310082]
    )
    assert model.weights[0, 3] == pytest.approx(
        [0.308203, 0.427194, 0.309345, 0.277694]
    )
