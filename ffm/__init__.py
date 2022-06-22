import warnings
from typing import IO, List, Optional, Sequence, Tuple

import numpy as np

from ffm.libffm import predict as ffm_predict
from ffm.libffm import train as libffm_train

__all__ = ["Dataset", "Model", "train"]


class Dataset:
    def __init__(
        self,
        data: Sequence[Sequence[Tuple[int, int, float]]],
        labels: Sequence[float],
        importance_weights: Optional[Sequence[float]] = None,
    ) -> None:
        assert len(data) == len(labels), "data and labels must be the same length"
        self.data = data
        self.labels = labels
        assert importance_weights is None or len(labels) == len(
            importance_weights
        ), "data and weights must be the same length"
        self.importance_weights = importance_weights

    @classmethod
    def read_ffm_data(cls, data_path: str, weights_path: str = "") -> "Dataset":
        with open(data_path) as f:
            data, labels = read_ffm_data(f)

        weights: Optional[List[float]] = None
        if weights_path:
            with open(weights_path) as f:
                weights = read_importance_weights(f)
            assert len(labels) == len(
                weights
            ), "data and weights must be the same length"
        return cls(data=data, labels=labels, importance_weights=weights)


class Model:
    def __init__(self, weights: np.ndarray, best_iteration: int, normalization: bool):
        self.weights = weights
        self.best_iteration = best_iteration
        self.normalization = normalization

    def dump_model(self, fp: IO) -> None:
        """Dump FFM model to file-like object."""
        assert len(self.weights.shape) == 3
        float_fmt = "{:.6g}"  # This is the same format with ffm-train command.

        fp.write(f"n {self.weights.shape[0]}\n")
        fp.write(f"m {self.weights.shape[1]}\n")
        fp.write(f"k {self.weights.shape[2]}\n")
        fp.write(f"normalization {int(self.normalization)}\n")

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                w = " ".join([float_fmt.format(v) for v in self.weights[i, j]])
                # Put space before break line to keep LIBFFM's output compatibility.
                fp.write(f"w{i},{j} {w} \n")

    def predict(
        self, data: Sequence[Tuple[int, int, float]], nds_rate: Optional[float] = 1.0
    ) -> float:
        return ffm_predict(self.weights, data, self.normalization, nds_rate)

    def dump_libffm_weights(self, fp: IO, key_prefix: str = "") -> None:
        """Dump weights of FFM model like ffm-train's "-m" option"""
        warnings.warn(
            "dump_libffm_weights() is deprecated because it's no longer used.",
            category=DeprecationWarning,
        )

        assert len(self.weights.shape) == 3
        float_fmt = "{:.6g}"  # This is the same format with ffm-train command.

        for i in range(self.weights.shape[0]):
            items = []
            for j in range(self.weights.shape[1]):
                float_arr = [float_fmt.format(v) for v in self.weights[i, j]]
                items.append(f'"{j}":[{",".join(float_arr)}]')
            key = f"{key_prefix}_{i}" if key_prefix else str(i)
            value_json = '{"key":"%s","value":{%s}}' % (key, ",".join(items))
            fp.write(value_json + "\n")

    @classmethod
    def read_ffm_model(cls, model_path: str) -> "Model":
        with open(model_path) as f:
            return read_ffm_model(f)


def train(
    train_data: Dataset,
    valid_data: Optional[Dataset] = None,
    eta: float = 0.2,
    lam: float = 0.00002,
    nr_iters: int = 15,
    k: int = 4,
    nr_threads: int = 1,
    auto_stop=True,
    auto_stop_threshold=-1,
    quiet: bool = True,
    normalization: bool = True,
    random: bool = True,
    nds_rate: float = 1.0,
) -> Model:
    tr = (train_data.data, train_data.labels)
    iw = train_data.importance_weights

    va, iwv = None, None
    if valid_data is not None:
        va = (valid_data.data, valid_data.labels)
        iwv = valid_data.importance_weights

    weights, best_iteration, normalization = libffm_train(
        tr,
        va=va,
        iw=iw,
        iwv=iwv,
        eta=eta,
        lambda_=lam,
        nr_iters=nr_iters,
        k=k,
        nds_rate=nds_rate,
        nr_threads=nr_threads,
        auto_stop=auto_stop,
        auto_stop_threshold=auto_stop_threshold,
        quiet=quiet,
        normalization=normalization,
        random=random,
    )
    return Model(
        weights=weights, best_iteration=best_iteration, normalization=normalization
    )


def read_importance_weights(fp: IO) -> List[float]:
    weights: List[float] = []
    for line in fp:
        line = line.rstrip()
        if not line:
            continue

        value = float(line)
        assert value >= 0, "weights should be positive"
        weights.append(value)
    return weights


def read_ffm_data(
    fp: IO,
) -> Tuple[Sequence[Sequence[Tuple[int, int, float]]], Sequence[float]]:
    data: List[List[Tuple[int, int, float]]] = []
    labels: List[float] = []
    for line in fp:
        line = line.rstrip()
        if not line:
            continue

        line = line.replace("\t", " ")
        items = line.split(" ")
        assert len(items) >= 2

        labels.append(float(items[0]))

        features: List[Tuple[int, int, float]] = []
        for item in items[1:]:
            x = item.split(":")
            assert len(x) == 3

            field = int(x[0])
            feature = int(x[1])
            value = float(x[2])
            assert field >= 0, "field should be larger or equal than 0"
            assert feature >= 0, "feature should be larger or equal than 0"
            features.append((field, feature, value))
        data.append(features)
    assert len(data) == len(labels), "data and labels must be the same length"
    return data, labels


def read_ffm_model(fp: IO) -> Model:
    """Read model_file and return Model object."""

    def _get_value(key: str) -> str:
        line = fp.readline()
        if not line.startswith(key):
            return _get_value(key)

        value = line.split(" ")[1].rstrip()
        return value

    n, m, k = int(_get_value("n")), int(_get_value("m")), int(_get_value("k"))
    normalization = _get_value("normalization") == "1"

    weights = np.empty(shape=(n, m, k), dtype=np.float32)
    for line in fp:
        if not line.startswith("w"):
            continue
        columns = line.rstrip().split(" ")
        assert len(columns) >= 2

        # e.g. columns[0] == "w1,2" -> i_and_j == ["1", "2"]
        i_and_j = columns[0][1:].split(",")
        assert len(i_and_j) == 2
        i, j = int(i_and_j[0]), int(i_and_j[1])

        for k, col in enumerate(columns[1:]):
            weights[i, j, k] = float(col)

    return Model(weights, -1, normalization)
