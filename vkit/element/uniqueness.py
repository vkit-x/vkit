from typing import Union, Tuple, Sequence, cast
import math

import numpy as np


def check_element_uniqueness(
    value0: Union['Image', 'ScoreMap', 'Mask', np.ndarray, Tuple[int, ...], int, float],
    value1: Union['Image', 'ScoreMap', 'Mask', np.ndarray, Tuple[int, ...], int, float],
):
    if type(value0) is not type(value1):
        return False

    if isinstance(value0, Image):
        value1 = cast(Image, value1)
        if value0.shape != value1.shape:
            return False
        return (value0.mat == value1.mat).all()

    elif isinstance(value0, ScoreMap):
        value1 = cast(ScoreMap, value1)
        if value0.shape != value1.shape:
            return False
        return np.isclose(value0.mat, value1.mat).all()

    elif isinstance(value0, Mask):
        value1 = cast(Mask, value1)
        if value0.shape != value1.shape:
            return False
        return (value0.mat == value1.mat).all()

    elif isinstance(value0, np.ndarray):
        value1 = cast(np.ndarray, value1)
        if value0.shape != value1.shape:
            return False
        if value0.dtype != value1.dtype:
            return False
        if np.issubdtype(value0.dtype, np.floating):
            return np.isclose(value0, value1).all()
        else:
            return (value0 == value1).all()

    elif isinstance(value0, tuple):
        value1 = cast(tuple, value1)
        assert len(value0) == len(value1)
        return value0 == value1

    elif isinstance(value0, int):
        value1 = cast(int, value1)
        return value0 == value1

    elif isinstance(value0, float):
        value1 = cast(float, value1)
        return math.isclose(value0, value1)

    else:
        raise NotImplementedError()


def check_elements_uniqueness(
    values: Sequence[Union['Image', 'ScoreMap', 'Mask', np.ndarray, Tuple[int, ...], int, float]],
):
    unique = True
    for idx, value in enumerate(values):
        if idx == 0:
            continue
        if not check_element_uniqueness(values[0], value):
            unique = False
            break
    return unique


# Cyclic dependency, by design.
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
