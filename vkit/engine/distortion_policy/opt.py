from typing import Optional, Tuple
from enum import Enum, auto

from numpy.random import RandomState

from vkit.utility import rnd_choice_with_size

LEVEL_MAX = 10
CHANNELS = [0, 1, 2]


def sample_channels(rnd: RandomState):
    num_channels = rnd.randint(1, 4)
    channels = None
    if num_channels < 3:
        channels = sorted(rnd_choice_with_size(rnd, CHANNELS, num_channels))
    return channels


def sample_int(
    level: int,
    value_min: int,
    value_max: int,
    prob_negative: Optional[float],
    rnd: RandomState,
    inverse_level: bool = False,
):
    if inverse_level:
        level = LEVEL_MAX + 1 - level

    value_range = value_max - value_min
    level_value_min = round(value_min + (level - 1) / LEVEL_MAX * value_range)
    level_value_max = round(value_min + level / LEVEL_MAX * value_range)

    if level == LEVEL_MAX:
        # Make sure value_max could be sampled.
        level_value_max += 1

    value = rnd.randint(level_value_min, max(level_value_min + 1, level_value_max))
    if prob_negative and rnd.random() < prob_negative:
        value *= -1

    return value


class SampleFloatMode(Enum):
    LINEAR = auto()
    QUAD = auto()


def func_quad(x: float):
    return -x**2 + 2 * x


def sample_float(
    level: int,
    value_min: float,
    value_max: float,
    prob_reciprocal: Optional[float],
    rnd: RandomState,
    mode: SampleFloatMode = SampleFloatMode.LINEAR,
    inverse_level: bool = False,
):
    if inverse_level:
        level = LEVEL_MAX + 1 - level

    value_range = value_max - value_min

    if mode == SampleFloatMode.LINEAR:
        level_ratio_min = (level - 1) / LEVEL_MAX
        level_ratio_max = level / LEVEL_MAX

    elif mode == SampleFloatMode.QUAD:
        level_ratio_min = func_quad((level - 1) / LEVEL_MAX)
        level_ratio_max = func_quad(level / LEVEL_MAX)

    else:
        raise NotImplementedError()

    level_value_min = value_min + level_ratio_min * value_range
    level_value_max = value_min + level_ratio_max * value_range
    value = rnd.uniform(level_value_min, level_value_max)

    if prob_reciprocal and rnd.random() < prob_reciprocal:
        value = 1 / value

    return value


def generate_grid_size(
    grid_size_min: int,
    grid_size_ratio: float,
    shape: Tuple[int, int],
):
    return max(
        grid_size_min,
        int(grid_size_ratio * max(shape)),
    )
