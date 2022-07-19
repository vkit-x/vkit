from typing import (
    get_args,
    cast,
    Sequence,
    Optional,
    TypeVar,
    Any,
    Type,
    Union,
    Tuple,
    Mapping,
    List,
)
import subprocess
from os import PathLike
from collections import abc
import re

from numpy.random import Generator as RandomGenerator
import cv2 as cv
import iolite as io
import cattrs
from cattrs.errors import ClassValidationError

from vkit.utility import PathType


def is_path_type(path: Any):
    return isinstance(path, (str, PathLike))  # type: ignore


def read_json_file(path: PathType):
    return io.read_json(path, expandvars=True)


def get_data_folder(file: PathType):
    proc = subprocess.run(
        f'pyproject-data-folder "$VKIT_ROOT" "$VKIT_DATA" "{file}"',
        shell=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    data_folder = proc.stdout.strip()
    assert data_folder

    io.folder(data_folder, touch=True)

    return data_folder


_T_ITEM = TypeVar('_T_ITEM')


def rng_choice(
    rng: RandomGenerator,
    items: Sequence[_T_ITEM],
    probs: Optional[Sequence[float]] = None,
) -> _T_ITEM:
    idx = rng.choice(len(items), p=probs)
    return items[idx]


def rng_choice_with_size(
    rng: RandomGenerator,
    items: Sequence[_T_ITEM],
    size: int,
    probs: Optional[Sequence[float]] = None,
) -> Sequence[_T_ITEM]:
    indices = rng.choice(len(items), p=probs, size=size)
    return [items[idx] for idx in indices]


_CV_INTER_FLAGS = cast(
    Sequence[int],
    (
        # NOTE: Keep the EXACT version.
        # cv.INTER_NEAREST,
        # NOTE: this one is Any.
        cv.INTER_NEAREST_EXACT,
        # NOTE: Keep the EXACT version.
        # cv.INTER_LINEAR,
        cv.INTER_LINEAR_EXACT,
        cv.INTER_CUBIC,
        cv.INTER_LANCZOS4,
    ),
)


def sample_cv_resize_interpolation(
    rng: RandomGenerator,
    include_cv_inter_area: bool = False,
):
    flags = _CV_INTER_FLAGS
    if include_cv_inter_area:
        flags = (*_CV_INTER_FLAGS, cv.INTER_AREA)
    return rng_choice(rng, flags)


_T_TARGET = TypeVar('_T_TARGET')

_cattrs = cattrs.GenConverter(forbid_extra_keys=True)


def dyn_structure(
    dyn_object: Any,
    target_cls: Type[_T_TARGET],
    support_path_type: bool = False,
    force_path_type: bool = False,
    support_none_type: bool = False,
) -> _T_TARGET:
    if support_none_type and dyn_object is None:
        return target_cls()

    if support_path_type or force_path_type:
        dyn_object_is_path_type = is_path_type(dyn_object)
        if force_path_type:
            assert dyn_object_is_path_type
        if dyn_object_is_path_type:
            dyn_object = read_json_file(dyn_object)

    isinstance_target_cls = False
    try:
        if isinstance(dyn_object, target_cls):
            isinstance_target_cls = True
    except TypeError:
        # target_cls could be type annotation like Sequence[int].
        pass

    if isinstance_target_cls:
        # Do nothing.
        pass
    elif isinstance(dyn_object, abc.Mapping):
        try:
            dyn_object = _cattrs.structure(dyn_object, target_cls)
        except ClassValidationError:
            # cattrs cannot handle Class with hierarchy structure,
            # in such case, fallback to manually initialization.
            dyn_object = target_cls(**dyn_object)
    elif isinstance(dyn_object, abc.Sequence):
        dyn_object = _cattrs.structure(dyn_object, target_cls)
    else:
        raise NotImplementedError()

    return dyn_object


def normalize_to_probs(weights: Sequence[float]):
    total = sum(weights)
    probs = [weight / total for weight in weights]
    return probs


_T_KEY = TypeVar('_T_KEY')


def normalize_to_keys_and_probs(
    key_weight_items: Union[Sequence[Tuple[_T_KEY, float]], Mapping[_T_KEY, float]]
) -> Tuple[Sequence[_T_KEY], Sequence[float]]:
    keys: List[_T_KEY] = []
    weights: List[float] = []

    if isinstance(key_weight_items, abc.Sequence):
        for key, weight in key_weight_items:
            keys.append(key)
            weights.append(weight)
    elif isinstance(key_weight_items, abc.Mapping):  # type: ignore
        for key, weight in key_weight_items.items():
            keys.append(key)
            weights.append(weight)
    else:
        raise NotImplementedError()

    probs = normalize_to_probs(weights)
    return keys, probs


def convert_camel_case_name_to_snake_case_name(name: str):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def get_config_class_snake_case_name(class_name: str):
    snake_case_name = convert_camel_case_name_to_snake_case_name(class_name)
    if snake_case_name.endswith('_config'):
        snake_case_name = snake_case_name[:-len('_config')]
    return snake_case_name


def get_generic_classes(cls: Type[Any]):
    return get_args(cls.__orig_bases__[0])  # type: ignore
