from typing import (
    get_args,
    Sequence,
    Optional,
    TypeVar,
    Any,
    Type,
    Union,
    Tuple,
    Dict,
    List,
)
import subprocess
from os import PathLike
from collections import abc
import re

from numpy.random import RandomState
import iolite as io
import cattrs
from cattrs.errors import ClassValidationError

from vkit.utility import PathType


def is_path_type(path: PathType):
    return isinstance(path, (str, bytes, PathLike))  # type: ignore


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


def rnd_choice(
    rnd: RandomState,
    items: Sequence[_T_ITEM],
    probs: Optional[Sequence[float]] = None,
) -> _T_ITEM:
    idx = rnd.choice(len(items), p=probs)
    return items[idx]


def rnd_choice_with_size(
    rnd: RandomState,
    items: Sequence[_T_ITEM],
    size: int,
    probs: Optional[Sequence[float]] = None,
) -> Sequence[_T_ITEM]:
    indices = rnd.choice(len(items), p=probs, size=size)
    return [items[idx] for idx in indices]


_T_TARGET = TypeVar('_T_TARGET')


def dyn_structure(
    dyn_object: Any,
    target_cls: Type[_T_TARGET],
    support_path_type: bool = False,
    support_none_type: bool = False,
) -> _T_TARGET:
    if support_none_type and dyn_object is None:
        return target_cls()

    if support_path_type and is_path_type(dyn_object):
        dyn_object = read_json_file(dyn_object)

    if isinstance(dyn_object, abc.Mapping):
        try:
            return cattrs.structure(dyn_object, target_cls)
        except ClassValidationError:
            return target_cls(**dyn_object)

    if isinstance(dyn_object, target_cls):
        return dyn_object

    raise NotImplementedError()


def normalize_to_probs(weights: Sequence[float]):
    total = sum(weights)
    probs = [weight / total for weight in weights]
    return probs


_T_KEY = TypeVar('_T_KEY')


def normalize_to_keys_and_probs(
    key_weight_items: Union[Sequence[Tuple[_T_KEY, float]], Dict[_T_KEY, float]]
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
