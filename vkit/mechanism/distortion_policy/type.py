# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import (
    Mapping,
    Any,
    Generic,
    TypeVar,
    Type,
    Iterable,
    Union,
    Tuple,
    Optional,
)

from numpy.random import Generator as RandomGenerator

from vkit.utility import (
    dyn_structure,
    get_generic_classes,
    PathType,
)
from vkit.element import (
    Shapable,
    Image,
    Point,
    PointList,
    PointTuple,
    Polygon,
    Mask,
    ScoreMap,
)
from ..distortion.interface import DistortionConfig, DistortionState, Distortion

_T_GENERATOR_CONFIG = TypeVar('_T_GENERATOR_CONFIG')
_T_CONFIG = TypeVar('_T_CONFIG', bound=DistortionConfig)
_T_STATE = TypeVar('_T_STATE', bound=DistortionState)


class DistortionConfigGenerator(Generic[_T_GENERATOR_CONFIG, _T_CONFIG]):

    @classmethod
    def get_generator_config_cls(cls) -> Type[_T_GENERATOR_CONFIG]:
        return get_generic_classes(cls)[0]  # type: ignore

    @classmethod
    def get_config_cls(cls) -> Type[_T_CONFIG]:
        return get_generic_classes(cls)[1]  # type: ignore

    def __init__(self, config: _T_GENERATOR_CONFIG, level: int) -> None:
        self.config = config
        assert 1 <= level <= 10
        self.level = level

    def __call__(self, shape: Tuple[int, int], rng: RandomGenerator) -> _T_CONFIG:
        raise NotImplementedError()


class DistortionPolicy(Generic[_T_GENERATOR_CONFIG, _T_CONFIG, _T_STATE]):

    def __init__(
        self,
        distortion: Distortion[_T_CONFIG, _T_STATE],
        config_for_config_generator: _T_GENERATOR_CONFIG,
        config_generator_cls: Type[DistortionConfigGenerator[_T_GENERATOR_CONFIG, _T_CONFIG]],
    ):
        self.distortion = distortion
        self.config_for_config_generator = config_for_config_generator
        self.config_generator_cls = config_generator_cls

    def distort(
        self,
        level: int,
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
        point: Optional[Point] = None,
        points: Optional[Union[PointList, PointTuple, Iterable[Point]]] = None,
        corner_points: Optional[Union[PointList, PointTuple, Iterable[Point]]] = None,
        polygon: Optional[Polygon] = None,
        polygons: Optional[Iterable[Polygon]] = None,
        rng: Optional[RandomGenerator] = None,
        enable_debug: bool = False,
    ):
        config_generator = self.config_generator_cls(
            self.config_for_config_generator,
            level,
        )
        return self.distortion.distort(
            config_or_config_generator=config_generator,
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
            point=point,
            points=points,
            corner_points=corner_points,
            polygon=polygon,
            polygons=polygons,
            rng=rng,
            get_config=enable_debug,
            get_state=enable_debug,
        )

    @property
    def name(self):
        return self.config_generator_cls.get_config_cls().get_name()

    def __repr__(self):
        return f'DistortionPolicy({self.name})'


class DistortionPolicyFactory(Generic[_T_GENERATOR_CONFIG, _T_CONFIG, _T_STATE]):

    def __init__(
        self,
        distortion: Distortion[_T_CONFIG, _T_STATE],
        config_generator_cls: Type[DistortionConfigGenerator[_T_GENERATOR_CONFIG, _T_CONFIG]],
    ):
        self.distortion = distortion
        self.config_generator_cls = config_generator_cls

    def create(
        self,
        config: Optional[Union[Mapping[str, Any], PathType, _T_GENERATOR_CONFIG]] = None,
    ):
        config = dyn_structure(
            config,
            self.config_generator_cls.get_generator_config_cls(),
            support_path_type=True,
            support_none_type=True,
        )
        return DistortionPolicy(
            self.distortion,
            config,
            self.config_generator_cls,
        )

    @property
    def name(self):
        return self.config_generator_cls.get_config_cls().get_name()
