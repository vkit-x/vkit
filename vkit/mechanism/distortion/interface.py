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
    cast,
    get_origin,
    Any,
    Callable,
    Mapping,
    Generic,
    Iterable,
    Type,
    TypeVar,
    Union,
    Sequence,
    Tuple,
    Optional,
)

import attrs
from numpy.random import default_rng, Generator as RandomGenerator

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
from vkit.utility import (
    dyn_structure,
    get_config_class_snake_case_name,
)


class DistortionConfig:

    _cached_name: str = ''

    @classmethod
    def get_name(cls):
        if not cls._cached_name:
            cls._cached_name = get_config_class_snake_case_name(cls.__name__)
        return cls._cached_name

    @property
    def name(self):
        return self.get_name()

    @property
    def supports_rng_state(self) -> bool:
        return False

    @property
    def rng_state(self) -> Optional[Mapping[str, Any]]:
        return None

    @rng_state.setter
    def rng_state(self, val: Mapping[str, Any]):
        pass


_T_CONFIG = TypeVar('_T_CONFIG', bound=DistortionConfig)


class DistortionState(Generic[_T_CONFIG]):

    def __init__(
        self,
        config: _T_CONFIG,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        raise NotImplementedError()

    @property
    def result_shape(self) -> Optional[Tuple[int, int]]:
        return None


class DistortionNopState(DistortionState[_T_CONFIG]):

    def __init__(
        self,
        config: _T_CONFIG,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        raise NotImplementedError()


_T_STATE = TypeVar('_T_STATE', bound=DistortionState)


@attrs.define
class DistortionResult:
    shape: Tuple[int, int]
    image: Optional[Image] = None
    mask: Optional[Mask] = None
    score_map: Optional[ScoreMap] = None
    active_mask: Optional[Mask] = None
    point: Optional[Point] = None
    points: Optional[PointTuple] = None
    corner_points: Optional[PointTuple] = None
    polygon: Optional[Polygon] = None
    polygons: Optional[Sequence[Polygon]] = None
    config: Optional[Any] = None
    state: Optional[Any] = None
    meta: Optional[Mapping[str, Any]] = None


@attrs.define
class DistortionInternals(Generic[_T_CONFIG, _T_STATE]):
    config: _T_CONFIG
    state: Optional[_T_STATE]
    shape: Tuple[int, int]
    rng: Optional[RandomGenerator]

    def restore_rng_if_supported(self):
        if self.rng:
            assert self.config.supports_rng_state and self.config.rng_state
            self.rng.bit_generator.state = self.config.rng_state


class Distortion(Generic[_T_CONFIG, _T_STATE]):

    # yapf: disable
    def __init__(
        self,
        config_cls: Type[_T_CONFIG],
        state_cls: Type[_T_STATE],
        func_image: Callable[
            [
                _T_CONFIG,
                Optional[_T_STATE],
                Image,
                Optional[RandomGenerator],
            ],
            Image,
        ],
        func_mask: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Mask,
                    Optional[RandomGenerator],
                ],
                Mask,
            ]
        ] = None,
        func_score_map: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    ScoreMap,
                    Optional[RandomGenerator],
                ],
                ScoreMap,
            ]
        ] = None,
        func_active_mask: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Tuple[int, int],
                    Optional[RandomGenerator],
                ],
                Mask,
            ]
        ] = None,
        func_point: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Tuple[int, int],
                    Point,
                    Optional[RandomGenerator],
                ],
                Point,
            ]
        ] = None,
        func_points: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Tuple[int, int],
                    Union[PointList, PointTuple, Iterable[Point]],
                    Optional[RandomGenerator],
                ],
                PointTuple,
            ]
        ] = None,
        func_polygon: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Tuple[int, int],
                    Polygon,
                    Optional[RandomGenerator],
                ],
                Polygon,
            ]
        ] = None,
        func_polygons: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Tuple[int, int],
                    Iterable[Polygon],
                    Optional[RandomGenerator],
                ],
                Sequence[Polygon],
            ]
        ] = None,
    ):
        # yapf: enable
        self.config_cls = config_cls
        self.state_cls = state_cls

        self.func_image = func_image
        self.func_score_map = func_score_map
        self.func_mask = func_mask
        self.func_active_mask = func_active_mask

        self.func_point = func_point
        self.func_points = func_points
        self.func_polygon = func_polygon
        self.func_polygons = func_polygons

    @property
    def is_geometric(self):
        return any((
            self.func_point,
            self.func_points,
            self.func_polygon,
            self.func_polygons,
            self.func_active_mask,
        ))

    # yapf: disable
    def prepare_config_and_rng(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ) -> Tuple[_T_CONFIG, Optional[RandomGenerator]]:
        # yapf: enable
        if callable(config_or_config_generator):
            config_generator = cast(
                Callable[[Tuple[int, int], RandomGenerator], Union[_T_CONFIG, Mapping[str, Any]]],
                config_or_config_generator,
            )
            if not rng:
                raise RuntimeError('config_generator but rng is None.')
            config = dyn_structure(config_generator(shape, rng), self.config_cls)

        else:
            config_or_config_generator = cast(
                Union[_T_CONFIG, Mapping[str, Any]],
                config_or_config_generator,
            )
            config = dyn_structure(config_or_config_generator, self.config_cls)

        if config.supports_rng_state:
            if not config.rng_state:
                if not rng:
                    raise RuntimeError('both config.rng_state and rng are None.')
                config.rng_state = rng.bit_generator.state

            # Calling rng methods changes rng's state. We don't want to change the state
            # of exterior rng, hence making a copy here.
            rng = default_rng()
            rng.bit_generator.state = config.rng_state

        else:
            # Force not passing rng.
            rng = None

        return config, rng

    @classmethod
    def get_shape_from_shapable_or_shape(cls, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        if isinstance(shapable_or_shape, (list, tuple)):
            assert len(shapable_or_shape) == 2
            return shapable_or_shape
        else:
            return shapable_or_shape.shape

    # yapf: disable
    def prepare_internals(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        state: Optional[_T_STATE],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rng: Optional[RandomGenerator] = None,
        disable_state_initialization: bool = False,
    ):
        # yapf: enable
        shape = self.get_shape_from_shapable_or_shape(shapable_or_shape)

        config, rng = self.prepare_config_and_rng(
            config_or_config_generator,
            shape,
            rng,
        )

        if get_origin(self.state_cls) is not DistortionNopState:
            if state is None and not disable_state_initialization:
                state = self.state_cls(config, shape, rng)
        else:
            state = None

        return DistortionInternals(config, state, shape, rng)

    # yapf: disable
    def generate_config_and_state(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        state: Optional[_T_STATE],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return internals.config, internals.state

    # yapf: disable
    def generate_config(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
            disable_state_initialization=True,
        )
        return internals.config

    # yapf: disable
    def generate_state(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return internals.state

    def distort_image_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        image: Image,
    ):
        internals.restore_rng_if_supported()

        return self.func_image(
            internals.config,
            internals.state,
            image,
            internals.rng,
        )

    # yapf: disable
    def distort_image(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        image: Image,
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=image,
            rng=rng,
        )
        return self.distort_image_based_on_internals(internals, image)

    def distort_score_map_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        score_map: ScoreMap,
    ):
        internals.restore_rng_if_supported()

        if self.func_score_map:
            return self.func_score_map(
                internals.config,
                internals.state,
                score_map,
                internals.rng,
            )

        else:
            # NOP.
            return score_map

    # yapf: disable
    def distort_score_map(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        score_map: ScoreMap,
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=score_map,
            rng=rng,
        )
        return self.distort_score_map_based_on_internals(internals, score_map)

    def distort_mask_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        mask: Mask,
    ):
        internals.restore_rng_if_supported()

        if self.func_mask:
            return self.func_mask(
                internals.config,
                internals.state,
                mask,
                internals.rng,
            )

        else:
            # NOP.
            return mask

    # yapf: disable
    def distort_mask(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        mask: Mask,
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=mask,
            rng=rng,
        )
        return self.distort_mask_based_on_internals(internals, mask)

    def get_active_mask_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
    ):
        # TODO: Something is wrong with cv.remap when dealing with border interpolation.
        # This method could generate a mask "connects" to the transformed border.
        internals.restore_rng_if_supported()

        if self.func_active_mask:
            return self.func_active_mask(
                internals.config,
                internals.state,
                internals.shape,
                internals.rng,
            )

        else:
            mask = Mask.from_shape(internals.shape, value=1)
            return self.distort_mask_based_on_internals(internals, mask)

    # yapf: disable
    def get_active_mask(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return self.get_active_mask_based_on_internals(internals)

    def distort_point_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        point: Point,
    ):
        internals.restore_rng_if_supported()

        if self.func_point:
            return self.func_point(
                internals.config,
                internals.state,
                internals.shape,
                point,
                internals.rng,
            )

        elif self.func_points:
            distorted_points = self.func_points(
                internals.config,
                internals.state,
                internals.shape,
                [point],
                internals.rng,
            )
            return distorted_points[0]

        else:
            if self.is_geometric:
                raise RuntimeError('Missing self.func_points or self.func_point.')

            # NOP.
            return point

    # yapf: disable
    def distort_point(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        point: Point,
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return self.distort_point_based_on_internals(internals, point)

    def distort_points_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        points: Union[PointList, PointTuple, Iterable[Point]],
    ):
        internals.restore_rng_if_supported()

        points = PointList(points)

        if self.func_points:
            return self.func_points(
                internals.config,
                internals.state,
                internals.shape,
                points,
                internals.rng,
            )

        else:
            new_points = PointList()
            for point in points:
                new_point = self.distort_point_based_on_internals(internals, point)
                new_points.append(new_point)
            return new_points.to_point_tuple()

    # yapf: disable
    def distort_points(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        points: Union[PointList, PointTuple, Iterable[Point]],
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return self.distort_points_based_on_internals(internals, points)

    def distort_polygon_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        polygon: Polygon,
    ):
        internals.restore_rng_if_supported()

        if self.func_polygon:
            return self.func_polygon(
                internals.config,
                internals.state,
                internals.shape,
                polygon,
                internals.rng,
            )

        elif self.func_polygons:
            distorted_polygons = self.func_polygons(
                internals.config,
                internals.state,
                internals.shape,
                [polygon],
                internals.rng,
            )
            return distorted_polygons[0]

        else:
            new_points = self.distort_points_based_on_internals(internals, polygon.points)
            return Polygon.create(points=new_points)

    # yapf: disable
    def distort_polygon(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        polygon: Polygon,
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return self.distort_polygon_based_on_internals(internals, polygon)

    def distort_polygons_based_on_internals(
        self,
        internals: DistortionInternals[_T_CONFIG, _T_STATE],
        polygons: Iterable[Polygon],
    ):
        internals.restore_rng_if_supported()

        if self.func_polygons:
            return self.func_polygons(
                internals.config,
                internals.state,
                internals.shape,
                polygons,
                internals.rng,
            )

        else:
            new_polygons = []
            for polygon in polygons:
                new_polygon = self.distort_polygon_based_on_internals(internals, polygon)
                new_polygons.append(new_polygon)
            return new_polygons

    # yapf: disable
    def distort_polygons(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        polygons: Iterable[Polygon],
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )
        return self.distort_polygons_based_on_internals(internals, polygons)

    @classmethod
    def get_shape(
        cls,
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
    ):
        if shapable_or_shape is None:
            shapable_or_shape = image or mask or score_map
        assert shapable_or_shape

        return cls.get_shape_from_shapable_or_shape(shapable_or_shape)

    def clip_result_elements(self, result: DistortionResult):
        if not self.is_geometric:
            return

        if result.point:
            result.point = result.point.to_clipped_point(result.shape)

        if result.points:
            result.points = result.points.to_clipped_points(result.shape)

        if result.corner_points:
            result.corner_points = result.corner_points.to_clipped_points(result.shape)

        if result.polygon:
            result.polygon = result.polygon.to_clipped_polygon(result.shape)

        if result.polygons:
            result.polygons = [
                polygon.to_clipped_polygon(result.shape) for polygon in result.polygons
            ]

    # yapf: disable
    def distort(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Mapping[str, Any]],
            Callable[
                [Tuple[int, int], RandomGenerator],
                Union[_T_CONFIG, Mapping[str, Any]],
            ],
        ],
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
        point: Optional[Point] = None,
        points: Optional[Union[PointList, PointTuple, Iterable[Point]]] = None,
        corner_points: Optional[Union[PointList, PointTuple, Iterable[Point]]] = None,
        polygon: Optional[Polygon] = None,
        polygons: Optional[Iterable[Polygon]] = None,
        get_active_mask: bool = False,
        get_config: bool = False,
        get_state: bool = False,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        shape = self.get_shape(
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
        )

        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=shape,
            rng=rng,
        )

        # If is geometric distortion, the shape will be updated.
        result = DistortionResult(shape=shape)

        if self.is_geometric:
            assert internals.state and internals.state.result_shape
            result.shape = internals.state.result_shape
        else:
            result.shape = shape

        if image:
            result.image = self.distort_image_based_on_internals(internals, image)
            assert result.shape == result.image.shape

        if mask:
            result.mask = self.distort_mask_based_on_internals(internals, mask)
            assert result.shape == result.mask.shape

        if score_map:
            result.score_map = self.distort_score_map_based_on_internals(internals, score_map)
            assert result.shape == result.score_map.shape

        if point:
            result.point = self.distort_point_based_on_internals(internals, point)

        if points:
            result.points = self.distort_points_based_on_internals(internals, points)

        if corner_points:
            result.corner_points = self.distort_points_based_on_internals(internals, corner_points)

        if polygon:
            result.polygon = self.distort_polygon_based_on_internals(internals, polygon)

        if polygons:
            result.polygons = self.distort_polygons_based_on_internals(internals, polygons)

        if get_active_mask:
            result.active_mask = self.get_active_mask_based_on_internals(internals)
            assert result.shape == result.active_mask.shape

        if get_config:
            result.config = internals.config

        if get_state:
            result.state = internals.state

        self.clip_result_elements(result)

        return result
