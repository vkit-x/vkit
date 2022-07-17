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
    Polygon,
    TextPolygon,
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

    def __init__(self, config: _T_CONFIG, shape: Tuple[int, int], rng: Optional[RandomGenerator]):
        raise NotImplementedError()


class DistortionNopState(DistortionState[_T_CONFIG]):

    def __init__(self, config: _T_CONFIG, shape: Tuple[int, int], rng: Optional[RandomGenerator]):
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
    points: Optional[PointList] = None
    polygon: Optional[Polygon] = None
    polygons: Optional[Sequence[Polygon]] = None
    text_polygon: Optional[TextPolygon] = None
    text_polygons: Optional[Sequence[TextPolygon]] = None
    config: Optional[Any] = None
    state: Optional[Any] = None
    meta: Optional[Mapping[str, Any]] = None


@attrs.define
class DistortionInternals(Generic[_T_CONFIG, _T_STATE]):
    config: _T_CONFIG
    state: Optional[_T_STATE]
    shape: Tuple[int, int]
    rng: Optional[RandomGenerator]


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
                    Union[PointList, Iterable[Point]],
                    Optional[RandomGenerator],
                ],
                PointList,
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
            # Process rng_state.
            if config.rng_state:
                # Create/replace rng.
                rng = default_rng()
                rng.bit_generator.state = config.rng_state
            else:
                # Copy rng.
                if not rng:
                    raise RuntimeError('both config.rng_state and rng are None.')
                config.rng_state = rng.bit_generator.state
                # Make sure rng_state.setter is overrided.
                assert config.rng_state

        else:
            # Force not passing rng.
            rng = None

        return config, rng

    @staticmethod
    def get_shape_from_shapable_or_shape(shapable_or_shape: Union[Shapable, Tuple[int, int]]):
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
        shape = Distortion.get_shape_from_shapable_or_shape(shapable_or_shape)

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
        return self.func_image(
            internals.config,
            internals.state,
            image,
            internals.rng,
        )

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
        if self.func_score_map:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=score_map,
                rng=rng,
            )
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
        if self.func_mask:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=mask,
                rng=rng,
            )
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
        if self.func_active_mask:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=shapable_or_shape,
                rng=rng,
            )
            return self.func_active_mask(
                internals.config,
                internals.state,
                internals.shape,
                internals.rng,
            )

        else:
            shape = Distortion.get_shape_from_shapable_or_shape(shapable_or_shape)
            mask = Mask.from_shape(shape)
            mask.mat.fill(1)
            return self.distort_mask(
                config_or_config_generator=config_or_config_generator,
                state=state,
                mask=mask,
                rng=rng,
            )

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

        if self.func_point:
            return self.func_point(
                internals.config,
                internals.state,
                internals.shape,
                point,
                internals.rng,
            )

        elif self.func_points:
            return self.func_points(
                internals.config,
                internals.state,
                internals.shape,
                [point],
                internals.rng,
            )[0]

        else:
            # NOP.
            return point

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
        points: Union[PointList, Iterable[Point]],
        state: Optional[_T_STATE] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        points = PointList(points)

        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rng=rng,
        )

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
                new_point = self.distort_point(
                    config_or_config_generator=internals.config,
                    shapable_or_shape=internals.shape,
                    point=point,
                    state=internals.state,
                    rng=None,
                )
                new_points.append(new_point)
            return new_points

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

        if self.func_polygon:
            return self.func_polygon(
                internals.config,
                internals.state,
                internals.shape,
                polygon,
                internals.rng,
            )

        elif self.func_polygons:
            return self.func_polygons(
                internals.config,
                internals.state,
                internals.shape,
                [polygon],
                internals.rng,
            )[0]

        else:
            new_points = self.distort_points(
                config_or_config_generator=internals.config,
                shapable_or_shape=internals.shape,
                state=internals.state,
                points=polygon.points,
                rng=None,
            )
            return Polygon(points=new_points)

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
                new_polygon = self.distort_polygon(
                    config_or_config_generator=internals.config,
                    state=internals.state,
                    shapable_or_shape=internals.shape,
                    polygon=polygon,
                    rng=None,
                )
                new_polygons.append(new_polygon)
            return new_polygons

    @staticmethod
    def initialize_distortion_result(
        shapable_or_shape: Optional[Union[Shapable, Tuple[int, int]]] = None,
        image: Optional[Image] = None,
        mask: Optional[Mask] = None,
        score_map: Optional[ScoreMap] = None,
    ):
        if shapable_or_shape is None:
            shapable_or_shape = image or mask or score_map
        assert shapable_or_shape

        return DistortionResult(
            shape=Distortion.get_shape_from_shapable_or_shape(shapable_or_shape),
        )

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
        points: Optional[PointList] = None,
        polygon: Optional[Polygon] = None,
        polygons: Optional[Iterable[Polygon]] = None,
        text_polygon: Optional[TextPolygon] = None,
        text_polygons: Optional[Iterable[TextPolygon]] = None,
        get_active_mask: bool = False,
        get_config: bool = False,
        get_state: bool = False,
        rng: Optional[RandomGenerator] = None,
    ):
        # yapf: enable
        result = Distortion.initialize_distortion_result(
            shapable_or_shape=shapable_or_shape,
            image=image,
            mask=mask,
            score_map=score_map,
        )

        config, state = self.generate_config_and_state(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=result.shape,
            rng=rng,
        )

        if image:
            result.image = self.distort_image(
                config,
                image,
                state=state,
                rng=rng,
            )

        if mask:
            result.mask = self.distort_mask(
                config,
                mask,
                state=state,
                rng=rng,
            )

        if score_map:
            result.score_map = self.distort_score_map(
                config,
                score_map,
                state=state,
                rng=rng,
            )

        if point:
            result.point = self.distort_point(
                config,
                result.shape,
                point,
                state=state,
                rng=rng,
            )

        if points:
            result.points = self.distort_points(
                config,
                result.shape,
                points,
                state=state,
                rng=rng,
            )

        if polygon:
            result.polygon = self.distort_polygon(
                config,
                result.shape,
                polygon,
                state=state,
                rng=rng,
            )

        if polygons:
            result.polygons = self.distort_polygons(
                config,
                result.shape,
                polygons,
                state=state,
                rng=rng,
            )

        if text_polygon:
            result.text_polygon = attrs.evolve(
                text_polygon,
                polygon=self.distort_polygon(
                    config,
                    result.shape,
                    text_polygon.polygon,
                    state=state,
                    rng=rng,
                ),
            )

        if text_polygons:
            text_polygons = tuple(text_polygons)
            distorted_polygons = self.distort_polygons(
                config,
                result.shape,
                [text_polygon.polygon for text_polygon in text_polygons],
                state=state,
                rng=rng,
            )
            result.text_polygons = [
                attrs.evolve(text_polygon, polygon=distorted_polygon)
                for text_polygon, distorted_polygon in zip(text_polygons, distorted_polygons)
            ]

        if get_active_mask:
            result.active_mask = self.get_active_mask(
                config,
                result.shape,
                state=state,
                rng=rng,
            )

        if get_config:
            result.config = config

        if get_state:
            result.state = state

        return result
