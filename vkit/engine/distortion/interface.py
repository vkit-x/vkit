from typing import (
    cast,
    get_origin,
    Any,
    Callable,
    Dict,
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
from numpy.random import RandomState

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
    def supports_rnd_state(self) -> bool:
        return False

    @property
    def rnd_state(self) -> Optional[Dict[str, Any]]:
        return None

    @rnd_state.setter
    def rnd_state(self, val: Dict[str, Any]):
        pass


_T_CONFIG = TypeVar('_T_CONFIG', bound=DistortionConfig)


class DistortionState(Generic[_T_CONFIG]):

    def __init__(self, config: _T_CONFIG, shape: Tuple[int, int], rnd: Optional[RandomState]):
        raise NotImplementedError()


class DistortionNopState(DistortionState[_T_CONFIG]):

    def __init__(self, config: _T_CONFIG, shape: Tuple[int, int], rnd: Optional[RandomState]):
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
    meta: Optional[Dict[str, Any]] = None


@attrs.define
class DistortionInternals(Generic[_T_CONFIG, _T_STATE]):
    config: _T_CONFIG
    state: Optional[_T_STATE]
    shape: Tuple[int, int]
    rnd: Optional[RandomState]


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
                Optional[RandomState],
            ],
            Image,
        ],
        func_mask: Optional[
            Callable[
                [
                    _T_CONFIG,
                    Optional[_T_STATE],
                    Mask,
                    Optional[RandomState],
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
                    Optional[RandomState],
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
                    Optional[RandomState],
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
                    Optional[RandomState],
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
                    Optional[RandomState],
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
                    Optional[RandomState],
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
                    Optional[RandomState],
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
    def prepare_config_and_rnd(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shape: Tuple[int, int],
        rnd: Optional[RandomState],
    ) -> Tuple[_T_CONFIG, Optional[RandomState]]:
        # yapf: enable
        if callable(config_or_config_generator):
            config_generator = cast(
                Callable[[Tuple[int, int], RandomState], Union[_T_CONFIG, Dict[str, Any]]],
                config_or_config_generator,
            )
            if not rnd:
                raise RuntimeError('config_generator but rnd is None.')
            config = dyn_structure(config_generator(shape, rnd), self.config_cls)

        else:
            config_or_config_generator = cast(
                Union[_T_CONFIG, Dict[str, Any]],
                config_or_config_generator,
            )
            config = dyn_structure(config_or_config_generator, self.config_cls)

        if config.supports_rnd_state:
            # Process rnd_state.
            if config.rnd_state:
                # Create/replace rnd.
                rnd = RandomState()
                rnd.set_state(config.rnd_state)
            else:
                # Copy rnd.
                if not rnd:
                    raise RuntimeError('both config.rnd_state and rnd are None.')
                config.rnd_state = rnd.get_state()
                # Make sure rnd_state.setter is overrided.
                assert config.rnd_state

        else:
            # Force not passing rnd.
            rnd = None

        return config, rnd

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
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        state: Optional[_T_STATE],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rnd: Optional[RandomState] = None,
        disable_state_initialization: bool = False,
    ):
        # yapf: enable
        shape = Distortion.get_shape_from_shapable_or_shape(shapable_or_shape)

        config, rnd = self.prepare_config_and_rnd(
            config_or_config_generator,
            shape,
            rnd,
        )

        if get_origin(self.state_cls) is not DistortionNopState:
            if state is None and not disable_state_initialization:
                state = self.state_cls(config, shape, rnd)
        else:
            state = None

        return DistortionInternals(config, state, shape, rnd)

    # yapf: disable
    def generate_config_and_state(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        state: Optional[_T_STATE],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )
        return internals.config, internals.state

    # yapf: disable
    def generate_config(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
            disable_state_initialization=True,
        )
        return internals.config

    # yapf: disable
    def generate_state(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=None,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )
        return internals.state

    # yapf: disable
    def distort_image(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        image: Image,
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=image,
            rnd=rnd,
        )
        return self.func_image(
            internals.config,
            internals.state,
            image,
            internals.rnd,
        )

    # yapf: disable
    def distort_score_map(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        score_map: ScoreMap,
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        if self.func_score_map:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=score_map,
                rnd=rnd,
            )
            return self.func_score_map(
                internals.config,
                internals.state,
                score_map,
                internals.rnd,
            )

        else:
            # NOP.
            return score_map

    # yapf: disable
    def distort_mask(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        mask: Mask,
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        if self.func_mask:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=mask,
                rnd=rnd,
            )
            return self.func_mask(
                internals.config,
                internals.state,
                mask,
                internals.rnd,
            )

        else:
            # NOP.
            return mask

    # yapf: disable
    def get_active_mask(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        if self.func_active_mask:
            internals = self.prepare_internals(
                config_or_config_generator=config_or_config_generator,
                state=state,
                shapable_or_shape=shapable_or_shape,
                rnd=rnd,
            )
            return self.func_active_mask(
                internals.config,
                internals.state,
                internals.shape,
                internals.rnd,
            )

        else:
            shape = Distortion.get_shape_from_shapable_or_shape(shapable_or_shape)
            mask = Mask.from_shape(shape)
            mask.mat.fill(1)
            return self.distort_mask(
                config_or_config_generator=config_or_config_generator,
                state=state,
                mask=mask,
                rnd=rnd,
            )

    # yapf: disable
    def distort_point(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        point: Point,
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )

        if self.func_point:
            return self.func_point(
                internals.config,
                internals.state,
                internals.shape,
                point,
                internals.rnd,
            )

        elif self.func_points:
            return self.func_points(
                internals.config,
                internals.state,
                internals.shape,
                [point],
                internals.rnd,
            )[0]

        else:
            # NOP.
            return point

    # yapf: disable
    def distort_points(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        points: Union[PointList, Iterable[Point]],
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        points = PointList(points)

        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )

        if self.func_points:
            return self.func_points(
                internals.config,
                internals.state,
                internals.shape,
                points,
                internals.rnd,
            )

        else:
            new_points = PointList()
            for point in points:
                new_point = self.distort_point(
                    config_or_config_generator=internals.config,
                    shapable_or_shape=internals.shape,
                    point=point,
                    state=internals.state,
                    rnd=None,
                )
                new_points.append(new_point)
            return new_points

    # yapf: disable
    def distort_polygon(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        polygon: Polygon,
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )

        if self.func_polygon:
            return self.func_polygon(
                internals.config,
                internals.state,
                internals.shape,
                polygon,
                internals.rnd,
            )

        elif self.func_polygons:
            return self.func_polygons(
                internals.config,
                internals.state,
                internals.shape,
                [polygon],
                internals.rnd,
            )[0]

        else:
            new_points = self.distort_points(
                config_or_config_generator=internals.config,
                shapable_or_shape=internals.shape,
                state=internals.state,
                points=polygon.points,
                rnd=None,
            )
            return Polygon(points=new_points)

    # yapf: disable
    def distort_polygons(
        self,
        config_or_config_generator: Union[
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
            ],
        ],
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        polygons: Iterable[Polygon],
        state: Optional[_T_STATE] = None,
        rnd: Optional[RandomState] = None,
    ):
        # yapf: enable
        internals = self.prepare_internals(
            config_or_config_generator=config_or_config_generator,
            state=state,
            shapable_or_shape=shapable_or_shape,
            rnd=rnd,
        )

        if self.func_polygons:
            return self.func_polygons(
                internals.config,
                internals.state,
                internals.shape,
                polygons,
                internals.rnd,
            )

        else:
            new_polygons = []
            for polygon in polygons:
                new_polygon = self.distort_polygon(
                    config_or_config_generator=internals.config,
                    state=internals.state,
                    shapable_or_shape=internals.shape,
                    polygon=polygon,
                    rnd=None,
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
            Union[_T_CONFIG, Dict[str, Any]],
            Callable[
                [Tuple[int, int], RandomState],
                Union[_T_CONFIG, Dict[str, Any]],
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
        rnd: Optional[RandomState] = None,
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
            rnd=rnd,
        )

        if image:
            result.image = self.distort_image(
                config,
                image,
                state=state,
                rnd=rnd,
            )

        if mask:
            result.mask = self.distort_mask(
                config,
                mask,
                state=state,
                rnd=rnd,
            )

        if score_map:
            result.score_map = self.distort_score_map(
                config,
                score_map,
                state=state,
                rnd=rnd,
            )

        if point:
            result.point = self.distort_point(
                config,
                result.shape,
                point,
                state=state,
                rnd=rnd,
            )

        if points:
            result.points = self.distort_points(
                config,
                result.shape,
                points,
                state=state,
                rnd=rnd,
            )

        if polygon:
            result.polygon = self.distort_polygon(
                config,
                result.shape,
                polygon,
                state=state,
                rnd=rnd,
            )

        if polygons:
            result.polygons = self.distort_polygons(
                config,
                result.shape,
                polygons,
                state=state,
                rnd=rnd,
            )

        if text_polygon:
            result.text_polygon = attrs.evolve(
                text_polygon,
                polygon=self.distort_polygon(
                    config,
                    result.shape,
                    text_polygon.polygon,
                    state=state,
                    rnd=rnd,
                ),
            )

        if text_polygons:
            text_polygons = tuple(text_polygons)
            distorted_polygons = self.distort_polygons(
                config,
                result.shape,
                [text_polygon.polygon for text_polygon in text_polygons],
                state=state,
                rnd=rnd,
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
                rnd=rnd,
            )

        if get_config:
            result.config = config

        if get_state:
            result.state = state

        return result
