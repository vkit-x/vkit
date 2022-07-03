from typing import Optional, Sequence, Tuple, Union, Iterable, TypeVar, Type
import math

import attrs
import numpy as np
from numpy.random import Generator as RandomGenerator
import cv2 as cv

from vkit.element import (
    Image,
    ScoreMap,
    Mask,
    Point,
    PointList,
    Polygon,
)
from ..interface import (
    DistortionConfig,
    DistortionState,
    Distortion,
)


def affine_mat(trans_mat: np.ndarray, dsize: Tuple[int, int], mat: np.ndarray) -> np.ndarray:
    if trans_mat.shape[0] == 2:
        return cv.warpAffine(mat, trans_mat, dsize)
    else:
        assert trans_mat.shape[0] == 3
        return cv.warpPerspective(mat, trans_mat, dsize)


def affine_np_points(trans_mat: np.ndarray, np_points: np.ndarray) -> np.ndarray:
    # (2, *)
    np_points = np_points.transpose()
    # (3, *)
    np_points = np.concatenate((
        np_points,
        np.ones((1, np_points.shape[1]), dtype=np.int32),
    ))

    new_np_points = np.matmul(trans_mat, np_points)
    if trans_mat.shape[0] == 2:
        # new_np_points.shape is (2, *), do nothing.
        pass

    else:
        assert trans_mat.shape[0] == 3
        new_np_points = new_np_points[:2, :] / new_np_points[2, :]

    return new_np_points.transpose()


def affine_points(trans_mat: np.ndarray, points: PointList):
    new_np_points = affine_np_points(trans_mat, points.to_np_array())
    return PointList.from_np_array(new_np_points)


def affine_polygons(trans_mat: np.ndarray, polygons: Sequence[Polygon]) -> Sequence[Polygon]:
    points_ranges = []
    points = PointList()
    for polygon in polygons:
        points_ranges.append((len(points), len(points) + len(polygon.points)))
        points.extend(polygon.points)

    new_np_points = affine_np_points(trans_mat, points.to_np_array())
    new_polygons = []
    for begin, end in points_ranges:
        new_polygons.append(Polygon.from_xy_pairs(new_np_points[begin:end]))

    return new_polygons


@attrs.define
class ShearHoriConfig(DistortionConfig):
    # angle: int, (-90, 90), positive value for rightward direction.
    angle: int

    @property
    def is_nop(self):
        return self.angle == 0


class ShearHoriState(DistortionState[ShearHoriConfig]):

    def __init__(
        self,
        config: ShearHoriConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        tan_phi = math.tan(math.radians(config.angle))

        height, width = shape
        shift_x = abs(height * tan_phi)
        self.dsize = (math.ceil(width + shift_x), height)

        if config.angle < 0:
            # Shear left & the negative part.
            self.trans_mat = np.array([
                (1, -tan_phi, 0),
                (0, 1, 0),
            ], dtype=np.float32)

        elif config.angle > 0:
            # Shear right.
            self.trans_mat = np.array([
                (1, -tan_phi, shift_x),
                (0, 1, 0),
            ], dtype=np.float32)

        else:
            # No need to transform.
            self.trans_mat = None
            self.dsize = None


@attrs.define
class ShearVertConfig(DistortionConfig):
    # angle: int, (-90, 90), positive value for downward direction.
    angle: int

    @property
    def is_nop(self):
        return self.angle == 0


class ShearVertState(DistortionState[ShearVertConfig]):

    def __init__(
        self,
        config: ShearVertConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        tan_abs_phi = math.tan(math.radians(abs(config.angle)))

        height, width = shape
        shift_y = width * tan_abs_phi
        self.dsize = (width, math.ceil(height + shift_y))

        if config.angle < 0:
            # Shear up.
            self.trans_mat = np.array(
                [
                    (1, 0, 0),
                    (-tan_abs_phi, 1, shift_y),
                ],
                dtype=np.float32,
            )
        elif config.angle > 0:
            # Shear down & the negative part.
            self.trans_mat = np.array(
                [
                    (1, 0, 0),
                    (tan_abs_phi, 1, 0),
                ],
                dtype=np.float32,
            )
        else:
            # No need to transform.
            self.trans_mat = None
            self.dsize = None


@attrs.define
class RotateConfig(DistortionConfig):
    # angle: int, [0, 360], clockwise angle.
    angle: int

    @property
    def is_nop(self):
        return self.angle == 0


class RotateState(DistortionState[RotateConfig]):

    def __init__(
        self,
        config: RotateConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape

        angle = config.angle % 360
        rad = math.radians(angle)

        shift_x = 0
        shift_y = 0

        if rad <= math.pi / 2:
            # 3-4 quadrant.
            shift_x = height * math.sin(rad)

            dst_width = height * math.sin(rad) + width * math.cos(rad)
            dst_height = height * math.cos(rad) + width * math.sin(rad)

        elif rad <= math.pi:
            # 2-3 quadrant.
            shift_rad = rad - math.pi / 2

            shift_x = width * math.sin(shift_rad) + height * math.cos(shift_rad)
            shift_y = height * math.sin(shift_rad)

            dst_width = shift_x
            dst_height = shift_y + width * math.cos(shift_rad)

        elif rad < math.pi * 3 / 2:
            # 1-2 quadrant.
            shift_rad = rad - math.pi

            shift_x = width * math.cos(shift_rad)
            shift_y = width * math.sin(shift_rad) + height * math.cos(shift_rad)

            dst_width = shift_x + height * math.sin(shift_rad)
            dst_height = shift_y

        else:
            # 1-4 quadrant.
            shift_rad = rad - math.pi * 3 / 2

            shift_y = width * math.cos(shift_rad)

            dst_width = width * math.sin(shift_rad) + height * math.cos(shift_rad)
            dst_height = shift_y + height * math.sin(shift_rad)

        shift_x = math.ceil(shift_x)
        shift_y = math.ceil(shift_y)

        self.trans_mat = np.array(
            [
                (math.cos(rad), -math.sin(rad), shift_x),
                (math.sin(rad), math.cos(rad), shift_y),
            ],
            dtype=np.float32,
        )

        self.dsize = (math.ceil(dst_width), math.ceil(dst_height))


@attrs.define
class SkewHoriConfig(DistortionConfig):
    # (-1.0, 0.0], shrink the left side.
    # [0.0, 1.0), shrink the right side.
    # The larger abs(ratio), the more to shrink.
    ratio: float

    @property
    def is_nop(self):
        return self.ratio == 0


class SkewHoriState(DistortionState[SkewHoriConfig]):

    def __init__(
        self,
        config: SkewHoriConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape

        src_xy_pairs = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]

        shrink_size = round(height * abs(config.ratio))
        shrink_up = shrink_size // 2
        shrink_down = shrink_size - shrink_up

        if config.ratio < 0:
            dst_xy_pairs = [
                (0, shrink_up),
                (width - 1, 0),
                (width - 1, height - 1),
                (0, height - shrink_down - 1),
            ]
        else:
            dst_xy_pairs = [
                (0, 0),
                (width - 1, shrink_up),
                (width - 1, height - shrink_down - 1),
                (0, height - 1),
            ]

        self.trans_mat = cv.getPerspectiveTransform(
            np.array(src_xy_pairs, dtype=np.float32),
            np.array(dst_xy_pairs, dtype=np.float32),
            cv.DECOMP_SVD,
        )
        self.dsize = (width, height)


@attrs.define
class SkewVertConfig(DistortionConfig):
    # (-1.0, 0.0], shrink the up side.
    # [0.0, 1.0), shrink the down side.
    # The larger abs(ratio), the more to shrink.
    ratio: float

    @property
    def is_nop(self):
        return self.ratio == 0


class SkewVertState(DistortionState[SkewVertConfig]):

    def __init__(
        self,
        config: SkewVertConfig,
        shape: Tuple[int, int],
        rng: Optional[RandomGenerator],
    ):
        height, width = shape

        src_xy_pairs = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]

        shrink_size = round(width * abs(config.ratio))
        shrink_left = shrink_size // 2
        shrink_right = shrink_size - shrink_left

        if config.ratio < 0:
            dst_xy_pairs = [
                (shrink_left, 0),
                (width - shrink_right - 1, 0),
                (width - 1, height - 1),
                (0, height - 1),
            ]
        else:
            dst_xy_pairs = [
                (0, 0),
                (width - 1, 0),
                (width - shrink_right - 1, height - 1),
                (shrink_right, height - 1),
            ]

        self.trans_mat = cv.getPerspectiveTransform(
            np.array(src_xy_pairs, dtype=np.float32),
            np.array(dst_xy_pairs, dtype=np.float32),
            cv.DECOMP_SVD,
        )
        self.dsize = (width, height)


_T_AFFINE_CONFIG = TypeVar(
    '_T_AFFINE_CONFIG',
    ShearHoriConfig,
    ShearVertConfig,
    RotateConfig,
    SkewHoriConfig,
    SkewVertConfig,
)
_T_AFFINE_STATE = TypeVar(
    '_T_AFFINE_STATE',
    ShearHoriState,
    ShearVertState,
    RotateState,
    SkewHoriState,
    SkewVertState,
)


def affine_trait_func_mat(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    mat: np.ndarray,
):
    assert state
    if config.is_nop:
        return mat
    else:
        assert state.trans_mat is not None
        assert state.dsize is not None
        return affine_mat(state.trans_mat, state.dsize, mat)


def affine_trait_func_image(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    image: Image,
    rng: Optional[RandomGenerator],
):
    return Image(mat=affine_trait_func_mat(config, state, image.mat))


def affine_trait_func_score_map(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    score_map: ScoreMap,
    rng: Optional[RandomGenerator],
):
    assert state
    return ScoreMap(mat=affine_trait_func_mat(config, state, score_map.mat))


def affine_trait_func_mask(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    mask: Mask,
    rng: Optional[RandomGenerator],
):
    assert state
    return Mask(mat=affine_trait_func_mat(config, state, mask.mat))


def affine_trait_func_points(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    shape: Tuple[int, int],
    points: Union[PointList, Iterable[Point]],
    rng: Optional[RandomGenerator],
):
    assert state
    points = PointList(points)
    if config.is_nop:
        return points
    else:
        assert state.trans_mat is not None
        return affine_points(state.trans_mat, points)


def affine_trait_func_polygons(
    config: _T_AFFINE_CONFIG,
    state: Optional[_T_AFFINE_STATE],
    shape: Tuple[int, int],
    polygons: Iterable[Polygon],
    rng: Optional[RandomGenerator],
):
    assert state
    polygons = tuple(polygons)
    if config.is_nop:
        return polygons
    else:
        assert state.trans_mat is not None
        return affine_polygons(state.trans_mat, polygons)


class DistortionAffine(Distortion[_T_AFFINE_CONFIG, _T_AFFINE_STATE]):

    def __init__(
        self,
        config_cls: Type[_T_AFFINE_CONFIG],
        state_cls: Type[_T_AFFINE_STATE],
    ):
        super().__init__(
            config_cls=config_cls,
            state_cls=state_cls,
            func_image=affine_trait_func_image,
            func_mask=affine_trait_func_mask,
            func_score_map=affine_trait_func_score_map,
            func_points=affine_trait_func_points,
            func_polygons=affine_trait_func_polygons,
        )


shear_hori = DistortionAffine(
    config_cls=ShearHoriConfig,
    state_cls=ShearHoriState,
)

shear_vert = DistortionAffine(
    config_cls=ShearVertConfig,
    state_cls=ShearVertState,
)

rotate = DistortionAffine(
    config_cls=RotateConfig,
    state_cls=RotateState,
)

skew_hori = DistortionAffine(
    config_cls=SkewHoriConfig,
    state_cls=SkewHoriState,
)

skew_vert = DistortionAffine(
    config_cls=SkewVertConfig,
    state_cls=SkewVertState,
)
