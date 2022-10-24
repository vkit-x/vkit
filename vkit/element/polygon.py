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
from typing import cast, Optional, Tuple, Union, Sequence, Iterable, List
import logging
import math

import attrs
import numpy as np
import cv2 as cv
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    MultiPolygon as ShapelyMultiPolygon,
    CAP_STYLE,
    JOIN_STYLE,
)
from shapely.ops import unary_union
import pyclipper

from vkit.utility import attrs_lazy_field
from .type import Shapable, ElementSetOperationMode

logger = logging.getLogger(__name__)

_T = Union[float, str]


@attrs.define
class PolygonInternals:
    bounding_box: 'Box'
    np_self_relative_points: np.ndarray

    _area: Optional[float] = attrs_lazy_field()
    _self_relative_polygon: Optional['Polygon'] = attrs_lazy_field()
    _np_mask: Optional[np.ndarray] = attrs_lazy_field()
    _mask: Optional['Mask'] = attrs_lazy_field()

    def lazy_post_init_area(self):
        if self._area is not None:
            return self._area

        self._area = float(ShapelyPolygon(self.np_self_relative_points).area)
        return self._area

    @property
    def area(self):
        return self.lazy_post_init_area()

    def lazy_post_init_self_relative_polygon(self):
        if self._self_relative_polygon is not None:
            return self._self_relative_polygon

        self._self_relative_polygon = Polygon.from_np_array(self.np_self_relative_points)
        return self._self_relative_polygon

    @property
    def self_relative_polygon(self):
        return self.lazy_post_init_self_relative_polygon()

    def lazy_post_init_np_mask(self):
        if self._np_mask is not None:
            return self._np_mask

        np_mask = np.zeros(self.bounding_box.shape, dtype=np.uint8)
        cv.fillPoly(np_mask, [self.self_relative_polygon.to_np_array()], 1)
        self._np_mask = np_mask.astype(np.bool8)
        return self._np_mask

    @property
    def np_mask(self):
        return self.lazy_post_init_np_mask()

    def lazy_post_init_mask(self):
        if self._mask is not None:
            return self._mask

        self._mask = Mask(mat=self.np_mask.astype(np.uint8))
        self._mask = self._mask.to_box_attached(self.bounding_box)
        return self._mask

    @property
    def mask(self):
        return self.lazy_post_init_mask()


@attrs.define(frozen=True, eq=False)
class Polygon:
    points: 'PointTuple'

    _internals: Optional[PolygonInternals] = attrs_lazy_field()

    def __attrs_post_init__(self):
        assert self.points

    def lazy_post_init_internals(self):
        if self._internals is not None:
            return self._internals

        np_self_relative_points = self.to_smooth_np_array()

        y_min = np_self_relative_points[:, 1].min()
        y_max = np_self_relative_points[:, 1].max()

        x_min = np_self_relative_points[:, 0].min()
        x_max = np_self_relative_points[:, 0].max()

        np_self_relative_points[:, 0] -= x_min
        np_self_relative_points[:, 1] -= y_min

        bounding_box = Box(
            up=round(y_min),
            down=round(y_max),
            left=round(x_min),
            right=round(x_max),
        )

        object.__setattr__(
            self,
            '_internals',
            PolygonInternals(
                bounding_box=bounding_box,
                np_self_relative_points=np_self_relative_points,
            ),
        )
        return cast(PolygonInternals, self._internals)

    ###############
    # Constructor #
    ###############
    @classmethod
    def create(cls, points: Union['PointList', 'PointTuple', Iterable['Point']]):
        return cls(points=PointTuple(points))

    ############
    # Property #
    ############
    @property
    def num_points(self):
        return len(self.points)

    @property
    def internals(self):
        return self.lazy_post_init_internals()

    @property
    def area(self):
        return self.internals.area

    @property
    def bounding_box(self):
        return self.internals.bounding_box

    @property
    def self_relative_polygon(self):
        return self.internals.self_relative_polygon

    @property
    def mask(self):
        return self.internals.mask

    ##############
    # Conversion #
    ##############
    @classmethod
    def from_xy_pairs(cls, xy_pairs: Iterable[Tuple[_T, _T]]):
        return cls(points=PointTuple.from_xy_pairs(xy_pairs))

    def to_xy_pairs(self):
        return self.points.to_xy_pairs()

    def to_smooth_xy_pairs(self):
        return self.points.to_smooth_xy_pairs()

    @classmethod
    def from_flatten_xy_pairs(cls, flatten_xy_pairs: Sequence[_T]):
        return cls(points=PointTuple.from_flatten_xy_pairs(flatten_xy_pairs))

    def to_flatten_xy_pairs(self):
        return self.points.to_flatten_xy_pairs()

    def to_smooth_flatten_xy_pairs(self):
        return self.points.to_smooth_flatten_xy_pairs()

    @classmethod
    def from_np_array(cls, np_points: np.ndarray):
        return cls(points=PointTuple.from_np_array(np_points))

    def to_np_array(self):
        return self.points.to_np_array()

    def to_smooth_np_array(self):
        return self.points.to_smooth_np_array()

    @classmethod
    def from_shapely_polygon(cls, shapely_polygon: ShapelyPolygon):
        xy_pairs = cls.remove_duplicated_xy_pairs(shapely_polygon.exterior.coords)  # type: ignore
        return cls.from_xy_pairs(xy_pairs)

    def to_shapely_polygon(self):
        return ShapelyPolygon(self.to_xy_pairs())

    def to_smooth_shapely_polygon(self):
        return ShapelyPolygon(self.to_smooth_xy_pairs())

    ############
    # Operator #
    ############
    def get_center_point(self):
        shapely_polygon = self.to_smooth_shapely_polygon()
        centroid = shapely_polygon.centroid
        x, y = centroid.coords[0]
        return Point.create(y=y, x=x)

    def get_rectangular_height(self):
        # See Box.to_polygon.
        assert self.num_points == 4
        (
            point_up_left,
            point_up_right,
            point_down_right,
            point_down_left,
        ) = self.points
        left_side_height = math.hypot(
            point_up_left.smooth_y - point_down_left.smooth_y,
            point_up_left.smooth_x - point_down_left.smooth_x,
        )
        right_side_height = math.hypot(
            point_up_right.smooth_y - point_down_right.smooth_y,
            point_up_right.smooth_x - point_down_right.smooth_x,
        )
        return (left_side_height + right_side_height) / 2

    def get_rectangular_width(self):
        # See Box.to_polygon.
        assert self.num_points == 4
        (
            point_up_left,
            point_up_right,
            point_down_right,
            point_down_left,
        ) = self.points
        up_side_width = math.hypot(
            point_up_left.smooth_y - point_up_right.smooth_y,
            point_up_left.smooth_x - point_up_right.smooth_x,
        )
        down_side_width = math.hypot(
            point_down_left.smooth_y - point_down_right.smooth_y,
            point_down_left.smooth_x - point_down_right.smooth_x,
        )
        return (up_side_width + down_side_width) / 2

    def to_clipped_points(self, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        return self.points.to_clipped_points(shapable_or_shape)

    def to_clipped_polygon(self, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        return Polygon(points=self.to_clipped_points(shapable_or_shape))

    def to_shifted_points(self, offset_y: int = 0, offset_x: int = 0):
        return self.points.to_shifted_points(offset_y=offset_y, offset_x=offset_x)

    def to_relative_points(self, origin_y: int, origin_x: int):
        return self.points.to_relative_points(origin_y=origin_y, origin_x=origin_x)

    def to_shifted_polygon(self, offset_y: int = 0, offset_x: int = 0):
        return Polygon(points=self.to_shifted_points(offset_y=offset_y, offset_x=offset_x))

    def to_relative_polygon(self, origin_y: int, origin_x: int):
        return Polygon(points=self.to_relative_points(origin_y=origin_y, origin_x=origin_x))

    def to_conducted_resized_polygon(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return Polygon(
            points=self.points.to_conducted_resized_points(
                shapable_or_shape=shapable_or_shape,
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )

    def to_resized_polygon(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return self.to_conducted_resized_polygon(
            shapable_or_shape=self.bounding_box.shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def to_bounding_rectangular_polygon(self, shape: Tuple[int, int]):
        shapely_polygon = self.to_smooth_shapely_polygon()

        assert isinstance(shapely_polygon.minimum_rotated_rectangle, ShapelyPolygon)
        polygon = self.from_shapely_polygon(shapely_polygon.minimum_rotated_rectangle)
        assert polygon.num_points == 4

        # NOTE: Could be out-of-bound.
        polygon = polygon.to_clipped_polygon(shape)

        return polygon

    def to_bounding_box(self):
        return self.bounding_box

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.mask.fill_np_array(
            mat=mat,
            value=value,
            alpha=alpha,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_mask(self, mask: 'Mask'):
        return self.mask.extract_mask(mask)

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.mask.fill_mask(
            mask=mask,
            value=value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        return self.mask.extract_score_map(score_map)

    def fill_score_map(
        self,
        score_map: 'ScoreMap',
        value: Union['ScoreMap', np.ndarray, float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        self.mask.fill_score_map(
            score_map=score_map,
            value=value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_image(self, image: 'Image'):
        return self.mask.extract_image(image)

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        self.mask.fill_image(
            image=image,
            value=value,
            alpha=alpha,
        )

    @classmethod
    def remove_duplicated_xy_pairs(cls, xy_pairs: Sequence[Tuple[int, int]]):
        xy_pairs = tuple(map(tuple, xy_pairs))
        unique_xy_pairs = []

        idx = 0
        while idx < len(xy_pairs):
            unique_xy_pairs.append(xy_pairs[idx])

            next_idx = idx + 1
            while next_idx < len(xy_pairs) and xy_pairs[idx] == xy_pairs[next_idx]:
                next_idx += 1
            idx = next_idx

        # Check head & tail.
        if len(unique_xy_pairs) > 1 and unique_xy_pairs[0] == unique_xy_pairs[-1]:
            unique_xy_pairs.pop()

        assert len(unique_xy_pairs) >= 3
        return unique_xy_pairs

    def to_vatti_clipped_polygon(self, ratio: float, shrink: bool):
        assert 0.0 <= ratio <= 1.0
        if ratio == 1.0:
            return self, 0.0

        xy_pairs = self.to_smooth_xy_pairs()

        shapely_polygon = ShapelyPolygon(xy_pairs)
        if shapely_polygon.area == 0:
            logger.warning('shapely_polygon.area == 0, this breaks vatti_clip.')

        distance: float = shapely_polygon.area * (1 - np.power(ratio, 2)) / shapely_polygon.length
        if shrink:
            distance *= -1

        clipper = pyclipper.PyclipperOffset()  # type: ignore
        clipper.AddPath(xy_pairs, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # type: ignore

        clipped_paths = clipper.Execute(distance)
        assert clipped_paths
        clipped_path: Sequence[Tuple[int, int]] = clipped_paths[0]

        clipped_xy_pairs = self.remove_duplicated_xy_pairs(clipped_path)
        clipped_polygon = self.from_xy_pairs(clipped_xy_pairs)

        return clipped_polygon, distance

    def to_shrank_polygon(
        self,
        ratio: float,
        no_exception: bool = True,
    ):
        try:
            shrank_polygon, _ = self.to_vatti_clipped_polygon(ratio, shrink=True)

            shrank_bounding_box = shrank_polygon.bounding_box
            vert_contains = (
                self.bounding_box.up <= shrank_bounding_box.up
                and shrank_bounding_box.down <= self.bounding_box.down
            )
            hori_contains = (
                self.bounding_box.left <= shrank_bounding_box.left
                and shrank_bounding_box.right <= self.bounding_box.right
            )
            if not (shrank_bounding_box.valid and vert_contains and hori_contains):
                logger.warning('Invalid shrank_polygon bounding box. Fallback to NOP.')
                return self

            if 0 < shrank_polygon.area <= self.area:
                return shrank_polygon
            else:
                logger.warning('Invalid shrank_polygon.area. Fallback to NOP.')
                return self

        except Exception:
            if no_exception:
                logger.exception('Failed to shrink. Fallback to NOP.')
                return self
            else:
                raise

    def to_dilated_polygon(
        self,
        ratio: float,
        no_exception: bool = True,
    ):
        try:
            dilated_polygon, _ = self.to_vatti_clipped_polygon(ratio, shrink=False)

            dilated_bounding_box = dilated_polygon.bounding_box
            vert_contains = (
                dilated_bounding_box.up <= self.bounding_box.up
                and self.bounding_box.down <= dilated_bounding_box.down
            )
            hori_contains = (
                dilated_bounding_box.left <= self.bounding_box.left
                and self.bounding_box.right <= dilated_bounding_box.right
            )
            if not (dilated_bounding_box.valid and vert_contains and hori_contains):
                logger.warning('Invalid dilated_polygon bounding box. Fallback to NOP.')
                return self

            if dilated_polygon.area >= self.area:
                return dilated_polygon
            else:
                logger.warning('Invalid dilated_polygon.area. Fallback to NOP.')
                return self

        except Exception:
            if no_exception:
                logger.exception('Failed to dilate. Fallback to NOP.')
                return self
            else:
                raise


# Experimental operations.
# TODO: Might add to Polygon class.
def get_line_lengths(shapely_polygon: ShapelyPolygon):
    assert shapely_polygon.exterior is not None
    points = tuple(shapely_polygon.exterior.coords)
    for idx, p0 in enumerate(points):
        p1 = points[(idx + 1) % len(points)]
        length = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        yield length


def estimate_shapely_polygon_height(shapely_polygon: ShapelyPolygon):
    length = max(get_line_lengths(shapely_polygon))
    return float(shapely_polygon.area) / length


def calculate_patch_buffer_eps(shapely_polygon: ShapelyPolygon):
    return estimate_shapely_polygon_height(shapely_polygon) / 10


def patch_unionized_unionized_shapely_polygon(unionized_shapely_polygon: ShapelyPolygon):
    eps = calculate_patch_buffer_eps(unionized_shapely_polygon)
    unionized_shapely_polygon = unionized_shapely_polygon.buffer(
        eps,
        cap_style=CAP_STYLE.round,
        join_style=JOIN_STYLE.round,
    )  # type: ignore
    unionized_shapely_polygon = unionized_shapely_polygon.buffer(
        -eps,
        cap_style=CAP_STYLE.round,
        join_style=JOIN_STYLE.round,
    )  # type: ignore
    return unionized_shapely_polygon


def unionize_polygons(polygons: Iterable[Polygon]):
    shapely_polygons = [polygon.to_smooth_shapely_polygon() for polygon in polygons]

    unionized_shapely_polygons = []

    # Patch unary_union.
    unary_union_output = unary_union(shapely_polygons)
    if not isinstance(unary_union_output, ShapelyMultiPolygon):
        assert isinstance(unary_union_output, ShapelyPolygon)
        unary_union_output = [unary_union_output]

    for unionized_shapely_polygon in unary_union_output:
        unionized_shapely_polygon = patch_unionized_unionized_shapely_polygon(
            unionized_shapely_polygon
        )
        unionized_shapely_polygons.append(unionized_shapely_polygon)

    unionized_polygons = [
        Polygon.from_xy_pairs(unionized_shapely_polygon.exterior.coords)
        for unionized_shapely_polygon in unionized_shapely_polygons
    ]

    scatter_indices: List[int] = []
    for shapely_polygon in shapely_polygons:
        best_unionized_polygon_idx = None
        best_area = 0.0
        conflict = False

        for unionized_polygon_idx, unionized_shapely_polygon in enumerate(
            unionized_shapely_polygons
        ):
            if not unionized_shapely_polygon.intersects(shapely_polygon):
                continue
            area = unionized_shapely_polygon.intersection(shapely_polygon).area
            if area > best_area:
                best_area = area
                best_unionized_polygon_idx = unionized_polygon_idx
                conflict = False
            elif area == best_area:
                conflict = True

        assert not conflict
        assert best_unionized_polygon_idx is not None
        scatter_indices.append(best_unionized_polygon_idx)

    return unionized_polygons, scatter_indices


def generate_fill_by_polygons_mask(
    shape: Tuple[int, int],
    polygons: Iterable[Polygon],
    mode: ElementSetOperationMode,
):
    if mode == ElementSetOperationMode.UNION:
        return None
    else:
        return Mask.from_polygons(shape, polygons, mode)


# Cyclic dependency, by design.
from .point import Point, PointList, PointTuple  # noqa: E402
from .box import Box  # noqa: E402
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
