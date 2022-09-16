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
    MultiPoint as ShapelyMultiPoint,
    Polygon as ShapelyPolygon,
    MultiPolygon as ShapelyMultiPolygon,
    CAP_STYLE,
    JOIN_STYLE,
)
from shapely.ops import unary_union
import pyclipper

from .type import Shapable, FillByElementsMode
from .opt import fill_np_array

logger = logging.getLogger(__name__)

T_VAL = Union[float, str]


@attrs.define
class PolygonFillNpArrayInternals:
    bounding_box: 'Box'
    shifted_np_points: np.ndarray

    _np_mask: Optional[np.ndarray] = attrs.field(default=None, repr=False)

    def lazy_post_init_np_mask(self):
        if self._np_mask is not None:
            return self._np_mask

        np_mask = np.zeros(self.bounding_box.shape, dtype=np.uint8)
        cv.fillPoly(np_mask, [self.shifted_np_points], 1)
        self._np_mask = np_mask.astype(np.bool8)
        return self._np_mask

    def get_shifted_polygon(self):
        return Polygon.from_np_array(self.shifted_np_points)

    def get_np_mask(self):
        return self.lazy_post_init_np_mask()


@attrs.define(frozen=True, eq=False)
class Polygon:
    points: 'PointTuple'

    _area: Optional[int] = attrs.field(default=None, repr=False)
    _fill_np_array_internals: Optional[PolygonFillNpArrayInternals] = \
        attrs.field(default=None, repr=False)

    def __attrs_post_init__(self):
        assert self.points

    def lazy_post_init_area(self):
        if self._area is not None:
            return self._area

        object.__setattr__(
            self,
            '_area',
            ShapelyPolygon(self.to_xy_pairs()).area,
        )
        return cast(int, self._area)

    def lazy_post_init_fill_np_array_internals(self):
        if self._fill_np_array_internals is not None:
            return self._fill_np_array_internals

        np_points = self.to_np_array()

        x_min = int(np_points[:, 0].min())
        y_min = int(np_points[:, 1].min())

        x_max = int(np_points[:, 0].max())
        y_max = int(np_points[:, 1].max())

        bounding_box = Box(up=y_min, down=y_max, left=x_min, right=x_max)

        np_points[:, 0] -= x_min
        np_points[:, 1] -= y_min

        object.__setattr__(
            self,
            '_fill_np_array_internals',
            PolygonFillNpArrayInternals(
                bounding_box=bounding_box,
                shifted_np_points=np_points,
            ),
        )
        return cast(PolygonFillNpArrayInternals, self._fill_np_array_internals)

    ###############
    # Constructor #
    ###############
    @staticmethod
    def create(points: Union['PointList', 'PointTuple', Iterable['Point']]):
        return Polygon(points=PointTuple(points))

    ############
    # Property #
    ############
    @property
    def area(self):
        return self.lazy_post_init_area()

    @property
    def fill_np_array_internals(self):
        return self.lazy_post_init_fill_np_array_internals()

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_xy_pairs(xy_pairs: Iterable[Tuple[T_VAL, T_VAL]]):
        return Polygon(points=PointTuple.from_xy_pairs(xy_pairs))

    def to_xy_pairs(self):
        return self.points.to_xy_pairs()

    @staticmethod
    def from_flatten_xy_pairs(flatten_xy_pairs: Sequence[T_VAL]):
        return Polygon(points=PointTuple.from_flatten_xy_pairs(flatten_xy_pairs))

    def to_flatten_xy_pairs(self):
        return self.points.to_flatten_xy_pairs()

    @staticmethod
    def from_np_array(np_points: np.ndarray):
        return Polygon(points=PointTuple.from_np_array(np_points))

    def to_np_array(self):
        return self.points.to_np_array()

    @staticmethod
    def from_shapely_polygon(shapely_polygon: ShapelyPolygon):
        xy_pairs = \
            Polygon.remove_duplicated_xy_pairs(shapely_polygon.exterior.coords)  # type: ignore
        return Polygon.from_xy_pairs(xy_pairs)

    def to_shapely_polygon(self):
        return ShapelyPolygon(self.to_xy_pairs())

    ############
    # Operator #
    ############
    def get_center_point(self):
        xy_pairs = self.to_xy_pairs()

        shapely_polygon = ShapelyPolygon(xy_pairs)
        centroid = shapely_polygon.centroid
        x, y = centroid.coords[0]
        return Point(y=round(y), x=round(x))

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
            shapable_or_shape=self.fill_np_array_internals.bounding_box.shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def to_bounding_rectangular_polygon(self):
        shapely_polygon = ShapelyMultiPoint(self.to_xy_pairs()).minimum_rotated_rectangle
        np_points = np.asarray(shapely_polygon.exterior.coords).astype(np.int32)  # type: ignore
        points = PointList.from_np_array(np_points)
        assert len(points) == 4
        return self.create(points=points)

    def to_bounding_box(self):
        return self.fill_np_array_internals.bounding_box

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        alpha: Union[np.ndarray, float] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        mat, value = self.fill_np_array_internals.bounding_box.prep_mat_and_value(mat, value)
        np_mask = self.fill_np_array_internals.get_np_mask()
        fill_np_array(
            mat=mat,
            value=value,
            np_mask=np_mask,
            alpha=alpha,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_mask(self, mask: 'Mask'):
        extracted_mask = self.fill_np_array_internals.bounding_box.extract_mask(mask)
        extracted_mask = extracted_mask.copy()

        polygon_mask = Mask.from_shapable(extracted_mask)
        shifted_polygon = self.fill_np_array_internals.get_shifted_polygon()
        shifted_polygon.fill_mask(polygon_mask)
        polygon_mask.to_inverted_mask().fill_mask(extracted_mask, value=0)

        return extracted_mask

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        assert mask.box is None

        if isinstance(value, Mask):
            value = value.mat

        with mask.writable_context:
            self.fill_np_array(
                mask.mat,
                value,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def to_mask(self):
        return Mask(
            mat=self.fill_np_array_internals.get_np_mask().astype(np.uint8),
            box=self.fill_np_array_internals.bounding_box,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        extracted_score_map = self.fill_np_array_internals.bounding_box.extract_score_map(score_map)
        extracted_score_map = extracted_score_map.copy()

        polygon_mask = Mask.from_shapable(extracted_score_map)
        shifted_polygon = self.fill_np_array_internals.get_shifted_polygon()
        shifted_polygon.fill_mask(polygon_mask)
        polygon_mask.to_inverted_mask().fill_score_map(extracted_score_map, value=0.0)

        return extracted_score_map

    def fill_score_map(
        self,
        score_map: 'ScoreMap',
        value: Union['ScoreMap', np.ndarray, float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        assert score_map.box is None

        if isinstance(value, ScoreMap):
            value = value.mat

        with score_map.writable_context:
            self.fill_np_array(
                score_map.mat,
                value,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def extract_image(self, image: 'Image'):
        extracted_image = self.fill_np_array_internals.bounding_box.extract_image(image)
        extracted_image = extracted_image.copy()

        polygon_mask = Mask.from_shapable(extracted_image)
        shifted_polygon = self.fill_np_array_internals.get_shifted_polygon()
        shifted_polygon.fill_mask(polygon_mask)
        polygon_mask.to_inverted_mask().fill_image(extracted_image, value=0)

        return extracted_image

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        assert image.box is None

        if isinstance(value, Image):
            value = value.mat
        if isinstance(alpha, ScoreMap):
            assert alpha.is_prob
            alpha = alpha.mat

        with image.writable_context:
            self.fill_np_array(image.mat, value, alpha=alpha)

    @staticmethod
    def remove_duplicated_xy_pairs(xy_pairs: Sequence[Tuple[int, int]]):
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

    @staticmethod
    def vatti_clip(polygon: 'Polygon', ratio: float, shrink: bool):
        xy_pairs = polygon.to_xy_pairs()

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
        clipped_path = clipped_paths[0]

        clipped_xy_pairs = Polygon.remove_duplicated_xy_pairs(clipped_path)
        clipped_polygon = Polygon.from_xy_pairs(clipped_xy_pairs)

        return clipped_polygon, distance

    def to_shrank_polygon(self, ratio: float, no_exception: bool = True):
        try:
            shrank_polygon, _ = Polygon.vatti_clip(self, ratio, True)
            return shrank_polygon
        except Exception:
            if no_exception:
                return self
            else:
                logger.exception('Failed to shrink.')
                raise

    def to_dilated_polygon(self, ratio: float, no_exception: bool = True):
        try:
            dilated_polygon, _ = Polygon.vatti_clip(self, ratio, False)
            return dilated_polygon
        except Exception:
            if no_exception:
                return self
            else:
                logger.exception('Failed to dilate.')
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
    shapely_polygons = []
    for polygon in polygons:
        xy_pairs = polygon.to_xy_pairs()
        shapely_polygons.append(ShapelyPolygon(xy_pairs))

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
    mode: FillByElementsMode,
):
    if mode == FillByElementsMode.UNION:
        return None

    polygons_mask = Mask.from_shape(shape)

    with polygons_mask.writable_context:
        for polygon in polygons:
            boxed_mat = polygon.fill_np_array_internals.bounding_box.extract_np_array(
                polygons_mask.mat
            )
            np_polygon_mask = polygon.fill_np_array_internals.get_np_mask()
            np_non_oob_mask = (boxed_mat < 255)
            boxed_mat[np_polygon_mask & np_non_oob_mask] += 1

        if mode == FillByElementsMode.DISTINCT:
            polygons_mask.mat[polygons_mask.mat > 1] = 0

        elif mode == FillByElementsMode.INTERSECT:
            polygons_mask.mat[polygons_mask.mat == 1] = 0

        else:
            raise NotImplementedError()

    return polygons_mask


# Cyclic dependency, by design.
from .point import Point, PointList, PointTuple  # noqa: E402
from .box import Box  # noqa: E402
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
