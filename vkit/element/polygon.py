from typing import Optional, Tuple, Union, Sequence, Iterable, Dict, Any, List
import logging
import copy
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

    def get_shifted_polygon(self):
        return Polygon.from_np_array(self.shifted_np_points)

    def get_np_mask(self):
        np_mask = np.zeros(self.bounding_box.shape, dtype=np.uint8)
        cv.fillPoly(np_mask, [self.shifted_np_points], 1)
        np_mask = np_mask.astype(np.bool8)
        return np_mask


@attrs.define
class Polygon:
    points: 'PointList'

    def __attrs_post_init__(self):
        assert self.points

    ###############
    # Constructor #
    ###############
    @staticmethod
    def create(points: Union['PointList', Iterable['Point']]):
        return Polygon(points=PointList(points))

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_xy_pairs(xy_pairs: Iterable[Tuple[T_VAL, T_VAL]]):
        return Polygon(points=PointList.from_xy_pairs(xy_pairs))

    def to_xy_pairs(self):
        return self.points.to_xy_pairs()

    @staticmethod
    def from_flatten_xy_pairs(flatten_xy_pairs: Sequence[T_VAL]):
        return Polygon(points=PointList.from_flatten_xy_pairs(flatten_xy_pairs))

    def to_flatten_xy_pairs(self):
        return self.points.to_flatten_xy_pairs()

    @staticmethod
    def from_np_array(np_points: np.ndarray):
        return Polygon(points=PointList.from_np_array(np_points))

    def to_np_array(self):
        return self.points.to_np_array()

    ############
    # Operator #
    ############
    def copy(self):
        return Polygon(points=self.points.copy())

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

    def to_shifted_points(self, y_offset: int = 0, x_offset: int = 0):
        return self.points.to_shifted_points(y_offset=y_offset, x_offset=x_offset)

    def to_shifted_polygon(self, y_offset: int = 0, x_offset: int = 0):
        return Polygon(points=self.to_shifted_points(y_offset=y_offset, x_offset=x_offset))

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
        internals = self.to_fill_np_array_internals()
        return self.to_conducted_resized_polygon(
            shapable_or_shape=internals.bounding_box.shape,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def to_bounding_rectangular_polygon(self):
        shapely_polygon = ShapelyMultiPoint(self.to_xy_pairs()).minimum_rotated_rectangle
        np_points = np.array(shapely_polygon.exterior.coords).astype(np.int32)  # type: ignore
        points = PointList.from_np_array(np_points)
        assert len(points) == 4
        return self.create(points=points)

    def to_fill_np_array_internals(self):
        np_points = self.to_np_array()

        x_min = np_points[:, 0].min()
        y_min = np_points[:, 1].min()

        x_max = np_points[:, 0].max()
        y_max = np_points[:, 1].max()

        bounding_box = Box(up=y_min, down=y_max, left=x_min, right=x_max)

        np_points[:, 0] -= x_min
        np_points[:, 1] -= y_min

        return PolygonFillNpArrayInternals(
            bounding_box=bounding_box,
            shifted_np_points=np_points,
        )

    def to_bounding_box(self):
        internals = self.to_fill_np_array_internals()
        return internals.bounding_box

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        alpha: Union[np.ndarray, float] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        internals = self.to_fill_np_array_internals()
        mat, value = internals.bounding_box.prep_mat_and_value(mat, value)
        np_mask = internals.get_np_mask()
        fill_np_array(
            mat=mat,
            value=value,
            np_mask=np_mask,
            alpha=alpha,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_mask(self, mask: 'Mask'):
        internals = self.to_fill_np_array_internals()

        extracted_mask = internals.bounding_box.extract_mask(mask)
        extracted_mask = extracted_mask.copy()

        polygon_mask = Mask.from_shapable(extracted_mask)
        shifted_polygon = internals.get_shifted_polygon()
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
        if isinstance(value, Mask):
            value = value.mat

        self.fill_np_array(
            mask.mat,
            value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        internals = self.to_fill_np_array_internals()

        extracted_score_map = internals.bounding_box.extract_score_map(score_map)
        extracted_score_map = extracted_score_map.copy()

        polygon_mask = Mask.from_shapable(extracted_score_map)
        shifted_polygon = internals.get_shifted_polygon()
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
        if isinstance(value, ScoreMap):
            value = value.mat

        self.fill_np_array(
            score_map.mat,
            value,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def extract_image(self, image: 'Image'):
        internals = self.to_fill_np_array_internals()

        extracted_image = internals.bounding_box.extract_image(image)
        extracted_image = extracted_image.copy()

        polygon_mask = Mask.from_shapable(extracted_image)
        shifted_polygon = internals.get_shifted_polygon()
        shifted_polygon.fill_mask(polygon_mask)
        polygon_mask.to_inverted_mask().fill_image(extracted_image, value=0)

        return extracted_image

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        if isinstance(value, Image):
            value = value.mat
        if isinstance(alpha, ScoreMap):
            assert alpha.is_prob
            alpha = alpha.mat

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


@attrs.define
class TextPolygon:
    text: str
    polygon: Polygon
    meta: Optional[Dict[str, Any]] = None

    def __attrs_post_init__(self):
        assert self.text

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(
            self,
            polygon=self.polygon.copy(),
            meta=None if not self.meta else copy.deepcopy(self.meta),
        )

    def to_conducted_resized_text_polygon(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return attrs.evolve(
            self,
            polygon=self.polygon.to_conducted_resized_polygon(
                shapable_or_shape=shapable_or_shape,
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )


def generate_fill_by_polygons_mask(
    shape: Tuple[int, int],
    polygons: Iterable[Polygon],
    mode: FillByElementsMode,
):
    if mode == FillByElementsMode.UNION:
        return None

    polygons_mask = Mask.from_shape(shape)

    for polygon in polygons:
        internals = polygon.to_fill_np_array_internals()
        boxed_mat = internals.bounding_box.extract_np_array(polygons_mask.mat)
        np_polygon_mask = internals.get_np_mask()
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
from .point import Point, PointList  # noqa: E402
from .box import Box  # noqa: E402
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
