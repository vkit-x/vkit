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
from typing import Optional, Tuple, Union, Iterable
import math
import warnings

import attrs
import numpy as np
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import box as build_shapely_polygon_as_box
from shapely.strtree import STRtree

from .type import Shapable, ElementSetOperationMode
from .opt import (
    clip_val,
    resize_val,
    extract_shape_from_shapable_or_shape,
    generate_shape_and_resized_shape,
    fill_np_array,
)

# Shapely version has been explicitly locked under 2.0, hence ignore this warning.
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)


@attrs.define(frozen=True)
class Box(Shapable):
    # By design, smooth positioning is not supported in Box.

    up: int
    down: int
    left: int
    right: int

    ###############
    # Constructor #
    ###############
    @classmethod
    def from_shape(cls, shape: Tuple[int, int]):
        height, width = shape
        return cls(
            up=0,
            down=height - 1,
            left=0,
            right=width - 1,
        )

    @classmethod
    def from_shapable(cls, shapable: Shapable):
        return cls.from_shape(shapable.shape)

    @classmethod
    def from_boxes(cls, boxes: Iterable['Box']):
        # Build a bounding box.
        boxes_iter = iter(boxes)

        first_box = next(boxes_iter)
        up = first_box.up
        down = first_box.down
        left = first_box.left
        right = first_box.right

        for box in boxes_iter:
            up = min(up, box.up)
            down = max(down, box.down)
            left = min(left, box.left)
            right = max(right, box.right)

        return cls(up=up, down=down, left=left, right=right)

    ############
    # Property #
    ############
    @property
    def height(self):
        return self.down + 1 - self.up

    @property
    def width(self):
        return self.right + 1 - self.left

    @property
    def valid(self):
        return (0 <= self.up <= self.down) and (0 <= self.left <= self.right)

    ##############
    # Conversion #
    ##############
    def to_polygon(self, step: Optional[int] = None):
        if self.up == self.down or self.left == self.right:
            raise RuntimeError(f'Cannot convert box={self} to polygon.')

        # NOTE: Up left -> up right -> down right -> down left
        # Char-level labelings are generated based on this ordering.
        if step is None:
            points = PointTuple.from_xy_pairs((
                (self.left, self.up),
                (self.right, self.up),
                (self.right, self.down),
                (self.left, self.down),
            ))

        else:
            assert step > 0

            xs = list(range(self.left, self.right + 1, step))
            if xs[-1] < self.right:
                xs.append(self.right)

            ys = list(range(self.up, self.down + 1, step))
            if ys[-1] == self.down:
                # NOTE: check first to avoid oob error.
                ys.pop()
            ys.pop(0)

            points = PointList()
            # Up.
            for x in xs:
                points.append(Point.create(y=self.up, x=x))
            # Right.
            for y in ys:
                points.append(Point.create(y=y, x=self.right))
            # Down.
            for x in reversed(xs):
                points.append(Point.create(y=self.down, x=x))
            # Left.
            for y in reversed(ys):
                points.append(Point.create(y=y, x=self.left))

        return Polygon.create(points=points)

    def to_shapely_polygon(self):
        return build_shapely_polygon_as_box(
            miny=self.up,
            maxy=self.down,
            minx=self.left,
            maxx=self.right,
        )

    ############
    # Operator #
    ############
    def get_center_point(self):
        return Point.create(y=(self.up + self.down) / 2, x=(self.left + self.right) / 2)

    def to_clipped_box(self, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        height, width = extract_shape_from_shapable_or_shape(shapable_or_shape)
        return Box(
            up=clip_val(self.up, height),
            down=clip_val(self.down, height),
            left=clip_val(self.left, width),
            right=clip_val(self.right, width),
        )

    def to_conducted_resized_box(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        (
            height,
            width,
            resized_height,
            resized_width,
        ) = generate_shape_and_resized_shape(
            shapable_or_shape=shapable_or_shape,
            resized_height=resized_height,
            resized_width=resized_width
        )
        return Box(
            up=round(resize_val(self.up, height, resized_height)),
            down=round(resize_val(self.down, height, resized_height)),
            left=round(resize_val(self.left, width, resized_width)),
            right=round(resize_val(self.right, width, resized_width)),
        )

    def to_resized_box(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return self.to_conducted_resized_box(
            shapable_or_shape=self,
            resized_height=resized_height,
            resized_width=resized_width,
        )

    def to_shifted_box(self, offset_y: int = 0, offset_x: int = 0):
        return Box(
            up=self.up + offset_y,
            down=self.down + offset_y,
            left=self.left + offset_x,
            right=self.right + offset_x,
        )

    def to_relative_box(self, origin_y: int, origin_x: int):
        return self.to_shifted_box(offset_y=-origin_y, offset_x=-origin_x)

    def to_dilated_box(self, ratio: float, clip_long_side: bool = False):
        expand_vert = math.ceil(self.height * ratio / 2)
        expand_hori = math.ceil(self.width * ratio / 2)

        if clip_long_side:
            expand_min = min(expand_vert, expand_hori)
            expand_vert = expand_min
            expand_hori = expand_min

        return Box(
            up=self.up - expand_vert,
            down=self.down + expand_vert,
            left=self.left - expand_hori,
            right=self.right + expand_hori,
        )

    def get_boxes_for_box_attached_opt(self, element_box: Optional['Box']):
        if element_box is None:
            relative_box = self
            new_element_box = None

        else:
            assert element_box.up <= self.up <= self.down <= element_box.down
            assert element_box.left <= self.left <= self.right <= element_box.right

            # NOTE: Some shape, implicitly.
            relative_box = self.to_relative_box(
                origin_y=element_box.up,
                origin_x=element_box.left,
            )
            new_element_box = self

        return relative_box, new_element_box

    def extract_np_array(self, mat: np.ndarray) -> np.ndarray:
        assert 0 <= self.up <= self.down <= mat.shape[0]
        assert 0 <= self.left <= self.right <= mat.shape[1]
        return mat[self.up:self.down + 1, self.left:self.right + 1]

    def extract_mask(self, mask: 'Mask'):
        relative_box, new_mask_box = self.get_boxes_for_box_attached_opt(mask.box)

        if relative_box.shape == mask.shape:
            return mask

        return attrs.evolve(
            mask,
            mat=relative_box.extract_np_array(mask.mat),
            box=new_mask_box,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        relative_box, new_score_map_box = self.get_boxes_for_box_attached_opt(score_map.box)

        if relative_box.shape == score_map.shape:
            return score_map

        return attrs.evolve(
            score_map,
            mat=relative_box.extract_np_array(score_map.mat),
            box=new_score_map_box,
        )

    def extract_image(self, image: 'Image'):
        relative_box, new_image_box = self.get_boxes_for_box_attached_opt(image.box)

        if relative_box.shape == image.shape:
            return image

        return attrs.evolve(
            image,
            mat=relative_box.extract_np_array(image.mat),
            box=new_image_box,
        )

    def prep_mat_and_value(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
    ):
        mat_shape_before_extraction = (mat.shape[0], mat.shape[1])
        if mat_shape_before_extraction != self.shape:
            mat = self.extract_np_array(mat)

        if isinstance(value, np.ndarray):
            value_shape_before_extraction = (value.shape[0], value.shape[1])
            if value_shape_before_extraction != (mat.shape[0], mat.shape[1]):
                assert value_shape_before_extraction == mat_shape_before_extraction
                value = self.extract_np_array(value)

            if value.dtype != mat.dtype:
                value = value.astype(mat.dtype)

        return mat, value

    @classmethod
    def get_np_mask_from_element_mask(cls, element_mask: Optional[Union['Mask', np.ndarray]]):
        np_mask = None
        if element_mask:
            if isinstance(element_mask, Mask):
                # NOTE: Mask.box is ignored.
                np_mask = element_mask.np_mask
            else:
                np_mask = element_mask
        return np_mask

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        np_mask: Optional[np.ndarray] = None,
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        mat, value = self.prep_mat_and_value(mat, value)

        if isinstance(alpha, ScoreMap):
            # NOTE:
            # 1. Place before np_mask to simplify ScoreMap opts.
            # 2. ScoreMap.box is ignored.
            assert alpha.is_prob
            alpha = alpha.mat

        if np_mask is None and isinstance(alpha, np.ndarray):
            # For optimizing sparse alpha matrix.
            np_mask = (alpha > 0.0)

        fill_np_array(
            mat=mat,
            value=value,
            np_mask=np_mask,
            alpha=alpha,
            keep_max_value=keep_max_value,
            keep_min_value=keep_min_value,
        )

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
        mask_mask: Optional[Union['Mask', np.ndarray]] = None,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        relative_box, _ = self.get_boxes_for_box_attached_opt(mask.box)

        if isinstance(value, Mask):
            if value.shape != self.shape:
                value = self.extract_mask(value)
            value = value.mat

        np_mask = self.get_np_mask_from_element_mask(mask_mask)

        with mask.writable_context:
            relative_box.fill_np_array(
                mask.mat,
                value,
                np_mask=np_mask,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_score_map(
        self,
        score_map: 'ScoreMap',
        value: Union['ScoreMap', np.ndarray, float],
        score_map_mask: Optional[Union['Mask', np.ndarray]] = None,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        relative_box, _ = self.get_boxes_for_box_attached_opt(score_map.box)

        if isinstance(value, ScoreMap):
            if value.shape != self.shape:
                value = self.extract_score_map(value)
            value = value.mat

        np_mask = self.get_np_mask_from_element_mask(score_map_mask)

        with score_map.writable_context:
            relative_box.fill_np_array(
                score_map.mat,
                value,
                np_mask=np_mask,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        image_mask: Optional[Union['Mask', np.ndarray]] = None,
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        relative_box, _ = self.get_boxes_for_box_attached_opt(image.box)

        if isinstance(value, Image):
            if value.shape != self.shape:
                value = self.extract_image(value)
            value = value.mat

        np_mask = self.get_np_mask_from_element_mask(image_mask)

        with image.writable_context:
            relative_box.fill_np_array(
                image.mat,
                value,
                np_mask=np_mask,
                alpha=alpha,
            )


class BoxOverlappingValidator:

    def __init__(self, boxes: Iterable[Box]):
        self.strtree = STRtree(box.to_shapely_polygon() for box in boxes)

    def is_overlapped(self, box: Box):
        shapely_polygon = box.to_shapely_polygon()
        for _ in self.strtree.query(shapely_polygon):
            # NOTE: No need to test intersection since the extent of a box is itself.
            return True
        return False


def generate_fill_by_boxes_mask(
    shape: Tuple[int, int],
    boxes: Iterable[Box],
    mode: ElementSetOperationMode,
):
    if mode == ElementSetOperationMode.UNION:
        return None
    else:
        return Mask.from_boxes(shape, boxes, mode)


# Cyclic dependency, by design.
from .point import Point, PointList, PointTuple  # noqa: E402
from .polygon import Polygon  # noqa: E402
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
