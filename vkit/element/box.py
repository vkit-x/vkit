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

import attrs
import numpy as np

from .type import Shapable, FillByElementsMode
from .opt import (
    clip_val,
    resize_val,
    extract_shape_from_shapable_or_shape,
    generate_shape_and_resized_shape,
    fill_np_array,
)


@attrs.define(frozen=True)
class Box(Shapable):
    up: int
    down: int
    left: int
    right: int

    ###############
    # Constructor #
    ###############
    @staticmethod
    def from_shape(height: int, width: int):
        return Box(
            up=0,
            down=height - 1,
            left=0,
            right=width - 1,
        )

    @staticmethod
    def from_shapable(shapable: Shapable):
        return Box.from_shape(height=shapable.height, width=shapable.width)

    @staticmethod
    def from_boxes(boxes: Iterable['Box']):
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

        return Box(up=up, down=down, left=left, right=right)

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
    def area(self):
        return self.height * self.width

    @property
    def valid(self):
        return self.area > 0

    ##############
    # Conversion #
    ##############
    def to_polygon(self, step: Optional[int] = None):
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
                points.append(Point(y=self.up, x=x))
            # Right.
            for y in ys:
                points.append(Point(y=y, x=self.right))
            # Down.
            for x in reversed(xs):
                points.append(Point(y=self.down, x=x))
            # Left.
            for y in reversed(ys):
                points.append(Point(y=y, x=self.left))

        return Polygon.create(points=points)

    ############
    # Operator #
    ############
    def get_center_point(self):
        return Point(y=(self.up + self.down) // 2, x=(self.left + self.right) // 2)

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
            up=resize_val(self.up, height, resized_height),
            down=resize_val(self.down, height, resized_height),
            left=resize_val(self.left, width, resized_width),
            right=resize_val(self.right, width, resized_width),
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

    def to_shifted_box(self, y_offset: int = 0, x_offset: int = 0):
        return Box(
            up=self.up + y_offset,
            down=self.down + y_offset,
            left=self.left + x_offset,
            right=self.right + x_offset,
        )

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

    def extract_np_array(self, mat: np.ndarray) -> np.ndarray:
        assert 0 <= self.up <= self.down <= mat.shape[0]
        assert 0 <= self.left <= self.right <= mat.shape[1]
        return mat[self.up:self.down + 1, self.left:self.right + 1]

    def get_relative_box(self, other: 'Box'):
        # Generate relative box with origin set as (self.left, self.up).
        return other.to_shifted_box(
            y_offset=-self.up,
            x_offset=-self.left,
        )

    def get_boxes_for_extract_opt(self, element_box: Optional['Box']):
        box = self
        new_box = None

        if element_box is not None:
            box = element_box.get_relative_box(self)
            new_box = self

        return box, new_box

    def extract_image(self, image: 'Image'):
        box, new_box = self.get_boxes_for_extract_opt(image.box)
        return attrs.evolve(
            image,
            mat=box.extract_np_array(image.mat),
            box=new_box,
        )

    def extract_mask(self, mask: 'Mask'):
        box, new_box = self.get_boxes_for_extract_opt(mask.box)
        return attrs.evolve(
            mask,
            mat=box.extract_np_array(mask.mat),
            box=new_box,
        )

    def extract_score_map(self, score_map: 'ScoreMap'):
        box, new_box = self.get_boxes_for_extract_opt(score_map.box)
        return attrs.evolve(
            score_map,
            mat=box.extract_np_array(score_map.mat),
            box=new_box,
        )

    def prep_mat_and_value(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
    ):
        mat_shape_before_extraction = (mat.shape[0], mat.shape[1])
        if mat_shape_before_extraction != self.shape:
            mat = self.extract_np_array(mat)

        if not isinstance(value, np.ndarray):
            if mat.ndim == 3:
                num_channels = mat.shape[2]
                if isinstance(value, tuple) and len(value) != num_channels:
                    raise RuntimeError('value is tuple but len(value) != num_channels.')

            value = np.full_like(mat, value)

        else:
            value_shape_before_extraction = (value.shape[0], value.shape[1])
            if value_shape_before_extraction != (mat.shape[0], mat.shape[1]):
                assert value_shape_before_extraction == mat_shape_before_extraction
                value = self.extract_np_array(value)

            if value.dtype != mat.dtype:
                value = value.astype(mat.dtype)

        return mat, value

    def fill_np_array(
        self,
        mat: np.ndarray,
        value: Union[np.ndarray, Tuple[float, ...], float],
        np_mask: Optional[np.ndarray] = None,
        alpha: Union[np.ndarray, float] = 1.0,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        mat, value = self.prep_mat_and_value(mat, value)

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

    def fill_image(
        self,
        image: 'Image',
        value: Union['Image', np.ndarray, Tuple[int, ...], int],
        mask: Optional[Union['Mask', np.ndarray]] = None,
        alpha: Union['ScoreMap', np.ndarray, float] = 1.0,
    ):
        if isinstance(value, Image):
            value = value.mat
        if isinstance(alpha, ScoreMap):
            assert alpha.is_prob
            alpha = alpha.mat
        np_mask = None
        if mask:
            if isinstance(mask, Mask):
                np_mask = mask.np_mask
            else:
                np_mask = mask

        self.fill_np_array(
            image.mat,
            value,
            np_mask=np_mask,
            alpha=alpha,
        )

    def fill_mask(
        self,
        mask: 'Mask',
        value: Union['Mask', np.ndarray, int] = 1,
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        if isinstance(value, Mask):
            value = value.mat

        with mask.writable_context:
            self.fill_np_array(
                mask.mat,
                value,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )

    def fill_score_map(
        self,
        score_map: 'ScoreMap',
        value: Union['ScoreMap', np.ndarray, float],
        keep_max_value: bool = False,
        keep_min_value: bool = False,
    ):
        if isinstance(value, ScoreMap):
            value = value.mat

        with score_map.writable_context:
            self.fill_np_array(
                score_map.mat,
                value,
                keep_max_value=keep_max_value,
                keep_min_value=keep_min_value,
            )


@attrs.define(frozen=True)
class CharBox(Shapable):
    char: str
    box: Box

    def __attrs_post_init__(self):
        assert len(self.char) == 1 and not self.char.isspace()

    ############
    # Property #
    ############
    @property
    def up(self):
        return self.box.up

    @property
    def down(self):
        return self.box.down

    @property
    def left(self):
        return self.box.left

    @property
    def right(self):
        return self.box.right

    @property
    def height(self):
        return self.box.height

    @property
    def width(self):
        return self.box.width

    ############
    # Operator #
    ############
    def to_conducted_resized_char_box(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return attrs.evolve(
            self,
            box=self.box.to_conducted_resized_box(
                shapable_or_shape=shapable_or_shape,
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )

    def to_resized_char_box(
        self,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return attrs.evolve(
            self,
            box=self.box.to_resized_box(
                resized_height=resized_height,
                resized_width=resized_width,
            ),
        )

    def to_shifted_char_box(self, y_offset: int = 0, x_offset: int = 0):
        return attrs.evolve(
            self,
            box=self.box.to_shifted_box(y_offset=y_offset, x_offset=x_offset),
        )


def generate_fill_by_boxes_mask(
    shape: Tuple[int, int],
    boxes: Iterable[Box],
    mode: FillByElementsMode,
):
    if mode == FillByElementsMode.UNION:
        return None

    boxes_mask = Mask.from_shape(shape)

    with boxes_mask.writable_context:
        for box in boxes:
            boxed_mat = box.extract_np_array(boxes_mask.mat)
            np_non_oob_mask = (boxed_mat < 255)
            boxed_mat[np_non_oob_mask] += 1

        if mode == FillByElementsMode.DISTINCT:
            boxes_mask.mat[boxes_mask.mat > 1] = 0

        elif mode == FillByElementsMode.INTERSECT:
            boxes_mask.mat[boxes_mask.mat == 1] = 0

        else:
            raise NotImplementedError()

    return boxes_mask


# Cyclic dependency, by design.
from .point import Point, PointList, PointTuple  # noqa: E402
from .polygon import Polygon  # noqa: E402
from .mask import Mask  # noqa: E402
from .score_map import ScoreMap  # noqa: E402
from .image import Image  # noqa: E402
