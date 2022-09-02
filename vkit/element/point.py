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
from typing import Optional, Tuple, Union, List, Sequence, Iterable

import attrs
import numpy as np

from .type import Shapable
from .opt import (
    extract_shape_from_shapable_or_shape,
    clip_val,
    resize_val,
    generate_shape_and_resized_shape,
)

T_VAL = Union[float, str]


@attrs.define
class Point:
    y: int
    x: int

    ###############
    # Constructor #
    ###############
    @staticmethod
    def create(y: T_VAL, x: T_VAL):
        return Point(y=round(float(y)), x=round(float(x)))

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_xy_pair(xy_pair: Tuple[T_VAL, T_VAL]):
        x, y = xy_pair
        return Point.create(y=y, x=x)

    def to_xy_pair(self):
        return (self.x, self.y)

    ############
    # Operator #
    ############
    def copy(self):
        return attrs.evolve(self)

    def to_clipped_point(self, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        height, width = extract_shape_from_shapable_or_shape(shapable_or_shape)
        return Point(
            y=clip_val(self.y, height),
            x=clip_val(self.x, width),
        )

    def to_shifted_point(self, y_offset: int = 0, x_offset: int = 0):
        return Point(y=self.y + y_offset, x=self.x + x_offset)

    def to_conducted_resized_point(
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
        return Point(
            y=resize_val(self.y, height, resized_height),
            x=resize_val(self.x, width, resized_width),
        )


class PointList(List[Point]):

    ###############
    # Constructor #
    ###############
    @staticmethod
    def from_point(point: Point):
        return PointList((point,))

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_xy_pairs(xy_pairs: Iterable[Tuple[T_VAL, T_VAL]]):
        return PointList(Point.from_xy_pair(xy_pair) for xy_pair in xy_pairs)

    def to_xy_pairs(self):
        return [point.to_xy_pair() for point in self]

    @staticmethod
    def from_flatten_xy_pairs(flatten_xy_pairs: Sequence[T_VAL]):
        # [x0, y0, x1, y1, ...]
        flatten_xy_pairs = tuple(flatten_xy_pairs)
        assert flatten_xy_pairs and len(flatten_xy_pairs) % 2 == 0

        points = PointList()
        idx = 0
        while idx < len(flatten_xy_pairs):
            x = flatten_xy_pairs[idx]
            y = flatten_xy_pairs[idx + 1]
            points.append(Point.create(y=y, x=x))
            idx += 2

        return points

    def to_flatten_xy_pairs(self):
        flatten_xy_pairs: List[int] = []
        for point in self:
            flatten_xy_pairs.extend(point.to_xy_pair())
        return flatten_xy_pairs

    @staticmethod
    def from_np_array(np_points: np.ndarray):
        points = PointList()
        for np_point in np_points:
            x, y = np_point
            points.append(Point.create(y=y, x=x))

        if len(points) > 2 and points[0] == points[-1]:
            # Handle the circled duplicated points generated by package like shapely.
            points.pop()

        return points

    def to_np_array(self):
        return np.array(self.to_xy_pairs(), dtype=np.int32)

    ############
    # Operator #
    ############
    def copy(self):
        points = PointList()
        for point in self:
            points.append(point.copy())
        return points

    def to_clipped_points(self, shapable_or_shape: Union[Shapable, Tuple[int, int]]):
        return PointList(point.to_clipped_point(shapable_or_shape) for point in self)

    def to_shifted_points(self, y_offset: int = 0, x_offset: int = 0):
        return PointList(
            point.to_shifted_point(
                y_offset=y_offset,
                x_offset=x_offset,
            ) for point in self
        )

    def to_conducted_resized_points(
        self,
        shapable_or_shape: Union[Shapable, Tuple[int, int]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ):
        return PointList(
            point.to_conducted_resized_point(
                shapable_or_shape=shapable_or_shape,
                resized_height=resized_height,
                resized_width=resized_width,
            ) for point in self
        )
