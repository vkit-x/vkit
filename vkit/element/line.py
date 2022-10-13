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
from typing import Tuple, Sequence, Union

import attrs

_T = Union[float, str]


@attrs.define
class Line:
    point_begin: 'Point'
    point_end: 'Point'

    ##############
    # Conversion #
    ##############
    @classmethod
    def from_xy_pairs(cls, xy_pairs: Sequence[Tuple[_T, _T]]):
        assert len(xy_pairs) == 2
        return cls(
            point_begin=Point.from_xy_pair(xy_pairs[0]),
            point_end=Point.from_xy_pair(xy_pairs[1]),
        )

    def to_xy_pairs(self):
        return [self.point_begin.to_xy_pair(), self.point_end.to_xy_pair()]

    @classmethod
    def from_flatten_xy_pairs(cls, flatten_xy_pairs: Sequence[_T]):
        assert len(flatten_xy_pairs) == 4
        x0, y0, x1, y1 = flatten_xy_pairs
        return cls(
            point_begin=Point.create(y=y0, x=x0),
            point_end=Point.create(y=y1, x=x1),
        )

    def to_flatten_xy_pairs(self):
        return [
            self.point_begin.x,
            self.point_begin.y,
            self.point_end.x,
            self.point_end.y,
        ]

    ############
    # Operator #
    ############
    def get_center_point(self):
        return Point.create(
            y=(self.point_begin.y + self.point_end.y) / 2,
            x=(self.point_begin.x + self.point_end.x) / 2,
        )


# Cyclic dependency, by design.
from .point import Point  # noqa: E402
