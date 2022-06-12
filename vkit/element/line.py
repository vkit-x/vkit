from typing import Tuple, Sequence, Union

import attrs

T_VAL = Union[float, str]


@attrs.define
class Line:
    point_begin: 'Point'
    point_end: 'Point'

    ##############
    # Conversion #
    ##############
    @staticmethod
    def from_xy_pairs(xy_pairs: Sequence[Tuple[T_VAL, T_VAL]]):
        assert len(xy_pairs) == 2
        return Line(
            point_begin=Point.from_xy_pair(xy_pairs[0]),
            point_end=Point.from_xy_pair(xy_pairs[1]),
        )

    def to_xy_pairs(self):
        return [self.point_begin.to_xy_pair(), self.point_end.to_xy_pair()]

    @staticmethod
    def from_flatten_xy_pairs(flatten_xy_pairs: Sequence[T_VAL]):
        assert len(flatten_xy_pairs) == 4
        x0, y0, x1, y1 = flatten_xy_pairs
        return Line(
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
        return Point(
            y=(self.point_begin.y + self.point_end.y) // 2,
            x=(self.point_begin.x + self.point_end.x) // 2,
        )


# Cyclic dependency, by design.
from .point import Point  # noqa: E402
