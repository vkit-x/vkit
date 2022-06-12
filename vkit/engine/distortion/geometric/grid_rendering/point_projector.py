from typing import List, Union, Iterable

from vkit.element import Point, PointList


class PointProjector:

    def project_point(self, src_point: Point) -> Point:
        raise NotImplementedError()

    def project_points(self, src_points: Union[PointList, Iterable[Point]]):
        dst_points: List[Point] = []
        for src_point in src_points:
            dst_points.append(self.project_point(src_point))
        return PointList(dst_points)
