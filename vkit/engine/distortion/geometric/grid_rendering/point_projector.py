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
from typing import Union, Iterable

from vkit.element import Point, PointList, PointTuple


class PointProjector:

    def project_point(self, src_point: Point) -> Point:
        raise NotImplementedError()

    def project_points(self, src_points: Union[PointList, PointTuple, Iterable[Point]]):
        dst_points = PointList()
        for src_point in src_points:
            dst_points.append(self.project_point(src_point))
        return dst_points.to_point_tuple()
