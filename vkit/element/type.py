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
from typing import Tuple
from enum import Enum, unique


class Shapable:

    @property
    def height(self) -> int:
        raise NotImplementedError()

    @property
    def width(self) -> int:
        raise NotImplementedError()

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width


@unique
class ElementSetOperationMode(Enum):
    # Active if overlapped with one or more elements.
    UNION = 'union'
    # Active if overlapped with one element.
    DISTINCT = 'distinct'
    # Active if overlapped with more than one elements.
    INTERSECT = 'intersect'
