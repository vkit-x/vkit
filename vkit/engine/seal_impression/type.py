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
from typing import Sequence, Tuple, Optional

import attrs
import numpy as np

from vkit.element import Point, Box, Mask


@attrs.define
class CharSlot:
    angle: int
    point_up: Point
    point_down: Point

    @classmethod
    def build(cls, point_up: Point, point_down: Point):
        theta = np.arctan2(
            point_up.smooth_y - point_down.smooth_y,
            point_up.smooth_x - point_down.smooth_x,
        )
        two_pi = 2 * np.pi
        theta = theta % two_pi
        angle = round(theta / two_pi * 360)
        return cls(angle=angle, point_up=point_up, point_down=point_down)


@attrs.define
class TextLineSlot:
    text_line_height: int
    char_aspect_ratio: float
    char_slots: Sequence[CharSlot]


@attrs.define
class SealImpression:
    alpha: float
    color: Tuple[int, int, int]
    background_mask: Mask
    text_line_slots: Sequence[TextLineSlot]
    internal_text_line_box: Optional[Box]

    @property
    def shape(self):
        return self.background_mask.shape


@attrs.define
class SealImpressionEngineRunConfig:
    height: int
    width: int
