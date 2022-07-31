from typing import Sequence, Tuple, Optional

import attrs
import numpy as np

from vkit.element import Point, Box, Mask


@attrs.define
class CharSlot:
    angle: int
    point_up: Point
    point_down: Point

    @staticmethod
    def build(point_up: Point, point_down: Point):
        theta = np.arctan2(
            point_up.y - point_down.y,
            point_up.x - point_down.x,
        )
        two_pi = 2 * np.pi
        theta = theta % two_pi
        angle = round(theta / two_pi * 360)
        return CharSlot(angle=angle, point_up=point_up, point_down=point_down)


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
