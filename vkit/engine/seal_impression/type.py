from typing import Sequence
import attrs
from vkit.element import Image, Point


@attrs.define
class CharLayout:
    angle: int
    point_up: Point
    point_down: Point


@attrs.define
class SealImpressionLayout:
    background_image: Image
    char_layouts: Sequence[CharLayout]


@attrs.define
class SealImpressionEngineRunConfig:
    pass


@attrs.define
class SealImpressionEngineResource:
    pass
