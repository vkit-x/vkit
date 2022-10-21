from typing import Sequence, Any

import attrs

from vkit.element import Polygon, ScoreMap


@attrs.define
class CharHeatmapEngineRunConfig:
    height: int
    width: int
    char_polygons: Sequence[Polygon]
    enable_debug: float = False


@attrs.define
class CharHeatmap:
    score_map: ScoreMap
    debug: Any = None
