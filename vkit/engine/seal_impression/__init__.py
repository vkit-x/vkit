from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import (
    CharSlot,
    SealImpressionLayout,
    SealImpressionEngineRunConfig,
)
from .ellipse import EllipseSealImpressionEngine
from .char_slot_filler import fill_text_line_to_seal_impression_layout

ellipse_seal_impression_engine_factory = EngineFactory(EllipseSealImpressionEngine)

font_factory = EngineRunnerAggregatorFactory([
    ellipse_seal_impression_engine_factory,
])
