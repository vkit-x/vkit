from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import (
    CharSlot,
    SealImpression,
    SealImpressionEngineRunConfig,
)
from .ellipse import EllipseSealImpressionEngine
from .char_slot_filler import fill_text_line_to_seal_impression

ellipse_seal_impression_factory = EngineFactory(EllipseSealImpressionEngine)

seal_impression_factory = EngineRunnerAggregatorFactory([
    ellipse_seal_impression_factory,
])
