from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import (
    CharSlot,
    SealImpressionLayout,
    SealImpressionEngineRunConfig,
)
from .ellipse import EllipseSealImpressionEngine

ellipse_seal_impression_engine_factory = EngineFactory(EllipseSealImpressionEngine)

font_factory = EngineRunnerAggregatorFactory([
    ellipse_seal_impression_engine_factory,
])
