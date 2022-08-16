from vkit.engine.interface import EngineExecutorAggregatorFactory
from .type import (
    CharSlot,
    SealImpression,
    SealImpressionEngineRunConfig,
)
from .ellipse import (
    seal_impression_ellipse_engine_executor_factory,
    SealImpressionEllipseEngineInitConfig,
    SealImpressionEllipseEngine,
)
from .text_line_slot_filler import fill_text_line_to_seal_impression

seal_impression_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory([
    seal_impression_ellipse_engine_executor_factory,
])
