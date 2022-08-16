from vkit.engine.interface import EngineExecutorAggregatorFactory
from .type import ImageEngineRunConfig
from .combiner import (
    image_combiner_engine_executor_factory,
    ImageCombinerEngineInitConfig,
    ImageCombinerEngine,
)
from .selector import (
    image_selector_engine_executor_factory,
    ImageSelectorEngineInitConfig,
    ImageSelectorEngine,
)

image_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory([
    image_combiner_engine_executor_factory,
    image_selector_engine_executor_factory,
])
