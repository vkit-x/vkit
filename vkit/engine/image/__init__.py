from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import ImageEngineRunConfig
from .combiner import (
    CombinerImageEngine,
    CombinerImageEngineConfig,
)
from .selector import (
    SelectorImageEngine,
    SelectorImageEngineConfig,
)

combiner_image_factory = EngineFactory(CombinerImageEngine)
selector_image_factory = EngineFactory(SelectorImageEngine)

image_factory = EngineRunnerAggregatorFactory([
    combiner_image_factory,
    selector_image_factory,
])
