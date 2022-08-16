from vkit.engine.interface import EngineExecutorAggregatorFactory
from .type import (
    FontEngineRunConfigStyle,
    FontEngineRunConfig,
    FontCollection,
    TextLine,
)
from .freetype import (
    font_freetype_default_engine_executor_factory,
    font_freetype_lcd_engine_executor_factory,
    font_freetype_monochrome_engine_executor_factory,
    FontFreetypeDefaultEngine,
    FontFreetypeLcdEngine,
    FontFreetypeMonochromeEngine,
)

font_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory([
    font_freetype_default_engine_executor_factory,
    font_freetype_lcd_engine_executor_factory,
    font_freetype_monochrome_engine_executor_factory,
])
