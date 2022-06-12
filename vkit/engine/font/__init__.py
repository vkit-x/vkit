from vkit.engine.interface import EngineFactory, EngineRunnerAggregatorFactory
from .type import FontEngineRunConfigStyle, FontEngineRunConfig, FontCollection, TextLine
from .freetype import (
    FreetypeDefaultFontEngine,
    FreetypeLcdFontEngine,
    FreetypeMonochromeFontEngine,
)

freetype_default_font_factory = EngineFactory(FreetypeDefaultFontEngine)
freetype_lcd_font_factory = EngineFactory(FreetypeLcdFontEngine)
freetype_monochrome_font_factory = EngineFactory(FreetypeMonochromeFontEngine)

font_factory = EngineRunnerAggregatorFactory([
    freetype_default_font_factory,
    freetype_lcd_font_factory,
    freetype_monochrome_font_factory,
])
