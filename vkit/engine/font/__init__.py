# Copyright 2022 vkit-x Administrator. All Rights Reserved.
#
# This project (vkit-x/vkit) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from vkit.engine.interface import EngineExecutorAggregatorFactory
from .type import (
    FontEngineRunConfigStyle,
    FontEngineRunConfigGlyphSequence,
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
