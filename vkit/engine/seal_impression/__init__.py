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
