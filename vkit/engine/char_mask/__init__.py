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
from ..interface import EngineExecutorAggregatorFactory
from .type import CharMaskEngineRunConfig, CharMask

from .default import (
    char_mask_default_engine_executor_factory,
    CharMaskDefaultEngineInitConfig,
    CharMaskDefaultEngine,
)
from .external_ellipse import (
    char_mask_external_ellipse_engine_executor_factory,
    CharMaskExternalEllipseEngineInitConfig,
    CharMaskExternalEllipseEngine,
)

char_mask_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory([
    char_mask_default_engine_executor_factory,
    char_mask_external_ellipse_engine_executor_factory,
])
