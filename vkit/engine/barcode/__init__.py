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
from .type import BarcodeEngineRunConfig

from .code39 import (
    barcode_code39_engine_executor_factory,
    BarcodeCode39EngineInitConfig,
    BarcodeCode39Engine,
)
from .qr import (
    barcode_qr_engine_executor_factory,
    BarcodeQrEngineInitConfig,
    BarcodeQrEngine,
)

barcode_engine_executor_aggregator_factory = EngineExecutorAggregatorFactory([
    barcode_code39_engine_executor_factory,
    barcode_qr_engine_executor_factory,
])
