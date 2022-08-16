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
