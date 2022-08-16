from .interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineStepCollectionFactory,
    PipelineState,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    PipelineRunRngStateOutput,
    Pipeline,
)
from .pool import PipelinePool

# Text detection.
from .text_detection.page_shape import (
    page_shape_step_factory,
    PageShapeStep,
    PageShapeStepConfig,
    PageShapeStepOutput,
)
from .text_detection.page_background import (
    page_background_step_factory,
    PageBackgroundStep,
    PageBackgroundStepConfig,
    PageBackgroundStepOutput,
)
from .text_detection.page_layout import (
    page_layout_step_factory,
    PageLayoutStep,
    PageLayoutStepConfig,
    PageLayoutStepOutput,
    PageLayout,
)
from .text_detection.page_image import (
    page_image_step_factory,
    PageImageStep,
    PageImageStepConfig,
    PageImageStepOutput,
    PageImageCollection,
)
from .text_detection.page_barcode import (
    page_barcode_step_factory,
    PageBarcodeStep,
    PageBarcodeStepConfig,
    PageBarcodeStepOutput,
)
from .text_detection.page_seal_impression import (
    page_seal_impresssion_step_factory,
    PageSealImpresssionStep,
    PageSealImpresssionStepConfig,
    PageSealImpresssionStepOutput,
)
from .text_detection.page_text_line import (
    page_text_line_step_factory,
    PageTextLineStep,
    PageTextLineStepConfig,
    PageTextLineStepOutput,
    PageTextLineCollection,
)
from .text_detection.page_non_text_symbol import (
    page_non_text_symbol_step_factory,
    PageNonTextSymbolStep,
    PageNonTextSymbolStepConfig,
    PageNonTextSymbolStepOutput,
)
from .text_detection.page_text_line_bounding_box import (
    page_text_line_bounding_box_step_factory,
    PageTextLineBoundingBoxStep,
    PageTextLineBoundingBoxStepConfig,
    PageTextLineBoundingBoxStepOutput,
)
from .text_detection.page_text_line_label import (
    page_text_line_label_step_factory,
    PageTextLineLabelStep,
    PageTextLineLabelStepConfig,
    PageTextLineLabelStepOutput,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)
from .text_detection.page_assembler import (
    page_assembler_step_factory,
    PageAssemblerStep,
    PageAssemblerStepOutput,
    Page,
)
from .text_detection.page_distortion import (
    page_distortion_step_factory,
    PageDistortionStep,
    PageDistortionStepConfig,
    PageDistortionStepOutput,
)
from .text_detection.page_resizing import (
    page_resizing_step_factory,
    PageResizingStep,
    PageResizingStepConfig,
    PageResizingStepOutput,
)
from .text_detection.page_cropping import (
    page_cropping_step_factory,
    PageCroppingStep,
    PageCroppingStepConfig,
    PageCroppingStepOutput,
    CroppedPage,
)

# Registry.
pipeline_step_collection_factory = PipelineStepCollectionFactory()

pipeline_step_collection_factory.register_step_factories(
    'text_detection',
    [
        page_shape_step_factory,
        page_background_step_factory,
        page_layout_step_factory,
        page_image_step_factory,
        page_barcode_step_factory,
        page_seal_impresssion_step_factory,
        page_text_line_step_factory,
        page_non_text_symbol_step_factory,
        page_text_line_bounding_box_step_factory,
        page_text_line_label_step_factory,
        page_assembler_step_factory,
        page_distortion_step_factory,
        page_resizing_step_factory,
        page_cropping_step_factory,
    ],
)
