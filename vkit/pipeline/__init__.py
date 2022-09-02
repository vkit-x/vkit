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
    PageShapeStepConfig,
    PageShapeStepInput,
    PageShapeStepOutput,
    PageShapeStep,
)
from .text_detection.page_background import (
    page_background_step_factory,
    PageBackgroundStepConfig,
    PageBackgroundStepInput,
    PageBackgroundStepOutput,
    PageBackgroundStep,
)
from .text_detection.page_layout import (
    page_layout_step_factory,
    PageLayoutStepConfig,
    PageLayoutStepInput,
    PageLayout,
    PageLayoutStepOutput,
    PageLayoutStep,
)
from .text_detection.page_image import (
    page_image_step_factory,
    PageImageStepConfig,
    PageImageStepInput,
    PageImageCollection,
    PageImageStepOutput,
    PageImageStep,
)
from .text_detection.page_barcode import (
    page_barcode_step_factory,
    PageBarcodeStepConfig,
    PageBarcodeStepInput,
    PageBarcodeStepOutput,
    PageBarcodeStep,
)
from .text_detection.page_seal_impression import (
    page_seal_impresssion_step_factory,
    PageSealImpresssionStepConfig,
    PageSealImpresssionStepInput,
    PageSealImpresssionStepOutput,
    PageSealImpresssionStep,
)
from .text_detection.page_text_line import (
    page_text_line_step_factory,
    PageTextLineStepConfig,
    PageTextLineStepInput,
    PageTextLineCollection,
    PageTextLineStepOutput,
    PageTextLineStep,
)
from .text_detection.page_non_text_symbol import (
    page_non_text_symbol_step_factory,
    PageNonTextSymbolStepConfig,
    PageNonTextSymbolStepInput,
    PageNonTextSymbolStepOutput,
    PageNonTextSymbolStep,
)
from .text_detection.page_text_line_bounding_box import (
    page_text_line_bounding_box_step_factory,
    PageTextLineBoundingBoxStepConfig,
    PageTextLineBoundingBoxStepInput,
    PageTextLineBoundingBoxStepOutput,
    PageTextLineBoundingBoxStep,
)
from .text_detection.page_text_line_label import (
    page_text_line_label_step_factory,
    PageTextLineLabelStepConfig,
    PageTextLineLabelStepInput,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
    PageTextLineLabelStepOutput,
    PageTextLineLabelStep,
)
from .text_detection.page_assembler import (
    page_assembler_step_factory,
    PageAssemblerStepConfig,
    PageAssemblerStepInput,
    Page,
    PageAssemblerStepOutput,
    PageAssemblerStep,
)
from .text_detection.page_distortion import (
    page_distortion_step_factory,
    PageDistortionStepConfig,
    PageDistortionStepInput,
    PageDistortionStepOutput,
    PageDistortionStep,
)
from .text_detection.page_resizing import (
    page_resizing_step_factory,
    PageResizingStepConfig,
    PageResizingStepInput,
    PageResizingStepOutput,
    PageResizingStep,
)
from .text_detection.page_cropping import (
    page_cropping_step_factory,
    PageCroppingStepConfig,
    PageCroppingStepInput,
    CroppedPage,
    PageCroppingStepOutput,
    PageCroppingStep,
)
from .text_detection.page_text_region import (
    page_text_region_step_factory,
    PageTextRegionStepConfig,
    PageTextRegionStepInput,
    PageTextRegionStepOutput,
    PageTextRegionStep,
)
from .text_detection.page_text_region_label import (
    page_text_region_label_step_factory,
    PageTextRegionLabelStepConfig,
    PageTextRegionLabelStepInput,
    PageCharRegressionLabelTag,
    PageCharRegressionLabel,
    PageTextRegionLabelStepOutput,
    PageTextRegionLabelStep,
)
from .text_detection.page_text_region_cropping import (
    page_text_region_cropping_step_factory,
    PageTextRegionCroppingStepConfig,
    PageTextRegionCroppingStepInput,
    CroppedPageTextRegion,
    PageTextRegionCroppingStepOutput,
    PageTextRegionCroppingStep,
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
        page_text_region_step_factory,
        page_text_region_label_step_factory,
        page_text_region_cropping_step_factory,
    ],
)
