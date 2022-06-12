from .interface import (
    PipelineStep,
    PipelineStepFactory,
    PipelineStepCollectionFactory,
    PipelineState,
    Pipeline,
    bypass_post_processor_factory,
)

# Text detection.
from .text_detection.page_shape import (
    page_shape_step_factory,
    PageShapeStepConfig,
    PageShapeStepOutput,
)
from .text_detection.page_background import (
    page_background_step_factory,
    PageBackgroundStepConfig,
    PageBackgroundStepOutput,
)
from .text_detection.page_layout import (
    page_layout_step_factory,
    PageLayoutStepConfig,
    PageLayoutStepOutput,
    PageLayout,
)
from .text_detection.page_image import (
    page_image_step_factory,
    PageImageStepConfig,
    PageImageStepOutput,
    PageImageCollection,
)
from .text_detection.page_text_line import (
    page_text_line_step_factory,
    PageTextLineStepConfig,
    PageTextLineStepOutput,
    PageTextLineCollection,
)
from .text_detection.page_text_line_label import (
    page_text_line_label_step_factory,
    PageTextLineLabelStepConfig,
    PageTextLineLabelStepOutput,
    PageTextLinePolygonCollection,
)
from .text_detection.page_assembler import (
    page_assembler_step_factory,
    NoneTypePipelineStepConfig,
    PageAssemblerStepOutput,
    Page,
)
from .text_detection.page_distortion import (
    page_distortion_step_factory,
    PageDistortionStepConfig,
    PageDistortionStepOutput,
)
from .text_detection.page_resizing import (
    page_resizing_step_factory,
    PageResizingStepConfig,
    PageResizingStepOutput,
)
from .text_detection.page_cropping import (
    page_cropping_step_factory,
    PageCroppingStepConfig,
    PageCroppingStepOutput,
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
        page_text_line_step_factory,
        page_text_line_label_step_factory,
        page_assembler_step_factory,
        page_distortion_step_factory,
        page_resizing_step_factory,
        page_cropping_step_factory,
    ],
)
