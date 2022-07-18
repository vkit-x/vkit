import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Image, Shapable
from ..interface import (
    PipelineStep,
    NoneTypePipelineStepConfig,
    PipelineStepFactory,
    PipelineState,
)
from .page_layout import PageLayoutStep
from .page_background import PageBackgroundStep
from .page_image import PageImageStep, PageImageCollection
from .page_qrcode import PageQrcodeStep
from .page_barcode import PageBarcodeStep
from .page_text_line import PageTextLineStep, PageTextLineCollection
from .page_text_line_label import (
    PageTextLineLabelStep,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)


@attrs.define
class Page(Shapable):
    image: Image
    page_image_collection: PageImageCollection
    page_text_line_collection: PageTextLineCollection
    page_char_polygon_collection: PageCharPolygonCollection
    page_text_line_polygon_collection: PageTextLinePolygonCollection

    @property
    def height(self):
        return self.image.height

    @property
    def width(self):
        return self.image.width


@attrs.define
class PageAssemblerStepOutput:
    page: Page


class PageAssemblerStep(
    PipelineStep[
        NoneTypePipelineStepConfig,
        PageAssemblerStepOutput,
    ]
):  # yapf: disable

    def run(self, state: PipelineState, rng: RandomGenerator):
        page_layout_step_output = state.get_pipeline_step_output(PageLayoutStep)
        page_layout = page_layout_step_output.page_layout

        page_background_step_output = state.get_pipeline_step_output(PageBackgroundStep)
        background_image = page_background_step_output.background_image

        page_image_step_output = state.get_pipeline_step_output(PageImageStep)
        page_image_collection = page_image_step_output.page_image_collection

        page_qrcode_step_output = state.get_pipeline_step_output(PageQrcodeStep)
        page_barcode_step_output = state.get_pipeline_step_output(PageBarcodeStep)

        page_text_line_step_output = state.get_pipeline_step_output(PageTextLineStep)
        page_text_line_collection = page_text_line_step_output.page_text_line_collection

        page_text_line_label_step_output = state.get_pipeline_step_output(PageTextLineLabelStep)
        page_char_polygon_collection = \
            page_text_line_label_step_output.page_char_polygon_collection
        page_text_line_polygon_collection = \
            page_text_line_label_step_output.page_text_line_polygon_collection

        # Page background.
        assert background_image.mat.shape == (page_layout.height, page_layout.width, 3)
        assembled_image = background_image.copy()

        # Page images.
        for page_image in page_image_collection.page_images:
            page_image.box.fill_image(
                assembled_image,
                page_image.image,
                alpha=page_image.alpha,
            )

        # Page QR codes.
        for qrcode_score_map in page_qrcode_step_output.qrcode_score_maps:
            assembled_image[qrcode_score_map] = (0, 0, 0)
        # Page Bar codes.
        for barcode_score_map in page_barcode_step_output.barcode_score_maps:
            assembled_image[barcode_score_map] = (0, 0, 0)

        # Page text lines.
        for text_line in page_text_line_collection.text_lines:
            # Fill only the masked pixels.
            if text_line.score_map:
                text_line.score_map.fill_image(
                    assembled_image,
                    text_line.glyph_color,
                )
            else:
                text_line.mask.fill_image(
                    assembled_image,
                    text_line.image,
                )

        page = Page(
            image=assembled_image,
            page_image_collection=page_image_collection,
            page_text_line_collection=page_text_line_collection,
            page_char_polygon_collection=page_char_polygon_collection,
            page_text_line_polygon_collection=page_text_line_polygon_collection,
        )
        return PageAssemblerStepOutput(page=page)


page_assembler_step_factory = PipelineStepFactory(PageAssemblerStep)
