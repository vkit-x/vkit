import attrs
from numpy.random import Generator as RandomGenerator

from vkit.element import Image, Shapable
from vkit.engine.seal_impression import fill_text_line_to_seal_impression
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
from .page_text_line import (
    PageTextLineStep,
    PageTextLineCollection,
    PageSealImpressionTextLineCollection,
)
from .page_text_line_label import (
    PageTextLineLabelStep,
    PageCharPolygonCollection,
    PageTextLinePolygonCollection,
)
from .page_text_line_bounding_box import PageTextLineBoundingBoxStep


@attrs.define
class Page(Shapable):
    image: Image
    page_image_collection: PageImageCollection
    page_text_line_collection: PageTextLineCollection
    page_seal_impression_text_line_collection: PageSealImpressionTextLineCollection
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
        page_seal_impression_text_line_collection = \
            page_text_line_step_output.page_seal_impression_text_line_collection

        page_text_line_bounding_box_step_output = \
            state.get_pipeline_step_output(PageTextLineBoundingBoxStep)
        text_line_bounding_box_score_maps = page_text_line_bounding_box_step_output.score_maps
        text_line_bounding_box_colors = page_text_line_bounding_box_step_output.colors

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

        # Page text line bounding boxes.
        for text_line_bounding_box_score_map, text_line_bounding_box_color in zip(
            text_line_bounding_box_score_maps, text_line_bounding_box_colors
        ):
            assembled_image[text_line_bounding_box_score_map] = text_line_bounding_box_color

        # Page text lines.
        for text_line in page_text_line_collection.text_lines:
            if text_line.score_map:
                text_line.score_map.fill_image(assembled_image, text_line.glyph_color)
            else:
                text_line.mask.fill_image(assembled_image, text_line.image)

        # Page seal impressions.
        for seal_impression, text_line, box in zip(
            page_seal_impression_text_line_collection.seal_impressions,
            page_seal_impression_text_line_collection.text_lines,
            page_seal_impression_text_line_collection.boxes,
        ):
            alpha = seal_impression.alpha
            color = seal_impression.color
            background_mask = seal_impression.background_mask

            background_mask.to_box_attached(box).fill_image(
                assembled_image,
                color,
                alpha=alpha,
            )

            text_line_filled_score_map = fill_text_line_to_seal_impression(
                seal_impression,
                text_line,
            )
            text_line_filled_score_map.to_box_attached(box).fill_image(
                assembled_image,
                color,
            )

        page = Page(
            image=assembled_image,
            page_image_collection=page_image_collection,
            page_text_line_collection=page_text_line_collection,
            page_seal_impression_text_line_collection=page_seal_impression_text_line_collection,
            page_char_polygon_collection=page_char_polygon_collection,
            page_text_line_polygon_collection=page_text_line_polygon_collection,
        )
        return PageAssemblerStepOutput(page=page)


page_assembler_step_factory = PipelineStepFactory(PageAssemblerStep)
